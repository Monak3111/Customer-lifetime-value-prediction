import pandas as pd

df = pd.read_csv("clv.csv")
df["purchase_history"] = pd.to_datetime(df["purchase_history"])

from lifetimes.utils import summary_data_from_transaction_data

summary = summary_data_from_transaction_data(
    df,
    customer_id_col="customer_id",
    datetime_col="purchase_history",
    monetary_value_col="total_spent",
    observation_period_end=df["purchase_history"].max()
)

print(summary.head())

import pandas as pd
from lifetimes import BetaGeoFitter

# Load your data into a DataFrame (df)
# Assuming 'df' contains the table you provided

summary = pd.DataFrame()
summary['frequency'] = df['purchase_history'] - 1
summary['tenure'] = df['tenure']
# We assume recency = tenure for this fit if not explicitly provided
summary['recency'] = df['tenure'] 

# Penalizer_coef is the secret to avoiding convergence errors.
# It adds L2 regularization to the likelihood function.
bgf = BetaGeoFitter(penalizer_coef=0.01)

bgf.fit(summary['frequency'], summary['recency'], summary['tenure'])

print(bgf)

import pandas as pd
from lifetimes import BetaGeoFitter

# Load your data into a DataFrame (df)
# Assuming 'df' contains the table you provided

summary = pd.DataFrame()
summary['frequency'] = df['purchase_history'] - 1
summary['tenure'] = df['tenure']
# We assume recency = tenure for this fit if not explicitly provided
summary['recency'] = df['tenure'] 

# Penalizer_coef is the secret to avoiding convergence errors.
# It adds L2 regularization to the likelihood function.
bgf = BetaGeoFitter(penalizer_coef=0.01)

bgf.fit(summary['frequency'], summary['recency'], summary['tenure'])

print(bgf)

summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    30,  # next 30 days
    summary["frequency"],
    summary["recency"],
    summary["tenure"]
)

print(summary[['predicted_purchases']].head())

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Example: your data
data = pd.DataFrame({
    'customer_id': [1,2,3],
    'tenure': [365, 200, 100],  # in days
    'purchase_history': [5, 3, 2],  # number of transactions
    'total_spent': [500, 300, 200]
})

# frequency = number of repeat purchases (frequency ≥1)
# monetary_value = total_spent / frequency
data['frequency'] = data['purchase_history'] - 1  # repeat purchases
data['monetary_value'] = data['total_spent'] / data['purchase_history']
data['T'] = data['tenure']  # time observed (days)

df['frequency'] = df['purchase_history'] - 1

import numpy as np

df['recency'] = np.where(
    df['frequency'] > 0,
    df['tenure'] * df['frequency'] / df['purchase_history'],
    0
)

df['T'] = df['tenure']

df['monetary_value'] = df['total_spent'] / df['purchase_history']

from lifetimes import GammaGammaFitter

# Initialize model
ggf = GammaGammaFitter(penalizer_coef=0)

# Fit the model
ggf.fit(df['frequency'], df['monetary_value'])

# Predict average value per transaction
df['predicted_avg_value'] = ggf.conditional_expected_average_profit(
    df['frequency'], df['monetary_value']
)

print(df[['customer_id', 'predicted_avg_value']].head())

import pandas as pd
from lifetimes import GammaGammaFitter

# Load CSV
df = pd.read_csv("CLV.csv")

# Rename columns for lifetimes
df = df.rename(columns={
    'purchase_history': 'frequency',
    'total_spent': 'monetary_value'
})

# Calculate average monetary value per transaction
df['monetary_value'] = df['monetary_value'] / df['frequency']

import pandas as pd
from lifetimes import GammaGammaFitter

# Load CSV
df = pd.read_csv("CLV.csv")

# Rename columns for lifetimes
df = df.rename(columns={
    'purchase_history': 'frequency',
    'total_spent': 'monetary_value'
})

# Calculate average monetary value per transaction
df['monetary_value'] = df['monetary_value'] / df['frequency']

# Fit the Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(df['frequency'], df['monetary_value'])

df['predicted_avg_value'] = ggf.conditional_expected_average_profit(
    df['frequency'], df['monetary_value']
)

prediction_horizon_months = 6  # next 6 months
df['predicted_transactions_6m'] = df['frequency'] / df['tenure'] * prediction_horizon_months

df['predicted_clv_6m'] = df['predicted_avg_value'] * df['predicted_transactions_6m']

print(df[['customer_id', 'predicted_avg_value', 'predicted_transactions_6m', 'predicted_clv_6m']].head())

pip install streamlit pandas lifetimes

import streamlit as st

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

import streamlit as st
import pandas as pd
from lifetimes import GammaGammaFitter

st.title("Customer Lifetime Value Predictor (Gamma-Gamma Model)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_cols = ['customer_id', 'tenure', 'purchase_history', 'total_spent']
    if all(col in df.columns for col in required_cols):

        # Prepare data
        df = df.rename(columns={
            'purchase_history': 'frequency',
            'total_spent': 'monetary_value'
        })
        df['monetary_value'] = df['monetary_value'] / df['frequency']

        # Fit Gamma-Gamma model
        ggf = GammaGammaFitter(penalizer_coef=0.1)
        ggf.fit(df['frequency'], df['monetary_value'])

        # Predict average transaction value
        df['predicted_avg_value'] = ggf.conditional_expected_average_profit(
            df['frequency'], df['monetary_value']
        )

        # Predict transactions for next 6 months (approximation)
        prediction_horizon_months = 6
        df['predicted_transactions_6m'] = df['frequency'] / df['tenure'] * prediction_horizon_months

        # Compute CLV
        df['predicted_clv_6m'] = df['predicted_avg_value'] * df['predicted_transactions_6m']

        st.subheader("Predicted CLV")
        st.dataframe(df[['customer_id', 'predicted_avg_value', 'predicted_transactions_6m', 'predicted_clv_6m']])

        # Optionally download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predicted_clv.csv',
            mime='text/csv'
        )
    else:
        st.error(f"CSV must contain columns: {required_cols}")
