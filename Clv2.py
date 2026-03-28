{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42bb2e0d-5b23-49a3-9d87-2641e126cd94",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv(\"clv.csv\")\n",
    "df[\"purchase_history\"] = pd.to_datetime(df[\"purchase_history\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf6083a1-0e9e-4da3-8173-cf1c2bb3a5a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from lifetimes.utils import summary_data_from_transaction_data\n",
    "\n",
    "summary = summary_data_from_transaction_data(\n",
    "    df,\n",
    "    customer_id_col=\"customer_id\",\n",
    "    datetime_col=\"purchase_history\",\n",
    "    monetary_value_col=\"total_spent\",\n",
    "    observation_period_end=df[\"purchase_history\"].max()\n",
    ")\n",
    "\n",
    "print(summary.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b5d14f9-3a28-4ee8-9b3c-10fdd71a1813",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from lifetimes import BetaGeoFitter\n",
    "\n",
    "# Load your data into a DataFrame (df)\n",
    "# Assuming 'df' contains the table you provided\n",
    "\n",
    "summary = pd.DataFrame()\n",
    "summary['frequency'] = df['purchase_history'] - 1\n",
    "summary['tenure'] = df['tenure']\n",
    "# We assume recency = tenure for this fit if not explicitly provided\n",
    "summary['recency'] = df['tenure'] \n",
    "\n",
    "# Penalizer_coef is the secret to avoiding convergence errors.\n",
    "# It adds L2 regularization to the likelihood function.\n",
    "bgf = BetaGeoFitter(penalizer_coef=0.01)\n",
    "\n",
    "bgf.fit(summary['frequency'], summary['recency'], summary['tenure'])\n",
    "\n",
    "print(bgf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9a14567-4433-450e-a650-58a219e0bc2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "bgf = BetaGeoFitter(penalizer_coef=0.01)\n",
    "\n",
    "bgf.fit(summary['frequency'], summary['recency'], summary['tenure'])\n",
    "\n",
    "print(bgf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7662c4b-895f-41d5-b469-7d67bcf7210a",
   "metadata": {},
   "outputs": [],
   "source": [
    "summary[\"predicted_purchases\"] = bgf.conditional_expected_number_of_purchases_up_to_time(\n",
    "    30,  # next 30 days\n",
    "    summary[\"frequency\"],\n",
    "    summary[\"recency\"],\n",
    "    summary[\"tenure\"]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c4eb4fb-57e3-4794-aebe-668ed438a0cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(summary[['predicted_purchases']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f3cb51e-0d1e-4a47-823e-a0dedbec557c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from lifetimes import BetaGeoFitter, GammaGammaFitter\n",
    "\n",
    "# Example: your data\n",
    "data = pd.DataFrame({\n",
    "    'customer_id': [1,2,3],\n",
    "    'tenure': [365, 200, 100],  # in days\n",
    "    'purchase_history': [5, 3, 2],  # number of transactions\n",
    "    'total_spent': [500, 300, 200]\n",
    "})\n",
    "\n",
    "# frequency = number of repeat purchases (frequency ≥1)\n",
    "# monetary_value = total_spent / frequency\n",
    "data['frequency'] = data['purchase_history'] - 1  # repeat purchases\n",
    "data['monetary_value'] = data['total_spent'] / data['purchase_history']\n",
    "data['T'] = data['tenure']  # time observed (days)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d83525c-b9f1-4493-a99c-128fc55d759a",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d1cdc79-fc2f-45b9-aa2d-554fdbab29d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['frequency'] = df['purchase_history'] - 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a11e820-98c7-4c64-a65a-7ce7acc6a809",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "df['recency'] = np.where(\n",
    "    df['frequency'] > 0,\n",
    "    df['tenure'] * df['frequency'] / df['purchase_history'],\n",
    "    0\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df1553a8-ebe4-488f-955c-27bdafa83cfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['T'] = df['tenure']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ff51a6b-2f34-405a-ab70-fbce057f2062",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['monetary_value'] = df['total_spent'] / df['purchase_history']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64da6dac-340f-4cdd-848a-3d09a7939955",
   "metadata": {},
   "outputs": [],
   "source": [
    "from lifetimes import GammaGammaFitter\n",
    "\n",
    "# Initialize model\n",
    "ggf = GammaGammaFitter(penalizer_coef=0)\n",
    "\n",
    "# Fit the model\n",
    "ggf.fit(df['frequency'], df['monetary_value'])\n",
    "\n",
    "# Predict average value per transaction\n",
    "df['predicted_avg_value'] = ggf.conditional_expected_average_profit(\n",
    "    df['frequency'], df['monetary_value']\n",
    ")\n",
    "\n",
    "print(df[['customer_id', 'predicted_avg_value']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ceaf026-d92b-48d7-8fed-65b812abcf83",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from lifetimes import GammaGammaFitter\n",
    "\n",
    "# Load CSV\n",
    "df = pd.read_csv(\"CLV.csv\")\n",
    "\n",
    "# Rename columns for lifetimes\n",
    "df = df.rename(columns={\n",
    "    'purchase_history': 'frequency',\n",
    "    'total_spent': 'monetary_value'\n",
    "})\n",
    "\n",
    "# Calculate average monetary value per transaction\n",
    "df['monetary_value'] = df['monetary_value'] / df['frequency']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29b84e30-0a4e-4315-ae33-a2660f2ed497",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit the Gamma-Gamma model\n",
    "ggf = GammaGammaFitter(penalizer_coef=0.1)\n",
    "ggf.fit(df['frequency'], df['monetary_value'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bc14921-bb67-46d2-b796-82c189aae5c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['predicted_avg_value'] = ggf.conditional_expected_average_profit(\n",
    "    df['frequency'], df['monetary_value']\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9785a70e-5b64-48cb-a7e3-bbe91acede4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction_horizon_months = 6  # next 6 months\n",
    "df['predicted_transactions_6m'] = df['frequency'] / df['tenure'] * prediction_horizon_months"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a870e60-34f5-490e-97ee-c1233769643b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['predicted_clv_6m'] = df['predicted_avg_value'] * df['predicted_transactions_6m']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e50348e0-6c26-487b-8725-e3ac4521bcd5",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df[['customer_id', 'predicted_avg_value', 'predicted_transactions_6m', 'predicted_clv_6m']].head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "535736bc-3967-45a5-8611-6d0e9699488b",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install streamlit pandas lifetimes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "267e759b-1d47-4179-bc13-061e30c05ea7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload your CSV file\", type=[\"csv\"])\n",
    "if uploaded_file:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    st.write(\"Data preview:\")\n",
    "    st.dataframe(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5dabfdbf-21e3-4732-ab8a-892c5bdfaee5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from lifetimes import GammaGammaFitter\n",
    "\n",
    "st.title(\"Customer Lifetime Value Predictor (Gamma-Gamma Model)\")\n",
    "\n",
    "# Upload CSV\n",
    "uploaded_file = st.file_uploader(\"Upload your CSV file\", type=[\"csv\"])\n",
    "if uploaded_file:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "\n",
    "    st.subheader(\"Data Preview\")\n",
    "    st.dataframe(df.head())\n",
    "\n",
    "    # Ensure required columns exist\n",
    "    required_cols = ['customer_id', 'tenure', 'purchase_history', 'total_spent']\n",
    "    if all(col in df.columns for col in required_cols):\n",
    "\n",
    "        # Prepare data\n",
    "        df = df.rename(columns={\n",
    "            'purchase_history': 'frequency',\n",
    "            'total_spent': 'monetary_value'\n",
    "        })\n",
    "        df['monetary_value'] = df['monetary_value'] / df['frequency']\n",
    "\n",
    "        # Fit Gamma-Gamma model\n",
    "        ggf = GammaGammaFitter(penalizer_coef=0.1)\n",
    "        ggf.fit(df['frequency'], df['monetary_value'])\n",
    "\n",
    "        # Predict average transaction value\n",
    "        df['predicted_avg_value'] = ggf.conditional_expected_average_profit(\n",
    "            df['frequency'], df['monetary_value']\n",
    "        )\n",
    "\n",
    "        # Predict transactions for next 6 months (approximation)\n",
    "        prediction_horizon_months = 6\n",
    "        df['predicted_transactions_6m'] = df['frequency'] / df['tenure'] * prediction_horizon_months\n",
    "\n",
    "        # Compute CLV\n",
    "        df['predicted_clv_6m'] = df['predicted_avg_value'] * df['predicted_transactions_6m']\n",
    "\n",
    "        st.subheader(\"Predicted CLV\")\n",
    "        st.dataframe(df[['customer_id', 'predicted_avg_value', 'predicted_transactions_6m', 'predicted_clv_6m']])\n",
    "\n",
    "        # Optionally download predictions\n",
    "        csv = df.to_csv(index=False).encode('utf-8')\n",
    "        st.download_button(\n",
    "            label=\"Download Predictions as CSV\",\n",
    "            data=csv,\n",
    "            file_name='predicted_clv.csv',\n",
    "            mime='text/csv'\n",
    "        )\n",
    "    else:\n",
    "        st.error(f\"CSV must contain columns: {required_cols}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26ab564d-a052-468d-ad44-5cdf5085c926",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53bb8a6a-d8dd-49a0-859b-f008d071f402",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
