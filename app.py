import joblib
import pandas as pd
import streamlit as st

from utils.explanations import explain_transaction


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    data = joblib.load("fraud_model.pkl")

    return data["model"], data["features"]


model, features = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ FraudGuard AI")

st.subheader(
    "Real-Time Fraud & Anomaly Detection Engine"
)

st.write(
    "Analyze financial transactions and identify "
    "potential fraudulent behavior using machine learning."
)

st.divider()


# --------------------------------------------------
# TRANSACTION INPUT
# --------------------------------------------------

st.subheader("💳 Transaction Details")


col1, col2 = st.columns(2)


with col1:

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=5000.0,
        step=100.0
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=14
    )

    transaction_frequency = st.number_input(
        "Recent Transaction Frequency",
        min_value=0,
        value=3,
        step=1
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=1,
        value=500,
        step=1
    )


with col2:

    location_change = st.selectbox(
        "Location Changed?",
        ["No", "Yes"]
    )

    time_since_last_transaction = st.number_input(
        "Minutes Since Last Transaction",
        min_value=0.1,
        value=60.0,
        step=1.0
    )

    device_change = st.selectbox(
        "Device Changed?",
        ["No", "Yes"]
    )


# Convert Yes/No into model-compatible values

location_change_value = (
    1 if location_change == "Yes" else 0
)

device_change_value = (
    1 if device_change == "Yes" else 0
)


# --------------------------------------------------
# CREATE TRANSACTION
# --------------------------------------------------

transaction = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "transaction_frequency": transaction_frequency,
    "account_age_days": account_age_days,
    "location_change": location_change_value,
    "time_since_last_transaction": time_since_last_transaction,
    "device_change": device_change_value
}])


st.divider()


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Analyze Transaction",
    type="primary",
    use_container_width=True
):

    # Get fraud probability

    probability = model.predict_proba(
        transaction[features]
    )[0][1]

    risk_score = probability * 100


    # Get explanation

    reasons = explain_transaction(
        transaction.iloc[0]
    )


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("🚨 Detection Result")


    if risk_score >= 50:

        st.error(
            f"⚠️ HIGH RISK TRANSACTION\n\n"
            f"Fraud Risk Score: {risk_score:.2f}%"
        )

        decision = "BLOCK"

    else:

        st.success(
            f"✅ LOW RISK TRANSACTION\n\n"
            f"Fraud Risk Score: {risk_score:.2f}%"
        )

        decision = "ALLOW"


    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Fraud Risk",
            f"{risk_score:.1f}%"
        )


    with metric2:

        st.metric(
            "Decision",
            decision
        )


    with metric3:

        if risk_score >= 50:

            risk_level = "HIGH"

        elif risk_score >= 25:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        st.metric(
            "Risk Level",
            risk_level
        )


    # --------------------------------------------------
    # RISK BAR
    # --------------------------------------------------

    st.subheader("📊 Risk Analysis")

    st.progress(
        min(int(risk_score), 100)
    )


    # --------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------

    st.subheader(
        "🔎 Why was this transaction flagged?"
    )


    for reason in reasons:

        st.write(
            f"• {reason}"
        )


    # --------------------------------------------------
    # TRANSACTION DATA
    # --------------------------------------------------

    with st.expander(
        "View Transaction Features"
    ):

        st.dataframe(
            transaction,
            use_container_width=True
        )