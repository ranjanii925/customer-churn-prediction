import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Trained Model
MODEL_PATH = Path(__file__).parent / "churn_prediction.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# Application Header
st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    ### Predict Customer Churn Using Machine Learning

    Enter the customer's details below to predict whether
    the customer is likely to leave the telecom service.
    """
)

st.divider()

# Customer Input Form
st.subheader("👤 Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

        st.markdown("### Phone Services")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        st.markdown("### Internet Services")

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        st.markdown("### Additional Services")

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        st.markdown("### Contract Information")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        st.markdown("### Billing Information")

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        type="primary",
        use_container_width=True
    )


# Prediction Section
if submitted:

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        # Apply the same encoding used during training
        customer_encoded = pd.get_dummies(
            customer_data,
            drop_first=True,
            dtype=int
        )

        # Match the training columns and their order
        if not hasattr(model, "feature_names_in_"):
            st.error(
                "Model feature names are unavailable. "
                "Please check the model training code."
            )
            st.stop()

        customer_encoded = customer_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Predict churn
        prediction = model.predict(customer_encoded)[0]

        st.divider()
        st.header("📈 Prediction Result")

        if prediction == "Yes":
            st.error(
                "⚠️ Customer Churn Predicted: "
                "This customer may leave the service."
            )
        else:
            st.success(
                "✅ Customer Retention Predicted: "
                "This customer may continue using the service."
            )

        # Display churn probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer_encoded)[0]
            classes = list(model.classes_)

            if "Yes" in classes:
                churn_probability = (
                    probabilities[classes.index("Yes")] * 100
                )

                st.metric(
                    "Estimated Churn Probability",
                    f"{churn_probability:.2f}%"
                )

                st.progress(
                    min(max(float(churn_probability) / 100, 0.0), 1.0)
                )

        with st.expander("View Customer Details"):
            st.dataframe(
                customer_data,
                use_container_width=True
            )

        st.caption(
            "Note: Predictions are estimates from a Machine Learning "
            "model and are not guarantees of future customer behaviour."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")


# Footer
st.divider()

st.caption(
    "Customer Churn Prediction | Logistic Regression | "
    "Machine Learning Project"
)