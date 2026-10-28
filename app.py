import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

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

# Analytical Dashboard
st.header("📊 Customer Analytics Dashboard")
st.caption(
    "Model evaluation results are from the held-out test set. "
    "Customer distribution charts use the public Telco churn CSV when it is available."
)

metric_cols = st.columns(4)
metric_cols[0].metric("Model Accuracy", "80.48%")
metric_cols[1].metric("Churn Precision", "65.62%")
metric_cols[2].metric("Churn Recall", "55.61%")
metric_cols[3].metric("Churn F1-score", "60.20%")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Baseline vs Logistic Regression")
    performance_df = pd.DataFrame({
        "Model": ["Dummy Baseline", "Logistic Regression"],
        "Accuracy (%)": [73.46, 80.48],
    })
    fig_performance = px.bar(
        performance_df,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        title="Accuracy Comparison",
        range_y=[0, 100],
    )
    fig_performance.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_performance, use_container_width=True)

with chart_col2:
    st.subheader("Confusion Matrix")
    # Rows: actual [No, Yes]; columns: predicted [No, Yes].
    confusion_values = [[926, 109], [166, 208]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=confusion_values,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        text=confusion_values,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig_cm.update_layout(title="Test Set Predictions", xaxis_title="Prediction", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

# Optional dataset-driven charts. Keep the CSV beside app.py in the deployed repo.
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if DATA_PATH.exists():
    try:
        dashboard_df = pd.read_csv(DATA_PATH)
        if "Churn" in dashboard_df.columns:
            st.subheader("Customer Churn Distribution")
            churn_counts = (
                dashboard_df["Churn"]
                .value_counts()
                .rename_axis("Churn Status")
                .reset_index(name="Customers")
            )
            st.plotly_chart(
                px.pie(churn_counts, names="Churn Status", values="Customers", hole=0.45,
                       title="Customers Who Stayed vs Churned"),
                use_container_width=True,
            )

            if "Contract" in dashboard_df.columns:
                st.subheader("Churn by Contract Type")
                contract_churn = (
                    dashboard_df.groupby(["Contract", "Churn"]).size()
                    .reset_index(name="Customers")
                )
                st.plotly_chart(
                    px.bar(contract_churn, x="Contract", y="Customers", color="Churn",
                           barmode="group", title="Customer Churn Across Contract Types"),
                    use_container_width=True,
                )
    except Exception as error:
        st.warning(f"Could not load dashboard dataset: {error}")
else:
    st.info(
        "To show customer distribution and contract charts, add "
        "'WA_Fn-UseC_-Telco-Customer-Churn.csv' to the same GitHub repository as app.py."
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

        # Predict churn using threshold 0.30
        probabilities = model.predict_proba(customer_encoded)[0]
        classes = list(model.classes_)

        churn_probability = probabilities[classes.index("Yes")]

        if churn_probability >= 0.30:
            prediction = "Yes"
        else:
            prediction = "No"

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
