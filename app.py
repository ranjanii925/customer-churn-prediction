import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st




st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide"
)
```python
st.markdown("""
<style>
    .stApp {
        background-color: #111827;
        color: #F9FAFB;
    }

    [data-testid="stMetric"] {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 10px;
    }

    h1, h2, h3, p, label {
        color: #F9FAFB;
    }
</style>
""", unsafe_allow_html=True)
```


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_prediction.pkl"
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# A lower threshold helps identify more potential churn customers.
CHURN_THRESHOLD = 0.30


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()



st.title("📈 Customer Churn Prediction")

st.markdown(
    """
    Predict which telecom customers may leave the service
    using a Machine Learning model.

    Explore customer behaviour, review model performance,
    and check the churn probability for an individual customer.
    """
)

st.divider()


st.header("💹Customer Analytics Dashboard")

st.caption(
    "The model metrics below are from the held-out test set "
    "using a churn prediction threshold of 0.30."
)

# These values come from the previously evaluated test results.
metric_cols = st.columns(4)

metric_cols[0].metric("Model Accuracy", "75.30%")
metric_cols[1].metric("Churn Precision", "52.40%")
metric_cols[2].metric("Churn Recall", "75.94%")
metric_cols[3].metric("Churn F1-score", "62.01%")

st.divider()



chart_col1, chart_col2 = st.columns(2)


# Model accuracy comparison
with chart_col1:

    st.subheader("Model Performance Comparison")

    performance_df = pd.DataFrame({
        "Model": [
            "Dummy Baseline",
            "Logistic Regression"
        ],
        "Accuracy (%)": [
            73.46,
            75.30
        ]
    })

    fig_performance = px.bar(
        performance_df,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        color="Model",
        title="Baseline vs Logistic Regression",
        range_y=[0, 100]
    )

    fig_performance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_performance.update_layout(
        yaxis_title="Accuracy (%)",
        xaxis_title="Model",
        showlegend=False
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )


# Confusion matrix for threshold 0.30
with chart_col2:

    st.subheader("Confusion Matrix")

    confusion_values = [
        [777, 258],
        [90, 284]
    ]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=confusion_values,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            text=confusion_values,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True
        )
    )

    fig_cm.update_layout(
        title="Confusion Matrix (Threshold = 0.30)",
        xaxis_title="Predicted Churn",
        yaxis_title="Actual Churn"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


dashboard_df = None

if DATA_PATH.exists():

    try:
        dashboard_df = pd.read_csv(DATA_PATH)

    except Exception as error:
        st.warning(f"Unable to read the customer dataset: {error}")

else:
    st.info(
        "Customer analysis charts require "
        "'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
        "in the same repository as app.py."
    )



if dashboard_df is not None and "Churn" in dashboard_df.columns:

    st.divider()
    st.subheader("Customer Churn Distribution")

    churn_counts = (
        dashboard_df["Churn"]
        .value_counts()
        .rename_axis("Churn Status")
        .reset_index(name="Customers")
    )

    total_customers = len(dashboard_df)

    churned_customers = int(
        (dashboard_df["Churn"] == "Yes").sum()
    )

    retained_customers = int(
        (dashboard_df["Churn"] == "No").sum()
    )

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers > 0
        else 0
    )

    # Display dataset-level customer statistics.
    customer_cols = st.columns(3)

    customer_cols[0].metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    customer_cols[1].metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    customer_cols[2].metric(
        "Customer Churn Rate",
        f"{churn_rate:.2f}%"
    )

    churn_col1, churn_col2 = st.columns(2)

    # Customer distribution chart
    with churn_col1:

        fig_churn = px.pie(
            churn_counts,
            names="Churn Status",
            values="Customers",
            hole=0.45,
            title="Customers Who Stayed vs Churned"
        )

        st.plotly_chart(
            fig_churn,
            use_container_width=True
        )

    # Customer count chart
    with churn_col2:

        fig_customer_count = px.bar(
            churn_counts,
            x="Churn Status",
            y="Customers",
            color="Churn Status",
            text="Customers",
            title="Customer Count by Churn Status"
        )

        fig_customer_count.update_traces(
            textposition="outside"
        )

        fig_customer_count.update_layout(
            xaxis_title="Churn Status",
            yaxis_title="Number of Customers",
            showlegend=False
        )

        st.plotly_chart(
            fig_customer_count,
            use_container_width=True
        )



if (
    dashboard_df is not None
    and "Churn" in dashboard_df.columns
    and "Contract" in dashboard_df.columns
):

    st.divider()
    st.subheader("Churn by Contract Type")

    contract_churn = (
        dashboard_df
        .groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Customer Churn Across Contract Types",
        labels={
            "Contract": "Contract Type",
            "Customers": "Number of Customers",
            "Churn": "Churn Status"
        }
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )

    # Calculate churn percentage for each contract type.
    contract_summary = (
        dashboard_df
        .groupby("Contract")
        .agg(
            Total_Customers=("Churn", "size"),
            Churned_Customers=(
                "Churn",
                lambda values: (values == "Yes").sum()
            )
        )
        .reset_index()
    )

    contract_summary["Churn Rate (%)"] = (
        contract_summary["Churned_Customers"]
        / contract_summary["Total_Customers"]
        * 100
    )

    st.subheader("Contract-wise Churn Summary")

    st.dataframe(
        contract_summary.sort_values(
            "Churn Rate (%)",
            ascending=False
        ).round(2),
        use_container_width=True,
        hide_index=True
    )


if dashboard_df is not None and "Churn" in dashboard_df.columns:

    st.divider()
    st.subheader("🔎 Business Insights")

    if total_customers > 0:

        st.markdown(
            f"""
            - The dataset contains **{total_customers:,} customers**.
            - **{churned_customers:,} customers** have churned.
            - The overall churn rate is **{churn_rate:.2f}%**.
            """
        )

    if (
        "Contract" in dashboard_df.columns
        and not contract_summary.empty
    ):

        highest_churn_contract = contract_summary.loc[
            contract_summary["Churn Rate (%)"].idxmax()
        ]

        st.markdown(
            f"""
            - **{highest_churn_contract['Contract']}** has the
              highest churn rate in this dataset:
              **{highest_churn_contract['Churn Rate (%)']:.2f}%**.
            """
        )

    st.caption(
        "These insights describe the available dataset. "
        "They do not establish the cause of customer churn."
    )


st.divider()



st.header("👩‍💼 Predict Churn for an Individual Customer")

st.write(
    "Enter the customer's details and click Predict Churn "
    "to estimate the likelihood of leaving the service."
)

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No"
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

        st.subheader("Phone Services")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        st.subheader("Internet Services")

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

        st.subheader("Additional Services")

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

        st.subheader("Contract Information")

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

        st.subheader("Billing Information")

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

        # Convert categorical values into model-compatible columns.
        customer_encoded = pd.get_dummies(
            customer_data,
            drop_first=True,
            dtype=int
        )

        # Keep the same feature columns and order as model training.
        if not hasattr(model, "feature_names_in_"):

            st.error(
                "The model does not contain feature names. "
                "Please check the model training process."
            )

            st.stop()

        customer_encoded = customer_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Get the probability for each model class.
        probabilities = model.predict_proba(customer_encoded)[0]
        model_classes = list(model.classes_)

        if "Yes" not in model_classes:

            st.error(
                "The trained model does not contain the "
                "'Yes' churn class."
            )

            st.stop()

        churn_probability = float(
            probabilities[model_classes.index("Yes")]
        )

        # Apply the same threshold used in the dashboard.
        if churn_probability >= CHURN_THRESHOLD:
            prediction = "Yes"
        else:
            prediction = "No"

        st.divider()
        st.header("📋 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == "Yes":

                st.error(
                    "⚠️ Customer Churn Predicted"
                )

                st.write(
                    "The model identifies this customer "
                    "as potentially likely to leave the service."
                )

            else:

                st.success(
                    "✅ Customer Retention Predicted"
                )

                st.write(
                    "The model predicts that this customer "
                    "is less likely to leave the service."
                )

        with result_col2:

            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

            st.progress(
                min(max(churn_probability, 0.0), 1.0)
            )

            st.caption(
                "Prediction threshold: 30%"
            )

        with st.expander("View Customer Details"):

            st.dataframe(
                customer_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This prediction is an estimate from a Machine Learning "
            "model and does not guarantee future customer behaviour."
        )

    except Exception as error:

        st.error(
            f"Unable to generate the prediction: {error}"
        )


st.divider()

st.caption(
    "Customer Churn Prediction | Logistic Regression | "
    "Streamlit | Machine Learning Project"
)
