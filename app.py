import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "churn_prediction.pkl"
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Simple dashboard theme
NAVY = "#172B4D"
TEAL = "#159D91"
TEXT = "#F7F9FC"
MUTED = "#AAB7C8"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #F3F6FA; }}
      [data-testid="stHeader"] {{ background-color: #F3F6FA; }}
      [data-testid="stSidebar"] {{ background-color: {NAVY}; }}
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] .stCaption {{ color: {TEXT} !important; }}
      /* Keep form fields light with dark text so entered values stay readable. */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea {{
        background-color: #FFFFFF !important;
        color: #172B4D !important;
        -webkit-text-fill-color: #172B4D !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        color: #172B4D !important;
      }}
      [data-testid="stSidebar"] [data-baseweb="select"] div,
      [data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: #172B4D !important;
        -webkit-text-fill-color: #172B4D !important;
      }}
      [data-baseweb="popover"] {{ color: #172B4D !important; }}
      [data-baseweb="menu"] {{ background-color: #FFFFFF !important; }}
      [data-baseweb="menu"] * {{ color: #172B4D !important; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px; }}
      h1, h2, h3 {{ color: {NAVY}; }}
      [data-testid="stMetric"] {{ background: white; border: 1px solid #E0E6EF; border-radius: 8px; padding: 12px; }}
      [data-testid="stMetricLabel"] {{ color: #5B6575 !important; }}
      [data-testid="stMetricValue"] {{ color: {NAVY} !important; }}
      [data-testid="stPlotlyChart"] {{ background: white; border: 1px solid #E0E6EF; border-radius: 8px; padding: 4px; }}
      div[data-testid="stForm"] {{ border: 1px solid #344663; border-radius: 8px; padding: 10px; }}
      .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {{ background-color: {TEAL}; color: white; border: none; border-radius: 6px; }}
      hr {{ border-color: #DDE4ED; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(fig, height=290):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#26364A", family="Arial", size=11),
        title=dict(font=dict(size=14), x=0.02),
        margin=dict(l=20, r=15, t=42, b=25),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E9EDF2", zeroline=False)
    return fig


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# Load the dataset if it is available.
dashboard_df = None
if DATA_PATH.exists():
    try:
        dashboard_df = pd.read_csv(DATA_PATH)
    except Exception as error:
        st.warning(f"Dataset could not be loaded: {error}")

# Customer details form
st.sidebar.markdown("## Customer prediction")
st.sidebar.caption("Enter the customer details below to predict churn.")
with st.sidebar.form("customer_form", border=True):
    st.markdown("**Customer profile**")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

    st.markdown("**Services**")
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    st.markdown("**Contract and billing**")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total charges ($)", min_value=0.0, value=840.0, step=10.0)
    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

# Dashboard heading
st.title("Customer Churn Dashboard")
st.caption("View model results and explore customer churn patterns.")


# Show the prediction result
if submitted:
    customer_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])
    try:
        customer_encoded = pd.get_dummies(customer_data, drop_first=True, dtype=int)
        if not hasattr(model, "feature_names_in_"):
            st.error("Model feature names are unavailable. Please check the model training code.")
            st.stop()
        customer_encoded = customer_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(customer_encoded)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer_encoded)[0]
            classes = list(model.classes_)
            churn_probability = float(probabilities[classes.index("Yes")] * 100) if "Yes" in classes else None
        else:
            churn_probability = None

        if prediction == "Yes":
            st.error("⚠️ **High attention:** the model predicts this customer may churn.")
        else:
            st.success("✅ **Retention outlook:** the model predicts this customer may stay.")
        if churn_probability is not None:
            result_cols = st.columns([1, 2, 2])
            result_cols[0].metric("Estimated churn risk", f"{churn_probability:.1f}%")
            result_cols[1].progress(min(max(churn_probability / 100, 0.0), 1.0), text="Estimated churn probability")
            result_cols[2].metric("Customer tenure", f"{tenure} months")
        with st.expander("View submitted customer details"):
            st.dataframe(customer_data, use_container_width=True, hide_index=True)
        st.caption("This result is an estimate from the model, not a guarantee of customer behaviour.")
    except Exception as error:
        st.error(f"Prediction error: {error}")

# Model performance summary
st.subheader("Model performance")
kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="small")
kpi1.metric("Model accuracy", "80.48%", help="Overall correct predictions on the held-out test set.")
kpi2.metric("Churn precision", "65.62%", help="Of customers predicted to churn, this share actually churned.")
kpi3.metric("Churn recall", "55.61%", help="Of customers who churned, this share was detected by the model.")
kpi4.metric("Churn F1-score", "60.20%", help="Balance between churn precision and recall.")

# Tabs keep related charts together and reduce page scrolling.
tab_overview, tab_model, tab_segments = st.tabs(["Overview", "Model performance", "Customer segments"])

with tab_overview:
    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        st.markdown("#### Model accuracy")
        performance_df = pd.DataFrame({"Model": ["Dummy baseline", "Logistic regression"], "Accuracy (%)": [73.46, 80.48]})
        fig = px.bar(performance_df, x="Model", y="Accuracy (%)", text="Accuracy (%)", range_y=[0, 100], color="Model", color_discrete_sequence=["#AAB7C8", TEAL])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, yaxis_title="Accuracy (%)", xaxis_title="")
        st.plotly_chart(style_chart(fig, 285), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### Confusion matrix")
        cm = go.Figure(data=go.Heatmap(z=[[926, 109], [166, 208]], x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"], text=[[926, 109], [166, 208]], texttemplate="%{text}", colorscale=[[0, "#E8F7F4"], [0.5, "#73D8C7"], [1, "#167D76"]], showscale=False, hovertemplate="%{y}<br>%{x}<br>Customers: %{z}<extra></extra>"))
        cm.update_layout(xaxis_title="", yaxis_title="", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_chart(cm, 285), use_container_width=True, config={"displayModeBar": False})
    st.caption("The benchmark and confusion matrix are fixed evaluation results from this project's held-out test set.")

with tab_model:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Churn metrics")
        quality_df = pd.DataFrame({"Metric": ["Precision", "Recall", "F1-score"], "Score (%)": [65.62, 55.61, 60.20]})
        fig = px.bar(quality_df, x="Metric", y="Score (%)", text="Score (%)", range_y=[0, 100], color="Metric", color_discrete_sequence=["#6C8CFF", TEAL, "#F6B85F"])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Score (%)")
        st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### Prediction results")
        pred_df = pd.DataFrame({"Prediction": ["Correct stay predictions", "False churn alerts", "Missed churn", "Correct churn predictions"], "Customers": [926, 109, 166, 208]})
        fig = px.bar(pred_df, x="Customers", y="Prediction", orientation="h", text="Customers", color="Prediction", color_discrete_sequence=["#21B99F", "#6C8CFF", "#F6B85F", "#167D76"])
        fig.update_layout(showlegend=False, xaxis_title="Customers", yaxis_title="")
        st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
    st.info("Recall is 55.61%, so the model misses some customers who actually churn. This is an important area to improve before using the model for real retention decisions.")

with tab_segments:
    if dashboard_df is not None and "Churn" in dashboard_df.columns:
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("#### Customer churn")
            churn_counts = dashboard_df["Churn"].value_counts().rename_axis("Churn status").reset_index(name="Customers")
            fig = px.pie(churn_counts, names="Churn status", values="Customers", hole=0.62, color="Churn status", color_discrete_map={"No": TEAL, "Yes": "#F06A79"})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
        with right:
            if "Contract" in dashboard_df.columns:
                st.markdown("#### Churn by contract type")
                contract_df = dashboard_df.groupby(["Contract", "Churn"]).size().reset_index(name="Customers")
                fig = px.bar(contract_df, x="Contract", y="Customers", color="Churn", barmode="group", color_discrete_map={"No": TEAL, "Yes": "#F06A79"})
                fig.update_layout(xaxis_title="Contract type", yaxis_title="Customers", legend_title="Churn")
                st.plotly_chart(style_chart(fig, 300), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Customer segment charts are calculated from the dataset in the repository ({len(dashboard_df):,} rows).")
    else:
        st.info("To enable customer segment charts, add `WA_Fn-UseC_-Telco-Customer-Churn.csv` beside `app.py` in the GitHub repository.")

st.markdown("---")
st.caption("Customer Churn Prediction | Logistic Regression")
