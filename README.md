# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave the service. The project uses Logistic Regression and includes a Streamlit web app with customer predictions and an interactive dashboard.

## Live Demo

Try the application here:

**https://customer-churn-prediction-j7azbqqbxdtjkhy7qcnyo7.streamlit.app/**

## Project Overview

Customer churn happens when a customer stops using a company's service. Identifying customers who may leave can help a business understand churn patterns and plan ways to improve customer retention.

In this project, I used the IBM Telco Customer Churn dataset, trained a Logistic Regression model, evaluated its performance, and built a Streamlit app to explore churn data and make predictions.

## Features

- Predict whether a customer is likely to churn.
- Compare Logistic Regression with a Dummy Classifier baseline.
- View the confusion matrix.
- Explore the overall customer churn distribution.
- Compare churn across contract types.
- Use an interactive web application to try customer predictions.

## Machine Learning Workflow

1. Load the Telco Customer Churn dataset.
2. Prepare the data for machine learning.
3. Encode categorical features.
4. Split the data into training and testing sets.
5. Train a Logistic Regression model.
6. Evaluate the model using classification metrics.
7. Save the trained model.
8. Use the saved model in the Streamlit app.

## Dataset

This project uses the **IBM Telco Customer Churn** dataset.

- **Target column:** `Churn`
- **Purpose:** Predict whether a customer leaves the telecom service.
- **Data includes:** Customer details, subscribed services, contract information, payment method, tenure, and billing amounts.

The dataset file used in this repository is `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Model Performance

The model was evaluated on the test dataset.

| Metric | Logistic Regression |
|---|---:|
| Test Accuracy | 80.48% |
| Churn Precision | 65.62% |
| Churn Recall | 55.61% |
| Churn F1-Score | 60.20% |

A Dummy Classifier was used as a baseline. It achieved 73.46% accuracy but did not identify churn customers. Logistic Regression performed better overall, although its recall shows that there is still room to improve the detection of customers who may churn.

## Dashboard

The Streamlit dashboard includes these visualizations:

- **Baseline vs Logistic Regression:** compares the baseline model with Logistic Regression.
- **Confusion Matrix:** shows correct and incorrect predictions.
- **Customer Churn Distribution:** displays the number of customers who stayed and left.
- **Churn by Contract Type:** helps compare churn patterns across contract categories.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- GitHub
- Streamlit Community Cloud

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── churn_prediction.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/ranjanii925/customer-churn-prediction.git
   ```

2. Open the project folder:

   ```bash
   cd customer-churn-prediction
   ```

3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the Streamlit app:

   ```bash
   python -m streamlit run app.py
   ```

## Future Improvements

- Improve churn recall and compare additional classification models.
- Explore different prediction thresholds.
- Add more analysis of customer services, tenure, and billing.
- Provide further insights that could help businesses plan customer-retention activities.

## Author

**Ranjani**

GitHub: https://github.com/ranjanii925
