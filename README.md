# ai-fraud-detection-geospatial-analytics
AI-powered fraud detection system with explainable ML models and interactive geospatial analytics using Streamlit, Plotly, and PyDeck.
# AI Fraud Detection + Geospatial Analytics

An end-to-end **Machine Learning system** for detecting fraudulent transactions with **explainability and geospatial intelligence**.

Built using **Streamlit, Scikit-learn, Plotly, and PyDeck**, this project demonstrates how AI can be applied to real-world financial fraud detection scenarios.

---

## Live Demo

---

##Key Features

###Fraud Detection Model

* Random Forest-based classification model
* Handles imbalanced fraud datasets
* Real-time prediction on transactions

###Explainable AI (XAI)

* Human-readable explanations for each prediction
* Identifies key fraud signals:

  * High transaction amount
  * Location mismatch
  * Night-time activity
  * Transaction velocity

###Interactive Visualizations

* Risk meter using Plotly gauge charts
* Feature importance analysis
* Model-driven insights

###Geospatial Analytics

* Interactive global transaction map using PyDeck
* Fraud vs normal transaction visualization
* Fraud hotspot detection

---

## Tech Stack

* **Frontend/UI**: Streamlit
* **Machine Learning**: Scikit-learn (Random Forest)
* **Data Processing**: Pandas, NumPy
* **Visualization**: Plotly, PyDeck
* **Model Persistence**: Joblib

---

##Project Structure

```
├── app.py                # Main Streamlit application
├── model.pkl            # Saved ML model (auto-generated)
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

##  How to Run Locally

### 1. Clone the Repository

```
git clone https://github.com/your-username/ai-fraud-detection-geospatial-analytics.git
cd ai-fraud-detection-geospatial-analytics
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the App

```
streamlit run app.py
```

---

##Dataset

This project uses a **synthetic dataset** generated using probabilistic rules to simulate real-world fraud patterns:

* Transaction amount anomalies
* Behavioral deviations
* Location mismatches
* Time-based fraud patterns

---

##Model Details

* Algorithm: Random Forest Classifier
* Handles imbalance using `class_weight='balanced'`
* Feature Engineering includes:

  * Transaction velocity
  * Amount deviation
  * Night-time activity
  * Location mismatch

---

##Use Cases

* Banking & Financial Fraud Detection
* Payment Gateway Monitoring
* Risk Analysis Systems
* Real-time Transaction Screening

---

##Future Improvements

* Integrate real-world datasets (e.g., credit card fraud dataset)
* Add SHAP-based explainability
* Deploy using Docker + AWS/GCP
* Real-time streaming (Kafka integration)
* API-based inference system


