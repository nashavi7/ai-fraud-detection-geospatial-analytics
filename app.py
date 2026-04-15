import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="Fraud Detection + Geo Analysis", page_icon="🛡️", layout="wide")

# ==========================================
# 1. SYNTHETIC DATASET
# ==========================================
@st.cache_data
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)

    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'London', 'Paris', 'Tokyo']

    data = {
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_samples)],
        'transaction_amount': np.round(np.random.exponential(scale=50, size=n_samples) + 5, 2),
        'hour': np.random.randint(0, 24, n_samples),
        'transaction_velocity': np.random.poisson(lam=2, size=n_samples),
        'amount_deviation': np.round(np.random.normal(loc=0.5, scale=0.5, size=n_samples), 2),
        'location': np.random.choice(locations, n_samples),
        'user_home': np.random.choice(locations, n_samples),
    }

    df = pd.DataFrame(data)

    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)
    df['location_mismatch'] = (df['location'] != df['user_home']).astype(int)

    prob_fraud = np.zeros(n_samples)
    prob_fraud += df['transaction_velocity'] * 0.05
    prob_fraud += np.maximum(0, df['amount_deviation']) * 0.15
    prob_fraud += df['location_mismatch'] * 0.2
    prob_fraud += df['is_night'] * 0.15
    prob_fraud += np.where(df['transaction_amount'] > 500, 0.3, 0)

    prob_fraud += np.random.normal(0, 0.1, n_samples)
    prob_fraud = np.clip(prob_fraud, 0, 1)

    threshold = np.percentile(prob_fraud, 95)
    df['fraud'] = (prob_fraud > threshold).astype(int)

    df['amount_deviation'] = np.abs(df['amount_deviation'])

    return df


# ==========================================
# 2. MODEL
# ==========================================
@st.cache_resource
def train_model(df):
    model_path = "model.pkl"

    features = ['transaction_amount', 'hour', 'is_night',
                'transaction_velocity', 'amount_deviation',
                'location_mismatch']

    X = df[features]
    y = df['fraud']

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return model, features, feat_imp, X, y


# ==========================================
# 3. EXPLAINABILITY
# ==========================================
def get_explanation(row, is_fraud, probability, feat_imp=None):
    reasons = []

    if is_fraud:
        reasons.append(f"🔴 **High Fraud Probability**: {probability * 100:.1f}%")

        if row['location_mismatch'] == 1:
            reasons.append(f"- 🌍 Location mismatch ({row['location']} vs {row['user_home']})")

        if row['transaction_amount'] > 200:
            reasons.append(f"- 💰 High amount (${row['transaction_amount']:.2f})")

        if row['transaction_velocity'] >= 5:
            reasons.append(f"- ⚡ High frequency ({row['transaction_velocity']})")

        if row['is_night'] == 1:
            reasons.append(f"- 🌙 Night transaction ({row['hour']}:00)")

        if feat_imp is not None:
            top = feat_imp.head(2)['Feature'].values
            reasons.append(f"- 📊 Model drivers: {', '.join(top)}")

    else:
        reasons.append(f"🟢 **Normal Transaction**: {probability * 100:.1f}% risk")
        reasons.append("- ✅ Behavior looks normal")

    return reasons


# ==========================================
# 4. LOCATION COORDINATES
# ==========================================
location_coords = {
    "New York": [40.7128, -74.0060],
    "Los Angeles": [34.0522, -118.2437],
    "Chicago": [41.8781, -87.6298],
    "Houston": [29.7604, -95.3698],
    "Miami": [25.7617, -80.1918],
    "London": [51.5074, -0.1278],
    "Paris": [48.8566, 2.3522],
    "Tokyo": [35.6762, 139.6503]
}


# ==========================================
# 5. MAIN APP
# ==========================================
def main():
    st.title("🛡️ Fraud Detection + Geospatial Analysis")
    st.markdown("AI-powered fraud detection with interactive map visualization")

    df = generate_synthetic_data()

    # Add geo features
    df['lat'] = df['location'].apply(lambda x: location_coords[x][0])
    df['lon'] = df['location'].apply(lambda x: location_coords[x][1])
    df['color'] = df['fraud'].apply(lambda x: [255, 0, 0] if x == 1 else [0, 200, 0])

    model, feature_cols, feat_imp, X, y = train_model(df)

    # ==========================================
    # 🔄 RANDOM TRANSACTION BUTTON
    # ==========================================
    if st.sidebar.button("🔄 Generate New Transaction"):
        st.session_state['txn_idx'] = np.random.randint(0, len(df))

    if 'txn_idx' not in st.session_state:
        st.session_state['txn_idx'] = np.random.randint(0, len(df))

    idx = st.session_state['txn_idx']
    txn = df.iloc[idx]

    # Prediction with spinner
    with st.spinner("🔍 Analyzing transaction..."):
        pred_prob = model.predict_proba(txn[feature_cols].values.reshape(1, -1))[0][1]
        pred_class = int(pred_prob > 0.4)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Transaction")
        st.write(f"Transaction ID: {txn['transaction_id']}")
        st.write(txn[['transaction_amount', 'location', 'user_home']])

    with col2:
        st.subheader("Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Explanation")
        exp = get_explanation(txn, pred_class, pred_prob, feat_imp)
        st.markdown("<br>".join(exp), unsafe_allow_html=True)

    # Feature importance
    st.subheader("Feature Importance")
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig_imp, use_container_width=True)

    # ==========================================
    # 🌍 MAP SECTION
    # ==========================================
    st.subheader("🌍 Geospatial Fraud Analysis")

    sample_df = df.sample(1000)

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=sample_df,
                get_position='[lon, lat]',
                get_color='color',
                get_radius=50000,
                pickable=True
            )
        ],
        tooltip={
            "html": "<b>Location:</b> {location} <br/> <b>Fraud:</b> {fraud}",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    )

    st.pydeck_chart(deck)

    # Fraud hotspots
    st.subheader("🚨 Fraud Hotspots")

    fraud_df = df[df['fraud'] == 1]

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=fraud_df,
                get_position='[lon, lat]',
                get_color='[255,0,0]',
                get_radius=70000
            )
        ]
    ))


if __name__ == "__main__":
    main()