
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")

st.title("🚀 PortKart Logistics Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    st.markdown("## 📊 Key Insights")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clients", len(df))
    col2.metric("Avg Spend", int(df["monthly_spend_value"].mean()))
    col3.metric("Avg Cost", int(df["total_cost"].mean()))
    col4.metric("Conversion Rate", f"{df['will_choose_portkart'].mean()*100:.1f}%")

    st.markdown("## 📈 Business Overview")

    fig1 = px.histogram(df, x="industry", color="will_choose_portkart")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="distance_km", y="total_cost", color="will_choose_portkart")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## 🤖 Prediction Model")

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("will_choose_portkart", axis=1)
    y = df["will_choose_portkart"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("Precision:", precision_score(y_test, preds))
    st.write("Recall:", recall_score(y_test, preds))
    st.write("F1 Score:", f1_score(y_test, preds))

    st.markdown("## 🧩 Customer Segments")

    kmeans = KMeans(n_clusters=3)
    df["cluster"] = kmeans.fit_predict(X)

    fig3 = px.scatter(df, x="monthly_spend_value", y="total_cost", color="cluster")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("## 💰 Spending Prediction")

    reg = RandomForestRegressor()
    reg.fit(X_train, df.loc[X_train.index, "monthly_spend_value"])
    preds_reg = reg.predict(X_test)

    st.write("Sample Spend Predictions:", preds_reg[:10])

    st.markdown("## 🔮 New Customer Prediction")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, value=0)

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([input_data]))
        st.success(f"Prediction: {pred[0]}")
