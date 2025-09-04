
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# Load and prepare dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Bengaluru Ola.csv")
    
    # Create Pickup Time
    df["Pickup Time"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df["hour"] = df["Pickup Time"].dt.hour
    df["day_of_week"] = df["Pickup Time"].dt.dayofweek
    
    # Encode pickup location
    le = LabelEncoder()
    df["Pickup_enc"] = le.fit_transform(df["Pickup Location"].astype(str))
    
    # Cluster areas
    X_cluster = df[["Pickup_enc", "hour", "day_of_week"]]
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_cluster)
    
    return df, le, kmeans

df, le, kmeans = load_data()

# Aggregate demand
agg_df = df.groupby(["cluster", "day_of_week", "hour"]).size().reset_index(name="num_requests")
agg_df = agg_df.sort_values(by=["cluster", "day_of_week", "hour"])
agg_df["prev_hour_requests"] = agg_df.groupby("cluster")["num_requests"].shift(1).fillna(0)

# Features and target
X = agg_df[["cluster", "day_of_week", "hour", "prev_hour_requests"]]
y = agg_df["num_requests"]

# ---------------------------
# Train Decision Tree Model
# ---------------------------
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🚖 Ola Ride Demand Predictor (Decision Tree)")

# User inputs
pickup_input = st.selectbox("Select Pickup Location", df["Pickup Location"].unique())
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
hour = st.slider("Hour of Day (0-23)", 0, 23, 9)

# Encode inputs
pickup_enc = le.transform([pickup_input])[0]
day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week)

# Find cluster for this input
cluster = kmeans.predict([[pickup_enc, hour, day_num]])[0]

# Estimate previous hour demand (fallback=0 if no record)
row = agg_df[(agg_df["cluster"]==cluster) & (agg_df["day_of_week"]==day_num) & (agg_df["hour"]==hour-1)]
prev_hour_requests = row["num_requests"].values[0] if not row.empty else 0

# Predict demand
features = np.array([[cluster, day_num, hour, prev_hour_requests]])
pred = model.predict(features)[0]

# Display result
st.subheader("Predicted Ride Requests")
st.metric(label="Expected demand", value=int(pred))

# Optional: show feature importance
if st.checkbox("Show Feature Importance"):
    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    st.bar_chart(importance.set_index("feature"))
