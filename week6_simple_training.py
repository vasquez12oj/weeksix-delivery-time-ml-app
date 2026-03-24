
import math
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Load the three Week 4 files
# -----------------------------
orders = pd.read_csv("simple_week4_orders.csv")
customers = pd.read_csv("simple_week4_customers.csv")
geo = pd.read_csv("simple_week4_geo.csv")

# -----------------------------
# 2. Merge and clean
# -----------------------------
orders = orders.merge(
    customers[["customer_id", "customer_zip_code_prefix"]],
    on="customer_id",
    how="left"
)

geo = geo[~geo["geolocation_zip_code_prefix"].duplicated()].copy()

orders = orders.merge(
    geo,
    left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left"
).rename(columns={
    "geolocation_lat": "seller_lat",
    "geolocation_lng": "seller_lng",
    "geolocation_city": "seller_city",
    "geolocation_state": "seller_state"
}).drop(columns=["geolocation_zip_code_prefix"])

orders = orders.merge(
    geo,
    left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix",
    how="left"
).rename(columns={
    "geolocation_lat": "customer_lat",
    "geolocation_lng": "customer_lng",
    "geolocation_city": "customer_city",
    "geolocation_state": "customer_state"
}).drop(columns=["geolocation_zip_code_prefix"])

# -----------------------------
# 3. Feature engineering
# -----------------------------
for col in [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_customer_date"
]:
    orders[col] = pd.to_datetime(orders[col])

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))

orders["distance_km"] = orders.apply(
    lambda row: haversine_km(
        row["seller_lng"], row["seller_lat"],
        row["customer_lng"], row["customer_lat"]
    ) if pd.notnull(row["customer_lat"]) else None,
    axis=1
)

orders["product_size_cm3"] = (
    orders["product_length_cm"] *
    orders["product_height_cm"] *
    orders["product_width_cm"]
)

orders["wait_time"] = (
    orders["order_delivered_customer_date"] -
    orders["order_purchase_timestamp"]
).dt.days

orders["est_wait_time"] = (
    orders["order_estimated_delivery_date"] -
    orders["order_purchase_timestamp"]
).dt.days

orders["purchase_dow"] = orders["order_purchase_timestamp"].dt.dayofweek
orders["purchase_month"] = orders["order_purchase_timestamp"].dt.month
orders["year"] = orders["order_purchase_timestamp"].dt.year
orders["delay"] = (orders["wait_time"] > orders["est_wait_time"]).astype(int)

# -----------------------------
# 4. Final Week 6-style dataset
# -----------------------------
final_df = orders[[
    "order_id",
    "seller_zip_code_prefix",
    "customer_zip_code_prefix",
    "purchase_dow",
    "purchase_month",
    "year",
    "product_size_cm3",
    "product_weight_g",
    "distance_km",
    "wait_time",
    "est_wait_time",
    "delay"
]].dropna().copy()

final_df.to_csv("simple_week6_final_dataset.csv", index=False)
print("Saved simple_week6_final_dataset.csv")

# -----------------------------
# 5. Train/test split
# -----------------------------
feature_cols = [
    "purchase_dow",
    "purchase_month",
    "year",
    "product_size_cm3",
    "product_weight_g",
    "distance_km"
]
target_col = "wait_time"

X = final_df[feature_cols]
y = final_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# -----------------------------
# 6. Train simple Week 6 models
# -----------------------------
lin_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
svr_model = SVR(C=10, epsilon=0.1, kernel="rbf")
voting_model = VotingRegressor([
    ("lin", lin_model),
    ("rf", rf_model),
    ("svr", svr_model)
])

models = {
    "LinearRegression": lin_model,
    "RandomForest": rf_model,
    "SVR": svr_model,
    "VotingRegressor": voting_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"{name} MSE: {mse:.4f}")

# -----------------------------
# 7. Save the voting model
# -----------------------------
with open("simple_week6_voting_model.pkl", "wb") as f:
    pickle.dump({"model": voting_model, "features": feature_cols}, f)

print("Saved simple_week6_voting_model.pkl")
