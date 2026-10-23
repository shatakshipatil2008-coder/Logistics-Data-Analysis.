"""
Yuva Intern - Week 1: Strategic Planning and Data Exploration in Logistics
Educational demonstration using illustrative/synthetic data only.

Install:
    pip install pandas scikit-learn

Run:
    python logistics_analysis.py
Optional:
    python logistics_analysis.py sample_logistics_data.csv
"""
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_FILE = "sample_logistics_data.csv"

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    df = pd.read_csv(file_path)

    required = {
        "distance_km", "traffic_level", "items", "warehouse_load_pct",
        "delivery_minutes", "on_time", "order_complete", "delivery_cost_inr"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    # 1. Basic data-quality checks
    print("\\n=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\\nMissing values per column:\\n", df.isna().sum())
    print("\\nDescriptive statistics:\\n", df.describe(numeric_only=True).round(2))

    # 2. KPI calculations
    on_time_rate = df["on_time"].mean() * 100
    order_fill_rate = df["order_complete"].mean() * 100
    avg_delivery_time = df["delivery_minutes"].mean()
    avg_cost = df["delivery_cost_inr"].mean()
    print("\\n=== KEY PERFORMANCE INDICATORS ===")
    print(f"On-time delivery rate: {on_time_rate:.1f}%")
    print(f"Order-completion rate: {order_fill_rate:.1f}%")
    print(f"Average delivery time: {avg_delivery_time:.1f} minutes")
    print(f"Average delivery cost: INR {avg_cost:.2f} per order")

    # 3. Illustrative regression model for delivery time.
    # For a real project, split data into train/test sets and validate performance.
    features = ["distance_km", "traffic_level", "items", "warehouse_load_pct"]
    X = df[features]
    y = df["delivery_minutes"]
    model = LinearRegression()
    model.fit(X, y)
    print("\\n=== EXPLORATORY REGRESSION ===")
    print("Model coefficients (association, not proof of causation):")
    for name, coefficient in zip(features, model.coef_):
        print(f"  {name}: {coefficient:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")
    example = pd.DataFrame([{
        "distance_km": 5.0, "traffic_level": 2, "items": 4,
        "warehouse_load_pct": 60
    }])
    print("Illustrative predicted delivery time for a 5 km, medium-traffic order:",
          f"{model.predict(example)[0]:.1f} minutes")

    # 4. Segment orders into operational profiles using K-means.
    cluster_features = ["distance_km", "traffic_level", "items",
                        "warehouse_load_pct", "delivery_minutes"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cluster_features])
    cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["operational_cluster"] = cluster_model.fit_predict(scaled)
    print("\\n=== ORDER SEGMENTS (cluster labels are arbitrary) ===")
    print(df.groupby("operational_cluster")[cluster_features].mean().round(2))

    # 5. Flag orders that need review.
    late_orders = df[df["on_time"] == 0]
    print("\\n=== ORDERS TO REVIEW ===")
    if late_orders.empty:
        print("No late orders in this sample.")
    else:
        print(late_orders[["order_id", "distance_km", "traffic_level",
                           "warehouse_load_pct", "delivery_minutes"]].to_string(index=False))

    # Export enriched output for inspection.
    output_file = "logistics_analysis_output.csv"
    df.to_csv(output_file, index=False)
    print(f"\\nSaved enriched data to {output_file}")
    print("Reminder: this dataset is synthetic and results are for demonstration only.")

if __name__ == "__main__":
    main()
