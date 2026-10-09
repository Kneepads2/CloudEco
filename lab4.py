from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AWS EC2 Price Predictor",
    layout="wide"
)

st.title("AWS EC2 Price Predictor")

filepath = "ec2dataset.csv"

try:
    original_data = pd.read_csv(filepath)
except FileNotFoundError:
    st.error(f"Could not find {filepath}. Make sure it is in the same folder as app.py.")
    st.stop()

data = original_data.copy()

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.replace("hourly", "", case=False, regex=False)
            .str.strip(),
            errors="coerce"
        )

required_columns = ["On Demand", "Instance Memory", "vCPUs"]
missing_columns = [column for column in required_columns if column not in data.columns]

if missing_columns:
    st.error(f"Required columns are missing from the dataset: {', '.join(missing_columns)}")
    st.stop()

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"]
    .astype(str)
    .str.replace("GiB", "", regex=False)
    .str.strip(),
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

def detect_outliers(df, column):
    values = df[column].dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

outliers_on_demand, lower_bound, upper_bound = detect_outliers(data, "On Demand")
data_without_outliers = data.drop(index=outliers_on_demand.index)
data_cleaned = data_without_outliers.dropna(subset=required_columns)

if len(data_cleaned) < 5:
    st.error("There are not enough valid rows to train and evaluate the model.")
    st.stop()

X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Instances", f"{len(original_data):,}")
col2.metric("Columns", f"{len(original_data.columns):,}")
col3.metric("Missing Values", f"{int(original_data.isna().sum().sum()):,}")
col4.metric("Rows Used by Model", f"{len(data_cleaned):,}")

st.subheader("Original Dataset")
st.dataframe(original_data, use_container_width=True, height=300)

with st.expander("Dataset Columns and Data Types"):
    dtype_df = pd.DataFrame({
        "Column": original_data.columns,
        "Data Type": original_data.dtypes.astype(str).values
    })
    st.write("Columns:", original_data.columns.tolist())
    st.dataframe(dtype_df, use_container_width=True)

st.divider()
st.header("Data Cleaning")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Outliers Detected", f"{len(outliers_on_demand):,}")
col2.metric("Outlier Lower Bound", f"${lower_bound:.4f}")
col3.metric("Outlier Upper Bound", f"${upper_bound:.4f}")
col4.metric("Rows Removed Overall", f"{len(original_data) - len(data_cleaned):,}")

st.subheader("On-Demand Cost Outliers")
if outliers_on_demand.empty:
    st.write("No On-Demand cost outliers were detected.")
else:
    st.dataframe(outliers_on_demand, use_container_width=True, height=220)

st.subheader("Cleaned Dataset")
st.dataframe(data_cleaned, use_container_width=True, height=300)

missing_df = pd.DataFrame({
    "Column": data_cleaned.columns,
    "Missing Values": data_cleaned.isna().sum().values
})
with st.expander("Missing Values After Cleaning"):
    st.dataframe(missing_df, use_container_width=True)



st.write(f"Training samples: {len(X_train):,} · Testing samples: {len(X_test):,}")

st.subheader("Actual vs Predicted On-Demand Costs")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.7)
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
ax.set_xlabel("Actual On-Demand Cost ($/hour)")
ax.set_ylabel("Predicted On-Demand Cost ($/hour)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()
st.header("EC2 Price Predictor")

possible_instance_columns = [
    "Instance Type",
    "Instance",
    "EC2 Instance",
    "Name",
    "Instance Name"
]
instance_column = next(
    (column for column in possible_instance_columns if column in data_cleaned.columns),
    None
)

# prediction_method = st.radio(
#     "Prediction method",
#     ["Select an EC2 instance", "Enter vCPUs and memory"],
#     horizontal=True
# )

# if prediction_method == "Select an EC2 instance":
#     if instance_column is None:
#         st.warning("No recognized EC2 instance-name column was found. Choose manual input instead.")
#     else:
#         instance_names = sorted(data_cleaned[instance_column].dropna().astype(str).unique().tolist())
#         if not instance_names:
#             st.warning("No EC2 instance names are available. Choose manual input instead.")
#         else:
#             selected_instance = st.selectbox("EC2 Instance", instance_names)
#             selected_rows = data_cleaned[
#                 data_cleaned[instance_column].astype(str) == selected_instance
#             ]
#             selected_row = selected_rows.iloc[0]
#             memory = float(selected_row["Instance Memory"])
#             vcpus = float(selected_row["vCPUs"])

#             col1, col2, col3 = st.columns(3)
#             col1.metric("Memory", f"{memory:g} GiB")
#             col2.metric("vCPUs", f"{vcpus:g}")
#             prediction = float(model.predict(pd.DataFrame({
#                 "Instance Memory": [memory],
#                 "vCPUs": [vcpus]
#             }))[0])
#             col3.metric("Predicted Hourly Cost", f"${prediction:.4f}")

#             st.metric("Estimated Monthly Cost (730 hours)", f"${prediction * 730:.2f}")
#             if pd.notna(selected_row["On Demand"]):
#                 st.write(f"Actual On-Demand price in dataset: **${selected_row['On Demand']:.4f}/hour**")
#                 st.write(f"Difference from dataset price: **${prediction - selected_row['On Demand']:.4f}/hour**")

# else:
col1, col2 = st.columns(2)
with col1:
        vcpus = st.number_input(
            "Number of vCPUs",
            min_value=1,
            max_value=256,
            value=2,
            step=1
        )
with col2:
        memory = st.number_input(
            "Memory (GiB)",
            min_value=0.5,
            max_value=2048.0,
            value=4.0,
            step=0.5
        )

prediction = float(model.predict(pd.DataFrame({
        "Instance Memory": [memory],
        "vCPUs": [vcpus]
    }))[0])

col1, col2 = st.columns(2)
col1.metric("Predicted On-Demand Cost", f"${prediction:.4f}/hour")
col2.metric("Estimated Monthly Cost (730 hours)", f"${prediction * 730:.2f}")

config = pd.DataFrame({
        "Specification": ["vCPUs", "Memory (GiB)", "Predicted Hourly Cost", "Estimated Monthly Cost"],
        "Value": [f"{vcpus}", f"{memory:g}", f"${prediction:.4f}", f"${prediction * 730:.2f}"]
    })
st.dataframe(config, use_container_width=True)

st.caption(
    "Price predictions are estimates based on the supplied dataset and may differ from current AWS pricing. "
    "Monthly estimates assume 730 hours of usage."
)

