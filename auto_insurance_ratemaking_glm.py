import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(42)
n_rows = 10000

# Simulate data

data = pd.DataFrame({
    "driver_age": np.random.randint(18, 75, n_rows),
    "vehicle_value": np.random.randint(5000, 50000, n_rows),
    "miles_driven": np.random.randint(2000, 20000, n_rows)
})

# Create age segments for pricing
data["age_segment"] = pd.cut(
    data["driver_age"],
    bins=[17, 25, 50, 75],
    labels=["Young", "Middle", "Senior"]
)

# Convert age segment into dummy variables
data = pd.get_dummies(data, columns=["age_segment"], dtype=int)
# Simulate claim frequency

lambda_freq = np.exp(
    -3.5
    + 0.5 * data["age_segment_Young"]
    + 0.00005 * data["miles_driven"]
)

data["claim_count"] = np.random.poisson(lambda_freq)

# Simulate claim severity
mu_sev = 500 + 0.05 * data["vehicle_value"]

shape_param = 2.0
scale_param = mu_sev / shape_param

data["total_claim_cost"] = data["claim_count"] * np.random.gamma(
    shape=shape_param,
    scale=scale_param
)

# Train / test split
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Frequency model (Poisson GLM)
freq_cols = ["age_segment_Young", "miles_driven"]

X_freq_train = sm.add_constant(train_data[freq_cols], has_constant="add")
X_freq_test = sm.add_constant(test_data[freq_cols], has_constant="add")

freq_model = sm.GLM(
    train_data["claim_count"],
    X_freq_train,
    family=sm.families.Poisson()
).fit()

# Severity model (Gamma GLM)
severity_train = train_data[train_data["claim_count"] > 0].copy()
severity_test = test_data[test_data["claim_count"] > 0].copy()

severity_train["avg_claim_cost"] = (
    severity_train["total_claim_cost"] / severity_train["claim_count"]
)
severity_test["avg_claim_cost"] = (
    severity_test["total_claim_cost"] / severity_test["claim_count"]
)

X_sev_train = sm.add_constant(severity_train[["vehicle_value"]], has_constant="add")
X_sev_test = sm.add_constant(severity_test[["vehicle_value"]], has_constant="add")

sev_model = sm.GLM(
    severity_train["avg_claim_cost"],
    X_sev_train,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()

# Predict on test data
test_data = test_data.copy()
test_data["expected_claim_count"] = freq_model.predict(X_freq_test)

X_sev_all_test = sm.add_constant(test_data[["vehicle_value"]], has_constant="add")
test_data["expected_claim_severity"] = sev_model.predict(X_sev_all_test)

test_data["pure_premium"] = (
    test_data["expected_claim_count"] * test_data["expected_claim_severity"]
)

print("--- FREQUENCY MODEL SUMMARY (Poisson) ---")
print(freq_model.summary())

print("\n--- SEVERITY MODEL SUMMARY (Gamma) ---")
print(sev_model.summary())

print("\n--- SAMPLE TEST-SET OUTPUT ---")
print(
    test_data[
        ["driver_age", "vehicle_value", "claim_count", "total_claim_cost", "pure_premium"]
    ].head(10)
)

# Visualization
plt.scatter(test_data["driver_age"], test_data["pure_premium"], alpha=0.5)
plt.title("Predicted Pure Premium by Driver Age")
plt.xlabel("Driver Age")
plt.ylabel("Pure Premium ($)")
plt.show()
