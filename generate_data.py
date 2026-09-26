import numpy as np
import pandas as pd


np.random.seed(42)

n_transactions = 10000

amount = np.random.lognormal(
    mean=7,
    sigma=1,
    size=n_transactions
)

transaction_hour = np.random.randint(
    0,
    24,
    n_transactions
)

transaction_frequency = np.random.poisson(
    3,
    n_transactions
)

account_age_days = np.random.randint(
    30,
    2000,
    n_transactions
)

location_change = np.random.binomial(
    1,
    0.08,
    n_transactions
)

time_since_last_transaction = np.random.exponential(
    scale=180,
    size=n_transactions
)

device_change = np.random.binomial(
    1,
    0.05,
    n_transactions
)

fraud_probability = (
    0.02
    + 0.18 * (amount > 10000)
    + 0.20 * (location_change == 1)
    + 0.18 * (device_change == 1)
    + 0.15 * (time_since_last_transaction < 10)
    + 0.10 * (transaction_hour < 5)
    + 0.10 * (transaction_frequency > 8)
)

fraud_probability = np.clip(
    fraud_probability,
    0,
    0.95
)

is_fraud = (
    np.random.random(n_transactions)
    < fraud_probability
).astype(int)

transactions = pd.DataFrame({
    "transaction_id": [
        f"TXN{i:06d}"
        for i in range(1, n_transactions + 1)
    ],
    "amount": amount.round(2),
    "transaction_hour": transaction_hour,
    "transaction_frequency": transaction_frequency,
    "account_age_days": account_age_days,
    "location_change": location_change,
    "time_since_last_transaction": (
        time_since_last_transaction.round(2)
    ),
    "device_change": device_change,
    "is_fraud": is_fraud
})

transactions.to_csv(
    "Dataset/transactions.csv",
    index=False
)

print("Dataset generated successfully.")
print(f"Total transactions: {len(transactions)}")
print(
    f"Fraud transactions: "
    f"{transactions['is_fraud'].sum()}"
)
print(
    f"Fraud percentage: "
    f"{transactions['is_fraud'].mean() * 100:.2f}%"
)