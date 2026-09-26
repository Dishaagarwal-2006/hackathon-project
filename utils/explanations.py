def explain_transaction(transaction):
    reasons = []

    if transaction["amount"] > 10000:
        reasons.append(
            "Unusually high transaction amount"
        )

    if transaction["location_change"] == 1:
        reasons.append(
            "Unusual location change detected"
        )

    if transaction["device_change"] == 1:
        reasons.append(
            "New or changed device detected"
        )

    if transaction["time_since_last_transaction"] < 10:
        reasons.append(
            "Very short interval between transactions"
        )

    if transaction["transaction_hour"] < 5:
        reasons.append(
            "Transaction occurred during unusual hours"
        )

    if transaction["transaction_frequency"] > 8:
        reasons.append(
            "Unusually high transaction frequency"
        )

    if not reasons:
        reasons.append(
            "No major behavioral anomaly detected"
        )

    return reasons