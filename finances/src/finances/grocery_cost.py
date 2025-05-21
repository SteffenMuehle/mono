from pathlib import Path

import pandas as pd

num_months = 6

relative_path = Path(__file__).parent.parent.parent / "data" / "manual_logs" / "monthly_estimate_2024_april_to_october"
file_names = [
    ("edeka.csv", ";"),
    ("budni.csv", ";"),
    ("paypal.csv", ","),
]

grand_total = 0

for file_name, separator in file_names:
    df = pd.read_csv(relative_path / file_name, sep=separator)
    # parse from german to english, then convert to float:
    df["Umsatz in EUR"] = df["Umsatz in EUR"].str.replace(",", ".").astype(float)
    monthly_cost = round(df["Umsatz in EUR"].sum() / num_months, 2)
    print(f"{file_name}: {monthly_cost}")
    grand_total += monthly_cost

print(f"Grand total: {round(grand_total, 2)}")
