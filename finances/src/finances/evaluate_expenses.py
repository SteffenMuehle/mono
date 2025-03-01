import toml
from pathlib import Path
from datetime import datetime

TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO = 4

relative_path = Path(__file__).parent.parent.parent / "data" / "input"
io_toml_file_path = relative_path / "io.toml"
io = toml.load(io_toml_file_path)

monthly_out_dict = io["monthly"]["out"]
print("# Monthly Expenses")
print("\n| Expense Name | Amount |")
print("|--------------|--------|")
for key, amount in monthly_out_dict.items():
    print(f"| {key} | {amount} |")

monthly_in = round(sum(io["monthly"]["in"].values()), 2)
monthly_out = round(sum(monthly_out_dict.values()), 2)
monthly_surplus = round(monthly_in - monthly_out, 2)

print("\n**Summary***")
print(f"- **Monthly In**:  {monthly_in}")
print(f"- **Monthly Out**: {monthly_out}")
print(f"- **Surplus**:     {monthly_surplus}")

# set target in giro
main_toml_file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / "main.toml"
leaves = toml.load(main_toml_file_path)
target_in_giro = monthly_out * TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO
print("- Setting target in giro to:", target_in_giro)
leaves["root"]["Steffen"]["float"]["giro"]["target_amount"] = target_in_giro
with open(main_toml_file_path, 'w') as f:
    toml.dump(leaves, f)

print("\n# Major expenses of the last 2 years:")
two_years_ago = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")
rolling_year_expenses = sorted([
    (key, expense["amount"])
    for key, expense in io["expense"].items()
    if expense["date"] >= two_years_ago
], key=lambda x: x[1], reverse=True)

print("\n| Expense Name | Amount |")
print("|--------------|--------|")
total = 0
for key, amount in rolling_year_expenses:
    print(f"| {key} | {amount} |")
    total += amount

percentage_used_by_major_expenses = total / (2*12 * monthly_surplus)
print("\n**Summary***")
print(f"- Total major expenses of the last 2 years: {total}")
print(f"- Per month, that amounts to {round(100 * percentage_used_by_major_expenses,2)}% of the current surplus.")
print(f"- Current monthly surplus unused by major expenses: {round(monthly_surplus * (1 - percentage_used_by_major_expenses),2)}")

elizabeth_contributions = [
    val["amount"]
    for key, val in io['transaction'].items()
    if key.startswith("Elizabeth")
]
print("\n# Elizabeth's Contributions")
print("\n| Transaction | Amount |")
print("|-------------|--------|")
for key, val in io['transaction'].items():
    if key.startswith("Elizabeth"):
        print(f"| {val['date']} | {val['amount']} |")

total_elizabeth_contributions = round(sum(elizabeth_contributions), 2)
print("\n**Summary***")
print(f"- **Total Contributions**: {total_elizabeth_contributions}\n")


# write monthly total out into io.toml:
with open(io_toml_file_path, 'w') as f:
    toml.dump(io, f)
