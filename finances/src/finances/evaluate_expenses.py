import toml
from pathlib import Path
from datetime import datetime

TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO = 4

relative_path = Path(__file__).parent.parent.parent / "data" / "input"
io_toml_file_path = relative_path / "io.toml"
io = toml.load(io_toml_file_path)


print("# History")


print("## Elizabeth's contributions")

elizabeth_contributions = [
    val["amount"]
    for key, val in io['transaction'].items()
    if key.startswith("Elizabeth")
]
print("\n| Transaction | Amount |")
print("|-------------|--------|")
for key, val in io['transaction'].items():
    if key.startswith("Elizabeth"):
        print(f"| {val['date']} | +{val['amount']} |")
elizabeth_contributions_total = round(sum(elizabeth_contributions), 2)
print(f"| total | +{elizabeth_contributions_total} |\n")


print("## Major expenses of the last 2 years:")
two_years_ago = (datetime.now().replace(year=datetime.now().year - 2)).strftime("%Y-%m-%d")
rolling_two_years_expenses = sorted([
    (expense["date"], key, expense["amount"])
    for key, expense in io["expense"].items()
    if expense["date"] >= two_years_ago
], key=lambda x: x[0], reverse=False)

print("\n| Date | Expense Name | Amount |")
print("|---|---|---|")
rolling_two_years_expenses_total = 0
for date, key, amount in rolling_two_years_expenses:
    print(f"| {date} | {key} | -{amount} |")
    rolling_two_years_expenses_total += amount
print(f"| | total | -{rolling_two_years_expenses_total} |\n")


print("# Monthly")

monthly_in_dict = io["monthly"]["in"]
monthly_out_dict = io["monthly"]["out"]
print("## Income")
print("\n| Source | Amount |")
print("|--------------|--------|")
monthly_income_total = 0
for key, amount in monthly_in_dict.items():
    print(f"| {key} | +{amount} |")
    monthly_income_total += amount
print(f"| total | +{monthly_income_total} |\n")

print("\n## Expenses")
print("\n| Expense | Amount |")
print("|---------|--------|")
monthly_expense_total = 0
for key, amount in monthly_out_dict.items():
    print(f"| {key} | -{amount} |")
    monthly_expense_total += amount
print(f"| total | -{monthly_expense_total:.2f} |\n")

monthly_out = round(sum(monthly_out_dict.values()), 2)
monthly_in = round(sum(monthly_in_dict.values()), 2)
monthly_surplus = round(monthly_in - monthly_out, 2)



print("## Summary")
percentage_used_by_major_expenses = rolling_two_years_expenses_total / (2*12 * monthly_surplus)
target_in_giro = monthly_out * TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO
print(f"- Monthly In:  {monthly_in}")
print(f"- Monthly Out: {monthly_out}   --> Setting target in giro to {TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO} monthly expenses, which is {target_in_giro}")
print(f"- Surplus:     {monthly_surplus}")
print(f"- Total major expenses of the last 2 years: {rolling_two_years_expenses_total}")
print(f"- Per month, that is {rolling_two_years_expenses_total/24:.2f}")
print(f"- That amounts to {round(100 * percentage_used_by_major_expenses,2)}% of the monthly surplus.")
print(f"- So, the effective monthly surplus is: {round(monthly_surplus * (1 - percentage_used_by_major_expenses),2)}")



# set target in giro
main_toml_file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / "main.toml"
main_dict = toml.load(main_toml_file_path)
main_dict["root"]["Steffen"]["float"]["giro"]["target_amount"] = target_in_giro

with open(main_toml_file_path, 'w') as f:
    toml.dump(main_dict, f)