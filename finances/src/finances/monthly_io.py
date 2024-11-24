import toml
from pathlib import Path

relative_path = Path(__file__).parent.parent.parent / "data" / "input"
toml_file_path = relative_path / "io.toml"
io = toml.load(toml_file_path)

monthly_out_dict = io["monthly"]["out"]
max_monthly_out_name_lengths = max([len(x) for x in list(monthly_out_dict.keys())])+1
print("\nMonthly costs:")
for key, amount in monthly_out_dict.items():
    print(f"{key.ljust(max_monthly_out_name_lengths)}: {amount}")

monthly_in = round(sum(io["monthly"]["in"].values()), 2)
monthly_out = round(sum(monthly_out_dict.values()), 2)
monthly_surplus = round(monthly_in - monthly_out, 2)

print("")
print(f"monthly in:  {monthly_in}")
print(f"monthly out: {monthly_out}")
print("--------------------")
print(f"surplus:     {monthly_surplus}")
print("")

today = "2023-09-29"
rolling_year_expenses = sorted([
    (key, expense["amount"])
    for key, expense in io["expense"].items()
    if expense["date"] > today
], key=lambda x: x[1], reverse=True)

# Calculate the length of the longest key
max_key_length = max(len(key) for key, _ in rolling_year_expenses)+1

print("Rolling year expenses above 200€:")
total = 0
for key, amount in rolling_year_expenses:
    print(f"{key.ljust(max_key_length)}: {amount}")
    total += amount
print("---------------")
print(f"total: {total}")

percentage_used_by_major_expenses = total / (12 * monthly_surplus)
print("")
print(f"That is {round(100 * percentage_used_by_major_expenses,2)}% of the rolling year surplus ({round(12 * monthly_surplus, 2)})")

elizabeth_contributions = [
    val["amount"]
    for key, val in io['transaction'].items()
    if key.startswith("Elizabeth")
]

print("")
print(f"Elizabeth total contribution: {round(sum(elizabeth_contributions),2)}")


# write monthly total out into io.toml:
print("\nUpdating io.toml:")
io["monthly"]["in_total"] = monthly_in
print(f"updated monthly.total_in: {monthly_in}")
io["monthly"]["out_total"] = monthly_out
print(f"updated monthly.total_out: {monthly_out}")
io["monthly"]["surplus"] = monthly_surplus
print(f"updated monthly.surplus: {monthly_surplus}")
io["monthly"]["surplus_unused_by_major_expenses"] = round(monthly_surplus * (1 - percentage_used_by_major_expenses),2)
print(f"updated monthly.surplus_unused_by_major_expenses: {round(monthly_surplus * (1 - percentage_used_by_major_expenses),2)}")
with open(toml_file_path, 'w') as f:
    toml.dump(io, f)