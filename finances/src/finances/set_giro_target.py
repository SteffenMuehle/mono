import toml
from pathlib import Path

TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO = 4

file_path_io = Path(__file__).parent.parent.parent / "input" / "io.toml"
file_path_graph_leaves = Path(__file__).parent.parent.parent / "input" / "graph" / "base.toml"

# get target
io = toml.load(file_path_io)
monthly_cost = io["monthly"]["out_total"]
target_in_giro = monthly_cost * TARGET_NUMBER_OF_MONTHLY_COSTS_IN_GIRO

# set target
leaves = toml.load(file_path_graph_leaves)
leaves["root"]["Steffen"]["float"]["giro"]["target_amount"] = target_in_giro

print("Setting target in giro to:", target_in_giro)
# save file
with open(file_path_graph_leaves, 'w') as f:
    toml.dump(leaves, f)