import toml
from pathlib import Path

def prompt_values(file_path):
    """
    prompt current_ammount values in leaf nodes of a toml file
    """
    def traverse_and_prompt(data, path=""):
        for key, val in data.items():
            if isinstance(val, dict):
                traverse_and_prompt(val, path + key + ".")
            else:
                if key == "current_amount":
                    value = input(
                        f"\n\nEnter 'current_amount' for '{path[:-1]}':\n"
                        f"[old value: {val}]\n"
                        f"[leave empty to keep old value]\n"
                    )
                    if value:
                        data[key] = float(value)
                    else:
                        print(f"Keeping old value ({val}) for key {path[:-1]}")

    data = toml.load(file_path)
    traverse_and_prompt(data)
    
    with open(file_path, 'w') as f:
        toml.dump(data, f)

if __name__ == "__main__":
    for file_name in [
        "base.toml",
    ]:
        file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / file_name
        prompt_values(file_path)