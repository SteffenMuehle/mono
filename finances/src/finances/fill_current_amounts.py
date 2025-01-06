import toml
from pathlib import Path

def fill_current_amounts(file_path):
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
    set_aside_savings(
        source_dict=data,
        source_key="root.Steffen.savings.Comdirect",
        target_file_path=Path(__file__).parent.parent.parent / "data" / "input" / "graph" / "set_aside.toml",
    )
    
    with open(file_path, 'w') as f:
        toml.dump(data, f)

def set_aside_savings(source_dict, source_key, target_file_path):
    """
    set aside savings from source_dict to target_file_path
    """
    print("Setting aside savings...")

    dict_to_be_traversed = source_dict
    for key in source_key.split("."):
        dict_to_be_traversed = dict_to_be_traversed[key]
    savings_source = dict_to_be_traversed
    print(f"The savings source ({source_key}) has in it: {savings_source["current_amount"]}")

    set_aside_dict = toml.load(target_file_path)["entries"]
    print(f"The target location for set aside savings is {target_file_path}.")
    print(set_aside_dict)
    for key,val in set_aside_dict.items():
        if not isinstance(val, dict):
            continue
        amount_to_be_set_aside = val["target_amount"]
        print(f"Setting aside {amount_to_be_set_aside} from savings source to {key}")
        val["current_amount"] = amount_to_be_set_aside
        savings_source["current_amount"] -= amount_to_be_set_aside
    # data["current_amount"] = source_dict[source_key]["current_amount"]
    # with open(target_file_path, 'w') as f:
    #     toml.dump(data, f)

if __name__ == "__main__":
    for file_name in [
        "main.toml",
    ]:
        file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / file_name
        fill_current_amounts(file_path)