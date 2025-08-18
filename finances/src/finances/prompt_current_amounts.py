from pathlib import Path

import toml


def fill_current_amounts(file_path):
    """
    prompt current_ammount values in leaf nodes of a toml file
    """
    set_aside_file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / "set_aside.md"
    set_aside_dict = toml.load(set_aside_file_path)["entries"]
    set_aside_total_amount = sum(
        [val["current_amount"] for key, val in set_aside_dict.items() if isinstance(val, dict)]
    )

    def traverse_and_prompt(data, path=""):
        for key, old_val in data.items():
            if isinstance(old_val, dict):
                traverse_and_prompt(old_val, path + key + ".")
            else:
                if key == "current_amount":
                    prompted_value = input(
                        f"\n\nEnter 'current_amount' for '{path[:-1]}':\n"
                        f"[old value: {old_val}]\n"
                        f"[leave empty to keep old value]\n"
                    )
                    if prompted_value:
                        data[key] = float(prompted_value)
                    else:
                        # special rule for ING Diba: adding set_aside savings
                        if str(path) == "root.Steffen.savings.ING.":
                            print(
                                f"trying to be smart: adding set aside toal amount ({set_aside_total_amount}) to old value ({old_val}) for key {path[:-1]}"
                            )
                            data[key] = old_val + set_aside_total_amount
                        else:
                            print(f"Keeping old value ({old_val}) for key {path[:-1]}")

    data = toml.load(file_path)
    traverse_and_prompt(data)
    set_aside_savings(
        source_dict=data,
        source_key="total.Steffen.savings",
        target_file_path=set_aside_file_path,
    )

    with open(file_path, "w") as f:
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
    print(f"The savings source ({source_key}) has in it: {savings_source['current_amount']}")

    set_aside_dict = toml.load(target_file_path)["entries"]
    print(f"The target location for set aside savings is {target_file_path}.")
    for key, val in set_aside_dict.items():
        if not isinstance(val, dict):
            continue
        amount_to_be_set_aside = val["current_amount"]
        print(f"Setting aside {amount_to_be_set_aside} from savings source to {key}")
        savings_source["current_amount"] -= amount_to_be_set_aside


if __name__ == "__main__":
    for file_name in [
        "main.md",
    ]:
        file_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph" / file_name
        fill_current_amounts(file_path)
