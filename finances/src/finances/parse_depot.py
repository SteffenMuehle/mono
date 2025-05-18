import csv
import toml
from pathlib import Path

def update_dict_from_csv(toml_dict, csv_file_path, bank):
    id_column = {
        "comdirect": "WKN",
        "ing": "ISIN",
    }[bank]

    value_column = {
        "comdirect": "Wert in EUR",
        "ing": "Kurswert",
    }[bank]

    # Read the CSV file and store the data in a dictionary
    csv_data = {}
    csv_ids = set()
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            wkn = row[id_column]
            value = float(row[value_column].replace('.', '').replace(',', '.'))
            csv_data[wkn] = value
            csv_ids.add(wkn)

    # Lists to keep track of altered and not found entries
    altered_entries = []
    not_found_entries = []

    # Function to recursively update the TOML data
    def update_toml(toml_data, parent_key=""):
        # is bottom layer? if no dicts below, then yes:
        more_dicts_below = any(isinstance(val, dict) for key, val in toml_data.items())
        if more_dicts_below:
            # If not bottom layer, then recurse
            for key, val in toml_data.items():
                if isinstance(val, dict):
                    new_parent_key = f"{parent_key}.{key}" if parent_key else key
                    update_toml(val, new_parent_key)
        else:
            # Infer the id from the parent_key
            inferred_id = parent_key.split('.')[-1]
            display_name = toml_data.get('print_name', inferred_id)
            display_entry = f"{display_name} ({inferred_id})"
            if inferred_id in csv_data:
                toml_data['current_amount'] = csv_data[inferred_id]
                if toml_data.get('freeze',False):
                    toml_data['target_amount'] = toml_data['current_amount']
                    # delete 'freeze' key
                    del toml_data['freeze']
                altered_entries.append(display_entry)
                csv_ids.discard(inferred_id)  # Remove found ID from the set
            else:
                not_found_entries.append(display_entry)

    # Update the TOML data
    update_toml(toml_dict)

    print("\n###############################################")

    # Print the results
    print("Altered entries:")
    for entry in altered_entries:
        print(f" - {entry}")

    print("\nIDs in TOML but not in CSV:")
    for entry in not_found_entries:
        print(f" - {entry}")

    print("\nIDs in CSV but not in TOML:")
    for csv_id in csv_ids:
        print(f" - {csv_id}")

    print("\n###############################################\n")
    
    return toml_dict

if __name__ == "__main__":
    toml_folder_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph"
    for toml_file_name in [
        'investments_manifest.md',
        'elisa_manifest.md',
    ]:
        # Read the TOML file
        print(f"Updating {toml_file_name}...")
        toml_dict = toml.load(toml_folder_path / toml_file_name)

        for bank in ["comdirect", "ing"]:
            csv_folder = Path(__file__).parent.parent.parent / "data" / "input" / "depot" / bank
            most_recent_csv_file = sorted(csv_folder.glob("*.csv"))[-1]
            csv_file_path = csv_folder / most_recent_csv_file

            print(f"Updating with csv from '{bank}': {csv_file_path}")

            toml_dict = update_dict_from_csv(
                toml_dict=toml_dict,
                csv_file_path=csv_file_path,
                bank=bank,
            )
            

        # Write the updated TOML data back to the file
        toml_file_path_out=toml_folder_path / toml_file_name.replace("_manifest","")
        with open(toml_file_path_out, 'w') as toml_file:
            toml.dump(toml_dict, toml_file)
