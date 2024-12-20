import os
import sys
import json
from pathlib import Path

def generate_index(directory: str) -> dict:
    index = {}
    for root, _, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        if files:
            index[relative_path] = files
    return _sort_dict(index)


def _sort_dict(data):
    if isinstance(data, dict):
        res = {}
        for k in sorted(data.keys(), key=str.lower):
            res[k] = _sort_dict(data[k])
        return res
    else:
        return data


def write_index_to_json(index: dict, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=4)


def read_index_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    absolute_path_of_directory_to_scan = sys.argv[1]
    output_folder = Path(__file__).parent.parent.parent / "output"
    output_filename = "test_index.json"
    output_file = output_folder / output_filename

    index = generate_index(absolute_path_of_directory_to_scan)
    write_index_to_json(index, output_file)
