import toml
from pathlib import Path

input_path = Path(__file__).parent.parent.parent / "data" / "input"
file_name = "stuff.toml"
file_path = input_path / file_name

some_dict = toml.load(file_path)

if __name__ == "__main__":
    print(some_dict)
