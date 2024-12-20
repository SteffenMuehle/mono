if __name__ == "__main__":
    import sys
    print(sys.path)
    

import os
import json

def generate_index(directory):
    index = {}
    for root, dirs, files in os.walk(directory):
        relative_path = os.path.relpath(root, directory)
        index[relative_path] = files
    return index

def write_index_to_json(index, output_file):
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python file_indexer.py <directory> <output_file>")
        sys.exit(1)
    directory = sys.argv[1]
    output_file = sys.argv[2]
    index = generate_index(directory)
    write_index_to_json(index, output_file)



    import json
import os
import shutil

def read_index_from_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)

def merge_indexes(index1, index2):
    merged_index = index1.copy()
    for key, files in index2.items():
        if key in merged_index:
            merged_index[key] = list(set(merged_index[key] + files))
        else:
            merged_index[key] = files
    return merged_index

def sync_folders(index1, index2, source_dir, target_dir):
    for relative_path, files in index1.items():
        source_path = os.path.join(source_dir, relative_path)
        target_path = os.path.join(target_dir, relative_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for file in files:
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            if not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
    
    for relative_path, files in index2.items():
        source_path = os.path.join(target_dir, relative_path)
        target_path = os.path.join(source_dir, relative_path)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for file in files:
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            if not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python folder_syncer.py <index1.json> <index2.json> <source_dir> <target_dir>")
        sys.exit(1)
    index1_file = sys.argv[1]
    index2_file = sys.argv[2]
    source_dir = sys.argv[3]
    target_dir = sys.argv[4]
    index1 = read_index_from_json(index1_file)
    index2 = read_index_from_json(index2_file)
    merged_index = merge_indexes(index1, index2)
    sync_folders(index1, index2, source_dir, target_dir)




#######
#######
#######
#######
#######

import json
import argparse

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_indexes(index1, index2):
    intersection = {}
    left_diff = {}
    right_diff = {}
    union = index1.copy()

    for key, files in index2.items():
        if key in union:
            union[key] = list(set(union[key] + files))
        else:
            union[key] = files

    for key, files in index1.items():
        if key in index2:
            intersection[key] = list(set(files) & set(index2[key]))
            left_diff[key] = list(set(files) - set(index2[key]))
            right_diff[key] = list(set(index2[key]) - set(files))
        else:
            left_diff[key] = files

    for key, files in index2.items():
        if key not in index1:
            right_diff[key] = files

    return intersection, left_diff, right_diff, union

def main():
    parser = argparse.ArgumentParser(description="Compare two JSON index files.")
    parser.add_argument("file1", help="Path to the first JSON file")
    parser.add_argument("file2", help="Path to the second JSON file")
    args = parser.parse_args()

    index1 = read_json(args.file1)
    index2 = read_json(args.file2)

    intersection, left_diff, right_diff, union = compare_indexes(index1, index2)

    print("Intersection:")
    print(json.dumps(intersection, indent=4))

    print("\nLeft Diff (in file1 but not in file2):")
    print(json.dumps(left_diff, indent=4))

    print("\nRight Diff (in file2 but not in file1):")
    print(json.dumps(right_diff, indent=4))

    print("\nUnion:")
    print(json.dumps(union, indent=4))

if __name__ == "__main__":
    main()