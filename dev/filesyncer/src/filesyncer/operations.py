import sys
from pathlib import Path
from filesyncer.indices import read_index_from_json, write_index_to_json


def compare_indices(index_left, index_right):
    intersection = {}
    left_only = {}
    right_only = {}
    union = index_left.copy()

    for key, files in index_right.items():
        if key in union:
            union[key] = list(set(union[key] + files))
        else:
            union[key] = files

    for key, files in index_left.items():
        if key in index_right:
            intersection_files = list(set(files) & set(index_right[key]))
            if intersection_files:
                intersection[key] = intersection_files
            
            left_only_files = list(set(files) - set(index_right[key]))
            if left_only_files:
                left_only[key] = left_only_files

            right_only_files = list(set(index_right[key]) - set(files))
            if right_only_files:
                right_only[key] = right_only_files
        else:
            left_only[key] = files

    for key, files in index_right.items():
        if key not in index_left:
            right_only[key] = files

    return intersection, left_only, right_only, union


if __name__ == "__main__":
    path_to_index_1 = sys.argv[1]
    path_to_index_2 = sys.argv[2]

    index_1 = read_index_from_json(path_to_index_1)
    index_2 = read_index_from_json(path_to_index_2)

    intersection, left_only, right_only, union = compare_indices(
        index_left = index_1,
        index_right = index_2,
    )

    write_index_to_json(intersection, "output/intersection.json")
    write_index_to_json(left_only, "output/left_only.json")
    write_index_to_json(right_only, "output/right_only.json")
    write_index_to_json(union, "output/union.json")
