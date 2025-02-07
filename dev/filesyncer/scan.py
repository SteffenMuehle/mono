import os
from pathlib import Path
from datetime import datetime

curr_folder_path = Path(__file__).parent

file_with_ignored_dirs = curr_folder_path / "in" / ".syncignore"
if file_with_ignored_dirs.exists():
    IGNORED_DIRECTORIES = file_with_ignored_dirs.read_text().splitlines()
    print(f"Ignored directories: {IGNORED_DIRECTORIES}")

def read_directory_content_to_set(
    directory: str,
):
    file_paths = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), directory)
            if all([not str(file_path).startswith(dir) for dir in IGNORED_DIRECTORIES]):    
                file_paths.append(file_path)
    
    return set(file_paths)


def write_set_to_file(
    content: set,
    output_filepath: Path
):
    with open(output_filepath, 'w') as f:
        for file_path in sorted(list(content)):
            f.write(file_path + '\n')



in_path = curr_folder_path / "in"
source_folder_path_file = in_path / "source_folder.txt"
target_folder_path_file = in_path / "target_folder.txt"
source_folder_path = source_folder_path_file.read_text()
target_folder_path = target_folder_path_file.read_text()

out_path = curr_folder_path / "out"

print(f'Current folder path: {curr_folder_path}')
print(f'Source folder path: {source_folder_path}')
print(f'Target folder path: {target_folder_path}')

source_content = read_directory_content_to_set(source_folder_path)
write_set_to_file(
    content=source_content,
    output_filepath=out_path / 'source.txt',
)

target_content = read_directory_content_to_set(target_folder_path)
write_set_to_file(
    content=target_content,
    output_filepath=out_path / 'target.txt',
)

source_only = source_content - target_content
write_set_to_file(
    content=source_only,
    output_filepath=out_path / 'source_only.txt',
)

target_only = target_content - source_content
write_set_to_file(
    content=target_only,
    output_filepath=out_path / 'target_only.txt',
)

union = source_content.union(target_content)
write_set_to_file(
    content=union,
    output_filepath=out_path / 'union.txt',
)

intersection = source_content.intersection(target_content)
write_set_to_file(
    content=intersection,
    output_filepath=out_path / 'intersection.txt',
)

# loop through the intersection and find files with different modification dates:
intersection_with_source_newer = set()
intersection_with_target_newer = set()

for file in intersection:
    file1_path = os.path.join(source_folder_path, file)
    file2_path = os.path.join(target_folder_path, file)

    if os.path.exists(file1_path) and os.path.exists(file2_path):
        source_time_sec = os.path.getmtime(file1_path)
        target_time_sec = os.path.getmtime(file2_path)
        diff_sec = source_time_sec - target_time_sec

        if diff_sec > 1.0:
            intersection_with_source_newer.add(file)
        if diff_sec < -1.0:
            intersection_with_target_newer.add(file)

write_set_to_file(
    content=intersection_with_source_newer,
    output_filepath=out_path / 'intersection_with_source_newer.txt',
)
write_set_to_file(
    content=intersection_with_target_newer,
    output_filepath=out_path / 'intersection_with_target_newer.txt',
)