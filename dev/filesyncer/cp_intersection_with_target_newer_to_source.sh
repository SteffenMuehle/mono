source_folder=$(cat ./in/source_folder.txt)
target_folder=$(cat ./in/target_folder.txt)

echo "source_folder: $source_folder"
echo "target_folder: $target_folder"
echo ""
echo "-----------------"

while IFS= read -r line
do
  echo "copying $line from source_folder to target_folder..."
  mkdir -p "$(dirname "$source_folder/$line")"
  cp -p "$target_folder/$line" "$source_folder/$line"
done < ./out/intersection_with_target_newer.txt

echo "cp_intersection_with_target_newer_to_source.sh" >> ./out/actions.log