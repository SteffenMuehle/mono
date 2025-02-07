source_folder=$(cat ./in/source_folder.txt)
target_folder=$(cat ./in/target_folder.txt)

echo "source_folder: $source_folder"
echo "target_folder: $target_folder"
echo ""
echo "-----------------"

while IFS= read -r line
do
  echo "copying $line from source_folder to target_folder..."
  mkdir -p "$(dirname "$target_folder/$line")"
  cp -p "$source_folder/$line" "$target_folder/$line"
done < ./out/source_only.txt

echo "cp_source_only_to_target.sh" >> ./out/actions.log