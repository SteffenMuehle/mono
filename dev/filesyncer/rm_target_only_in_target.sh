source_folder=$(cat ./in/source_folder.txt)
target_folder=$(cat ./in/target_folder.txt)

echo "source_folder: $source_folder"
echo "target_folder: $target_folder"
echo ""
echo "-----------------"

while IFS= read -r line
do
  echo "removing $line from target_folder..."
  rm "$target_folder/$line"
done < ./out/target_only.txt

echo "rm_target_only_in_target.sh" >> ./out/actions.log