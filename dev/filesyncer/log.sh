target_folder=$(cat ./in/target_folder.txt)

curr_date=$(date +"%Y-%m-%d_%H-%M-%S")
new_log_folder_name="$target_folder/sync/logs/$curr_date"
echo "Creating new log folder: $new_log_folder_name"
mkdir -p $new_log_folder_name
echo "Copying files to new log folder:\n$(ls 'out')"
cp ./out/* $new_log_folder_name
echo "Copying scripts to new log folder:\n$(ls | grep .sh)"
cp ./out/* $new_log_folder_name