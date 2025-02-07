# Always
rm -rf out/*
python scan.py
code out/*
touch out/actions.log
./cp_source_only_to_target.sh

# Review first (optional)
./cp_intersection_with_source_newer_to_target.sh
./cp_intersection_with_target_newer_to_source.sh

# Either or (if target_only not empty)
./cp_target_only_to_source.sh
./rm_target_only_in_target.sh

# Finally (always)
./log.sh