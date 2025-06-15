#!/bin/bash

current_path=$(pwd)
last_two_dirs=$(echo $current_path | awk -F '/' '{print $(NF-1)"/"$NF}')

export base_dir=tmp/logs

for dir in $base_dir/*; do
    export project=Search-R1
    experiment_name=$(basename "$dir")
    export experiment_name
    python verl/utils/jsonlogger.py
done


if [ -f "../../mv.sh" ]; then
    cd ../..
    ./mv.sh ${last_two_dirs}/tmp/logs
    cd ${last_two_dirs}
fi
