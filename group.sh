#!/bin/bash

# Group all csv's in the given folder into one and put it somewhere else
if [ $# -ne 2 ]; then
    echo "Usage: ./group.sh original new-file-with-all-csv"
    exit 1
fi

> $2
for file in $1/*; do
    echo $file >> $2
done