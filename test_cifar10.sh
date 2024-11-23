#!/bin/bash
DIRECTORY="images"

for FILE in $DIRECTORY/*.txt; do
    echo "Processing file: $FILE"
    ./a.out "$FILE"
    echo "-------------------------------------"
done

