#!/bin/bash
# cleanup.sh

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

TARGET_DIR="$1"

# Verify that the provided argument is a directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a directory or does not exist."
    exit 1
fi

echo "Deleting files starting with 'temp', 'output_' and 'fits_' in directory: $TARGET_DIR"

# Find and delete matching files
find "$TARGET_DIR" -type f \( -name 'temp*' -o -name 'output_*' -o -name 'fits_*' -o -name 'pred*.txt' \) -exec rm -v {} \;

echo "Deleting __pycache__ directories in directory: $TARGET_DIR"
# Find and delete __pycache__ directories recursively
find "$TARGET_DIR" -type d -name '__pycache__' -exec rm -rfv {} \;

echo "Cleanup complete."
