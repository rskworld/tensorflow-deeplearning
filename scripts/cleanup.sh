#!/bin/bash
# Clean __pycache__ folders and temporary files
# Author: RSK World - https://rskworld.in

echo "============================================================"
echo "Cleaning __pycache__ folders and .pyc files"
echo "Author: RSK World - https://rskworld.in"
echo "============================================================"
echo

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

# Remove .pyc, .pyo files
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Remove .pyd files (Windows)
find . -type f -name "*.pyd" -delete

echo "Cleanup complete!"
echo "============================================================"
