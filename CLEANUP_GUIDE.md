# Cleanup Guide for __pycache__ and Data Folders

**Author**: RSK World  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in  
**Phone**: +91 93305 39277

## Overview

This guide explains how to manage `__pycache__` folders and the `data` directory in the TensorFlow Deep Learning project.

## __pycache__ Folders

### What are __pycache__ folders?

Python automatically creates `__pycache__` directories to store compiled bytecode (`.pyc` files) for faster module loading. These are generated automatically and don't need to be committed to version control.

### Current Status

- ✅ `__pycache__/` is already in `.gitignore`
- ✅ All `.pyc`, `.pyo`, `.pyd` files are ignored
- ✅ No `__pycache__` folders currently exist in the project

### Cleaning __pycache__ Folders

#### Method 1: Using Python Script (Cross-platform)
```bash
python scripts/clean_cache.py
```

#### Method 2: Using Shell Script (Linux/Mac)
```bash
bash scripts/cleanup.sh
```

#### Method 3: Using Batch Script (Windows)
```cmd
scripts\cleanup.bat
```

#### Method 4: Manual Cleanup

**Windows (PowerShell):**
```powershell
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -File | Remove-Item -Force
```

**Linux/Mac:**
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

**Python:**
```python
import os
import shutil

for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))
    for file in files:
        if file.endswith('.pyc'):
            os.remove(os.path.join(root, file))
```

## Data Folder

### Current Structure

The `data/` folder contains:
- Generated datasets (`.npy`, `.csv` files)
- Metadata files (`.json`)
- Visualizations (if generated)
- README.md documentation

### Git Configuration

The `.gitignore` is configured to:
- ✅ Ignore generated data files (`*.npy`, `*.csv`, `*.json`)
- ✅ Keep directory structure (`.gitkeep` file)
- ✅ Keep documentation (`README.md`)

### Data Folder Contents

```
data/
├── .gitkeep              # Keeps folder in git
├── README.md            # Documentation (tracked)
├── classification_X.npy # Generated (ignored)
├── classification_y.npy  # Generated (ignored)
├── regression_X.npy     # Generated (ignored)
├── images_X.npy         # Generated (ignored)
├── sequences_X.npy      # Generated (ignored)
├── tabular.csv          # Generated (ignored)
└── *_metadata.json      # Generated (ignored)
```

### Regenerating Data

If you need to regenerate all data:
```bash
python scripts/generate_data_standalone.py
```

### Cleaning Data Folder

To remove all generated data (but keep structure):
```bash
# Remove all .npy files
find data -name "*.npy" -delete

# Remove all .csv files
find data -name "*.csv" -delete

# Remove all .json metadata files
find data -name "*_metadata.json" -delete
```

Or use Python:
```python
import os
import glob

data_dir = './data'
for pattern in ['*.npy', '*.csv', '*_metadata.json']:
    for file in glob.glob(os.path.join(data_dir, pattern)):
        os.remove(file)
        print(f"Removed: {file}")
```

## Best Practices

### 1. Before Committing
- Run cleanup script to remove `__pycache__` folders
- Verify `.gitignore` is working correctly
- Don't commit generated data files

### 2. Regular Maintenance
- Clean `__pycache__` folders periodically
- Regenerate data if needed for testing
- Keep data directory structure intact

### 3. CI/CD Integration
Add cleanup to your CI/CD pipeline:
```yaml
# Example GitHub Actions
- name: Clean __pycache__
  run: python scripts/clean_cache.py
```

## Troubleshooting

### Issue: __pycache__ folders keep appearing
**Solution**: This is normal! Python creates them automatically. Just run the cleanup script before committing.

### Issue: Data files are being tracked by git
**Solution**: Check `.gitignore` and ensure patterns are correct. Remove tracked files:
```bash
git rm --cached data/*.npy
git rm --cached data/*.csv
```

### Issue: Can't delete __pycache__ folders
**Solution**: Make sure no Python processes are using them. Close IDEs and Python interpreters, then try again.

## Scripts Available

1. **scripts/clean_cache.py** - Python cleanup script (recommended)
2. **scripts/cleanup.sh** - Shell script for Linux/Mac
3. **scripts/cleanup.bat** - Batch script for Windows

## Summary

- ✅ `__pycache__` folders are ignored by git
- ✅ Generated data files are ignored by git
- ✅ Directory structure is preserved
- ✅ Cleanup scripts are available
- ✅ Documentation is tracked

All cleanup operations are safe and won't affect your source code or project structure.
