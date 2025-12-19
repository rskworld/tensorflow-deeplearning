"""
Clean __pycache__ folders and temporary files
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script removes __pycache__ folders and .pyc files from the project.
"""

import os
import shutil
import sys

def remove_pycache(root_dir='.'):
    """
    Remove all __pycache__ directories and .pyc files.
    
    Args:
        root_dir: Root directory to search
    """
    removed_count = 0
    removed_size = 0
    
    print("=" * 60)
    print("Cleaning __pycache__ folders and .pyc files")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    print()
    
    for root, dirs, files in os.walk(root_dir):
        # Skip virtual environments and common ignore directories
        if any(skip in root for skip in ['venv', 'env', '.git', 'node_modules', '__pycache__']):
            continue
        
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                # Calculate size
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(pycache_path)
                    for filename in filenames
                )
                shutil.rmtree(pycache_path)
                removed_count += 1
                removed_size += size
                print(f"Removed: {pycache_path}")
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")
        
        # Remove .pyc, .pyo, .pyd files
        for file in files:
            if file.endswith(('.pyc', '.pyo', '.pyd')):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_count += 1
                    removed_size += size
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    print()
    print("=" * 60)
    print(f"Cleanup complete!")
    print(f"Removed {removed_count} items")
    print(f"Freed {removed_size / 1024:.2f} KB")
    print("=" * 60)

def clean_data_temp_files(data_dir='./data'):
    """
    Clean temporary files from data directory (keep structure).
    
    Args:
        data_dir: Data directory path
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print("\nCleaning temporary files from data directory...")
    
    temp_extensions = ['.tmp', '.temp', '.log']
    removed = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in temp_extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed += 1
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    if removed > 0:
        print(f"Removed {removed} temporary files from data directory")
    else:
        print("No temporary files found in data directory")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean __pycache__ and temporary files')
    parser.add_argument('--data', action='store_true', help='Also clean data directory temp files')
    parser.add_argument('--root', default='.', help='Root directory to clean (default: current directory)')
    
    args = parser.parse_args()
    
    # Clean __pycache__
    remove_pycache(args.root)
    
    # Clean data temp files if requested
    if args.data:
        clean_data_temp_files()

if __name__ == '__main__':
    main()
