"""
Data Generation Script
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Quick script to generate sample data for the project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_all_sample_data, visualize_generated_data

if __name__ == '__main__':
    print("Generating sample data for TensorFlow Deep Learning Project...")
    print()
    
    # Generate all data
    generate_all_sample_data(data_dir='./data')
    
    # Create visualizations
    print("\nCreating data visualizations...")
    visualize_generated_data(data_dir='./data')
    
    print("\n✓ All data generated successfully!")
    print("Check the ./data directory for generated files.")
