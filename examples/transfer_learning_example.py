"""
Example: Transfer Learning
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates transfer learning with pre-trained models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.transfer_learning import (
    create_transfer_learning_model,
    fine_tune_model
)

def main():
    """Main function."""
    print("=" * 60)
    print("Transfer Learning Example")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    
    # Create transfer learning model
    print("\nCreating transfer learning model with MobileNet...")
    model, base_model = create_transfer_learning_model(
        base_model_name='MobileNet',
        num_classes=10,
        input_shape=(224, 224, 3)
    )
    
    print("\nModel Summary:")
    model.summary(show_trainable=True)
    
    print("\nTransfer learning model created successfully!")
    print("You can now train this model on your dataset.")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
