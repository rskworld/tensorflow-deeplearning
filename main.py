"""
TensorFlow Deep Learning - Main Entry Point
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This is the main entry point for the TensorFlow Deep Learning project.
Run different modules to explore various deep learning concepts.
"""

import sys
import argparse

def main():
    """
    Main function to run different modules of the project.
    """
    parser = argparse.ArgumentParser(
        description='TensorFlow Deep Learning Project - RSK World',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --module neural_networks
  python main.py --module cnns
  python main.py --module rnns
  python main.py --module custom_layers
  python main.py --module training
  python main.py --module deployment
        """
    )
    
    parser.add_argument(
        '--module',
        type=str,
        choices=['neural_networks', 'cnns', 'rnns', 'custom_layers', 'training', 'deployment'],
        default='neural_networks',
        help='Module to run (default: neural_networks)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorFlow Deep Learning Project")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    print()
    
    if args.module == 'neural_networks':
        print("Running Neural Networks Module...")
        from src.neural_networks import example_usage
        example_usage()
    
    elif args.module == 'cnns':
        print("Running CNNs Module...")
        from src.cnns import example_usage
        example_usage()
    
    elif args.module == 'rnns':
        print("Running RNNs Module...")
        from src.rnns import example_usage
        example_usage()
    
    elif args.module == 'custom_layers':
        print("Running Custom Layers Module...")
        from src.custom_layers import example_usage
        example_usage()
    
    elif args.module == 'training':
        print("Running Model Training Module...")
        from src.model_training import example_usage
        example_usage()
    
    elif args.module == 'deployment':
        print("Running Model Deployment Module...")
        from src.model_deployment import example_usage
        example_usage()
    
    print("\n" + "=" * 60)
    print("Execution completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
