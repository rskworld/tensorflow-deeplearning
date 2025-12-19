"""
Model Evaluation and Metrics with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides comprehensive model evaluation and metrics.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Model evaluation utilities.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, model):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
    
    def evaluate_classification(self, X_test, y_test, class_names=None):
        """
        Evaluate classification model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Handle one-hot encoded labels
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        return metrics
    
    def evaluate_regression(self, X_test, y_test):
        """
        Evaluate regression model.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'predictions': y_pred
        }
        
        return metrics
    
    def plot_confusion_matrix(self, X_test, y_test, class_names=None, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
            figsize: Figure size
        """
        metrics = self.evaluate_classification(X_test, y_test, class_names)
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test, class_index=0):
        """
        Plot ROC curve for binary classification.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_index: Class index for ROC curve
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        if len(y_pred_proba.shape) > 1:
            y_scores = y_pred_proba[:, class_index]
        else:
            y_scores = y_pred_proba
        
        # Handle one-hot encoded labels
        if len(y_test.shape) > 1:
            y_true = y_test[:, class_index]
        else:
            y_true = (y_test == class_index).astype(int)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, X_test, y_test, class_index=0):
        """
        Plot precision-recall curve.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_index: Class index
        """
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        if len(y_pred_proba.shape) > 1:
            y_scores = y_pred_proba[:, class_index]
        else:
            y_scores = y_pred_proba
        
        # Handle one-hot encoded labels
        if len(y_test.shape) > 1:
            y_true = y_test[:, class_index]
        else:
            y_true = (y_test == class_index).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """
        Get model summary information.
        
        Returns:
            Dictionary with model information
        """
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        summary = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'num_layers': len(self.model.layers),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }
        
        return summary

def calculate_model_complexity(model):
    """
    Calculate model complexity metrics.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with complexity metrics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Estimate model size (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    complexity = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'estimated_size_mb': model_size_mb,
        'number_of_layers': len(model.layers)
    }
    
    return complexity

def compare_models(models, X_test, y_test, model_names=None):
    """
    Compare multiple models.
    
    Args:
        models: List of models
        X_test: Test features
        y_test: Test labels
        model_names: List of model names
    
    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    
    results = []
    
    for i, model in enumerate(models):
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate_classification(X_test, y_test)
        
        model_name = model_names[i] if model_names else f'Model {i+1}'
        
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
    
    return pd.DataFrame(results)

def example_usage():
    """
    Example usage of model evaluation functions.
    """
    # Create a simple model for demonstration
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate dummy data
    X_test = np.random.randn(100, 784).astype('float32')
    y_test = np.random.randint(0, 10, 100)
    
    # Train briefly
    X_train = np.random.randn(1000, 784).astype('float32')
    y_train = np.random.randint(0, 10, 1000)
    model.fit(X_train, y_train, epochs=1, verbose=0)
    
    # Evaluate model
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate_classification(X_test, y_test)
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Get model summary
    summary = evaluator.get_model_summary()
    print(f"\nModel Summary:")
    print(f"Total Parameters: {summary['total_params']:,}")
    print(f"Trainable Parameters: {summary['trainable_params']:,}")
    
    return evaluator, metrics

if __name__ == '__main__':
    print("Model Evaluation and Metrics with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    evaluator, metrics = example_usage()
