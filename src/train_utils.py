import numpy as np
import tensorflow as tf  # MUST keep this!
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
# ========================================

# Base directory function
def get_base_dir():
    """Get base directory from environment"""
    base = os.environ.get('HAR_BASE_DIR', r"F:\data-analysis-course-project-2025")
    return base

def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced dataset
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    print("📊 Class weights for imbalanced data:")
    for cls, weight in class_weights.items():
        print(f"   Class {cls}: {weight:.3f}")
    
    return class_weights

def plot_training_history(history, model_name):
    """
    Plot training history
    """
    BASE_DIR = get_base_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    figures_dir = os.path.join(BASE_DIR, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{model_name}_training_history.png'), 
                dpi=300, bbox_inches='tight')

def save_training_results(model, history, model_name, X_test, y_test):
    """
    Save training results and evaluation metrics
    """
    BASE_DIR = get_base_dir()
    
    # Create results directory
    results_dir = os.path.join(BASE_DIR, "reports", f"{model_name}_results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"💾 Saving results to: {results_dir}")
    
    # 1. Evaluate model - uses tf.keras.Model.evaluate()
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # 2. Make predictions - uses tf.keras.Model.predict()
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 3. Generate classification report - uses sklearn
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(results_dir, "classification_report.csv"))
    
    # 4. Generate confusion matrix - uses sklearn
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))
    
    # 5. Plot confusion matrix - uses seaborn & matplotlib
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.show()
    
    # 6. Save training history - uses pandas
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(results_dir, "training_history.csv"))
    
    # 7. Save model summary
    with open(os.path.join(results_dir, "model_summary.txt"), 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 8. Save evaluation metrics - uses json
    metrics = {
        'model_name': model_name,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_accuracy': float(history.history['accuracy'][-1]),
        'training_loss': float(history.history['loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
        'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
        'num_epochs': len(history.history['accuracy']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(results_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n📊 {model_name} Evaluation Results:")
    print("=" * 60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    return test_accuracy, test_loss