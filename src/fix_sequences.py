# src/fix_sequences.py - CORRECTED NORMALIZATION
import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib

def create_long_sequences(window_size=100, step_size=50):
    """
    CREATE LONG SEQUENCES for 80%+ accuracy
    Current: 50 timesteps (0.5s) → Target: 100-200 timesteps (1-2s)
    """
    print("CREATING LONG SEQUENCES FOR 80%+ ACCURACY")
    print("=" * 60)
    
    # Base directory
    BASE_DIR = r"F:\data-analysis-course-project-2025"
    
    # Paths
    processed_path = os.path.join(BASE_DIR, "data", "processed")
    sequences_path = os.path.join(BASE_DIR, "data", "sequences")
    
    # Create output directory
    os.makedirs(sequences_path, exist_ok=True)
    
    # Check if already exists
    x_path = os.path.join(sequences_path, "X_train_100.npy")
    y_path = os.path.join(sequences_path, "y_train_100.npy")
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading existing long sequences...")
        X_long = np.load(x_path)
        y_long = np.load(y_path)
        print(f"   Loaded: {X_long.shape}")
        return X_long, y_long
    
    # ========== LOAD ORIGINAL DATA ==========
    print("\nLoading original processed data...")
    try:
        X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).values.ravel()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    print("Original data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_test:  {y_test.shape}")
    
    # ========== SELECT IMPORTANT FEATURES ==========
    print("\nSelecting important features (from README)...")
    
    # According to README: Use ±16g accelerometer (better calibrated)
    important_features = [
        'heart_rate',
        'hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
        'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z',
        'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z'
    ]
    
    # Keep only available features
    available_features = [f for f in important_features if f in X_train.columns]
    print(f"   Available features: {len(available_features)}/{len(important_features)}")
    print(f"   Features: {available_features}")
    
    # Select features
    X_train_selected = X_train[available_features].values
    X_test_selected = X_test[available_features].values
    
    # ========== NORMALIZE FEATURES (CRITICAL FIX) ==========
    print("\nNormalizing features (z-score normalization)...")
    
    # CORRECT: Normalize each feature column independently
    # Shape: (samples, features) -> scale each feature column
    scaler = StandardScaler()
    
    # Fit on training, transform both
    X_train_normalized = scaler.fit_transform(X_train_selected)
    X_test_normalized = scaler.transform(X_test_selected)
    
    print("   Normalization complete:")
    print(f"   Training - Mean: {X_train_normalized.mean():.3f}, Std: {X_train_normalized.std():.3f}")
    print(f"   Test - Mean: {X_test_normalized.mean():.3f}, Std: {X_test_normalized.std():.3f}")
    
    # Check per-feature normalization
    print("\n   Per-feature statistics (should be mean≈0, std≈1):")
    for i, feature_name in enumerate(available_features):
        feat_mean = X_train_normalized[:, i].mean()
        feat_std = X_train_normalized[:, i].std()
        print(f"      {feature_name}: mean={feat_mean:.3f}, std={feat_std:.3f}")
    
    # Save the scaler for future use
    scaler_path = os.path.join(sequences_path, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved to: {scaler_path}")
    
    # ========== CREATE LONG SEQUENCES ==========
    print(f"\nCreating sequences: window={window_size}, step={step_size}")
    print(f"   (Current: 50 timesteps -> New: {window_size} timesteps)")
    
    def create_sequences(features, labels, window, step):
        """Helper function to create sequences"""
        sequences = []
        sequence_labels = []
        
        for start in range(0, len(features) - window + 1, step):
            seq = features[start:start + window]
            # Use mode (most common) label in window
            label = stats.mode(labels[start:start + window], keepdims=False)[0]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    # Create training sequences
    X_train_seq, y_train_seq = create_sequences(
        X_train_normalized, y_train, window_size, step_size
    )
    
    # Create test sequences (no overlap between subjects already guaranteed)
    X_test_seq, y_test_seq = create_sequences(
        X_test_normalized, y_test, window_size, step_size
    )
    
    print("LONG SEQUENCES CREATED:")
    print(f"   X_train_seq: {X_train_seq.shape}")
    print(f"   y_train_seq: {y_train_seq.shape}")
    print(f"   X_test_seq:  {X_test_seq.shape}")
    print(f"   y_test_seq:  {y_test_seq.shape}")
    
    # ========== VERIFY FINAL SEQUENCE NORMALIZATION ==========
    print("\nFinal sequence verification:")
    print(f"   Sequence shape: {X_train_seq.shape}")
    print(f"   Overall mean: {X_train_seq.mean():.3f} (should be ~0)")
    print(f"   Overall std: {X_train_seq.std():.3f} (should be ~1)")
    
    # Check per-feature in sequences
    print("   Per-feature in sequences:")
    for i, feature_name in enumerate(available_features):
        feat_data = X_train_seq[:, :, i].flatten()
        print(f"      {feature_name}: mean={feat_data.mean():.3f}, std={feat_data.std():.3f}")
    
    # ========== SAVE SEQUENCES ==========
    print("\nSaving sequences...")
    
    # Save training
    np.save(os.path.join(sequences_path, "X_train_100.npy"), X_train_seq)
    np.save(os.path.join(sequences_path, "y_train_100.npy"), y_train_seq)
    
    # Save test
    np.save(os.path.join(sequences_path, "X_test_100.npy"), X_test_seq)
    np.save(os.path.join(sequences_path, "y_test_100.npy"), y_test_seq)
    
    # Save feature list
    with open(os.path.join(sequences_path, "selected_features.txt"), 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")
    
    print(f"Saved to: {sequences_path}")
    
    return X_train_seq, y_train_seq

def check_sequence_quality(X, y):
    """
    Check if sequences are good quality
    """
    print("\nSEQUENCE QUALITY CHECK:")
    print(f"   Shape: {X.shape}")
    print(f"   Samples: {len(X):,}")
    print(f"   Timesteps per sample: {X.shape[1]}")
    print(f"   Features per timestep: {X.shape[2]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Check for NaN
    nan_count = np.isnan(X).sum()
    print(f"   NaN values: {nan_count:,} ({nan_count/X.size*100:.2f}%)")
    
    # Check normalization
    print("Normalization check:")
    print(f"      Overall mean: {X.mean():.4f} (should be ~0)")
    print(f"      Overall std:  {X.std():.4f} (should be ~1)")
    
    # Check per-feature
    print(" Per-feature statistics:")
    for i in range(X.shape[2]):
        feature_data = X[:, :, i].flatten()
        print(f"      Feature {i}: mean={feature_data.mean():.3f}, std={feature_data.std():.3f}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("   Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"      Class {cls}: {count:,} ({count/len(y)*100:.1f}%)")
    
    return nan_count == 0

# Test function
if __name__ == "__main__":
    print("Testing fix_sequences.py...")
    X, y = create_long_sequences(window_size=100, step_size=50)
    if X is not None:
        check_sequence_quality(X, y)
        print("\nfix_sequences.py is working correctly!")