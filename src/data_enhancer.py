# src/data_enhancer.py - CORRECTED & COMPLETE
import numpy as np
from sklearn.utils import resample

def select_important_features(X, variance_threshold=0.05):
    """
    Select only features with sufficient variance
    Removes constant/near-constant features
    """
    print(f"Selecting important features (threshold: {variance_threshold})")
    
    # Reshape to (samples*timesteps, features) for variance calculation
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    # Calculate variance for each feature
    feature_var = np.var(X_reshaped, axis=0)
    max_var = np.max(feature_var) if np.max(feature_var) > 0 else 1.0
    
    # Avoid division by zero
    if max_var == 0:
        print("All features have zero variance!")
        return X, np.arange(X.shape[-1])
    
    normalized_var = feature_var / max_var
    
    # Select features above threshold
    selected_idx = np.where(normalized_var > variance_threshold)[0]
    removed_idx = np.where(normalized_var <= variance_threshold)[0]
    
    print(f"   Original features: {X.shape[2]}")
    print(f"   Selected features: {len(selected_idx)}")
    print(f"   Removed features:  {len(removed_idx)}")
    
    if len(selected_idx) == 0:
        print("WARNING: No features selected! Keeping all features.")
        return X, np.arange(X.shape[-1])
    
    # Return selected features
    X_selected = X[:, :, selected_idx]
    print(f"   New shape: {X_selected.shape}")
    
    return X_selected, selected_idx

def balance_classes_simple(X, y, min_samples_per_class=500):
    """
    Balance dataset by oversampling minority classes
    """
    print(f"⚖️ Balancing classes (target: {min_samples_per_class} per class)")
    
    unique_classes = np.unique(y)
    print(f"   Classes: {len(unique_classes)}")
    
    X_balanced_list = []
    y_balanced_list = []
    
    for cls in unique_classes:
        # Get samples for this class
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        y_cls = y[cls_indices]
        
        current_count = len(X_cls)
        
        if current_count < min_samples_per_class:
            # Oversample minority class
            X_resampled, y_resampled = resample(
                X_cls, y_cls,
                n_samples=min_samples_per_class,
                random_state=42,
                replace=True  # Allow sampling with replacement
            )
            print(f"   Class {cls}: {current_count:4d} → {min_samples_per_class:4d} (oversampled)")
        else:
            # Undersample or keep as is
            keep_count = min(min_samples_per_class, current_count)
            if keep_count < current_count:
                indices = np.random.choice(current_count, keep_count, replace=False)
                X_resampled = X_cls[indices]
                y_resampled = y_cls[indices]
                print(f"   Class {cls}: {current_count:4d} → {keep_count:4d} (undersampled)")
            else:
                X_resampled = X_cls
                y_resampled = y_cls
                print(f"   Class {cls}: {current_count:4d} → {keep_count:4d} (kept)")
        
        X_balanced_list.append(X_resampled)
        y_balanced_list.append(y_resampled)
    
    # Combine all classes
    if len(X_balanced_list) > 0:
        X_balanced = np.vstack(X_balanced_list)
        y_balanced = np.hstack(y_balanced_list)
    else:
        X_balanced = X
        y_balanced = y
    
    # Shuffle the dataset
    if len(X_balanced) > 0:
        idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[idx]
        y_balanced = y_balanced[idx]
    
    print(f"Balanced dataset: {X_balanced.shape}")
    print(f"   Total samples: {len(X_balanced)}")
    
    return X_balanced, y_balanced

def add_simple_augmentation(X, y, augmentation_factor=1.2):
    """
    Add simple noise augmentation
    """
    print(f"Adding simple augmentation (factor: {augmentation_factor}x)")
    
    original_count = len(X)
    target_count = int(original_count * augmentation_factor)
    
    if target_count <= original_count:
        print("No augmentation needed")
        return X, y
    
    # Calculate how many extra samples needed
    extra_needed = target_count - original_count
    print(f"   Need {extra_needed} additional samples")
    
    X_augmented = [X]
    y_augmented = [y]
    
    # Add noisy versions
    indices_to_augment = np.random.choice(original_count, extra_needed, replace=True)
    
    for idx in indices_to_augment:
        noise = np.random.normal(0, 0.01, X[idx].shape)  # Small noise
        X_augmented.append(X[idx] + noise)
        y_augmented.append(y[idx])
    
    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    
    print(f"Augmented dataset: {X_final.shape}")
    print(f"   Increased from {original_count} to {len(X_final)} samples")
    
    return X_final, y_final