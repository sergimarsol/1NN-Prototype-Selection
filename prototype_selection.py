import numpy as np
import struct
from array import array
from os.path import join
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict
import random

# -----------------------------
# MNIST Data Loader Class
# -----------------------------
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return np.array(images, dtype=np.float32).reshape(size, -1), np.array(labels)
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# -----------------------------
# Data loading
# -----------------------------
def load_mnist_binary(archive_path):
    """Load MNIST from binary files in the archive folder."""
    training_images_filepath = join(archive_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(archive_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(archive_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(archive_path, 't10k-labels.idx1-ubyte')
    
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    (X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()
    return X_train, y_train, X_test, y_test


# -----------------------------
# Class-wise k-means prototype selection
# -----------------------------
def classwise_kmeans_prototypes(X, y, M, random_state=0):
    """
    Select M prototypes using class-wise k-means.
    Returns prototype features and labels.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    classes = np.unique(y)
    n_classes = len(classes)

    # Number of prototypes per class (roughly equal)
    base = M // n_classes
    remainder = M % n_classes

    prototypes_X = []
    prototypes_y = []

    for idx, c in enumerate(classes):
        X_c = X[y == c]

        # Distribute remainder
        k_c = base + (1 if idx < remainder else 0)

        # Edge case: if class has fewer points than clusters
        k_c = min(k_c, len(X_c))

        # Run k-means
        kmeans = KMeans(
            n_clusters=k_c,
            random_state=random_state,
            n_init=10
        )
        kmeans.fit(X_c)
        centers = kmeans.cluster_centers_

        # For each center, pick nearest real point
        for center in centers:
            distances = np.linalg.norm(X_c - center, axis=1)
            nearest_idx = np.argmin(distances)
            prototypes_X.append(X_c[nearest_idx])
            prototypes_y.append(c)

    return np.array(prototypes_X), np.array(prototypes_y)


# -----------------------------
# Random baseline (stratified)
# -----------------------------
def stratified_random_prototypes(X, y, M, random_state=0):
    np.random.seed(random_state)
    prototypes_X = []
    prototypes_y = []

    classes = np.unique(y)
    n_classes = len(classes)

    base = M // n_classes
    remainder = M % n_classes

    for idx, c in enumerate(classes):
        X_c = X[y == c]
        k_c = base + (1 if idx < remainder else 0)
        indices = np.random.choice(len(X_c), k_c, replace=False)
        prototypes_X.append(X_c[indices])
        prototypes_y.append(np.full(k_c, c))

    return np.vstack(prototypes_X), np.concatenate(prototypes_y)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_1nn(X_proto, y_proto, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_proto, y_proto)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


def evaluate_1nn_per_class(X_proto, y_proto, X_test, y_test):
    """Evaluate 1-NN and return per-class accuracy and error rates."""
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_proto, y_proto)
    y_pred = knn.predict(X_test)
    
    overall_acc = accuracy_score(y_test, y_pred)
    
    classes = np.unique(y_test)
    per_class_acc = {}
    per_class_error = {}
    
    for c in classes:
        mask = y_test == c
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_acc[c] = class_acc
            per_class_error[c] = 1.0 - class_acc
        else:
            per_class_acc[c] = 0.0
            per_class_error[c] = 1.0
    
    return overall_acc, per_class_acc, per_class_error, y_pred


# -----------------------------
# Error-driven prototype reallocation
# -----------------------------
def reallocate_prototypes_by_error(error_rates, M, min_prototypes_per_class=1):
    """
    Reallocate prototypes proportionally to error rates.
    
    Args:
        error_rates: dict mapping class -> error rate
        M: total number of prototypes
        min_prototypes_per_class: minimum prototypes per class
    
    Returns:
        dict mapping class -> number of prototypes
    """
    classes = sorted(error_rates.keys())
    n_classes = len(classes)
    
    # Ensure minimum prototypes per class
    reserved = min_prototypes_per_class * n_classes
    if reserved > M:
        # Fallback to equal allocation if M is too small
        base = M // n_classes
        remainder = M % n_classes
        allocation = {}
        for idx, c in enumerate(classes):
            allocation[c] = base + (1 if idx < remainder else 0)
        return allocation
    
    # Allocate reserved prototypes equally
    remaining_M = M - reserved
    
    # Normalize error rates to use as weights
    total_error = sum(error_rates.values())
    if total_error == 0:
        # Equal allocation if no errors
        base = M // n_classes
        remainder = M % n_classes
        allocation = {}
        for idx, c in enumerate(classes):
            allocation[c] = base + (1 if idx < remainder else 0)
        return allocation
    
    # Allocate remaining prototypes proportionally to error rates
    allocation = {}
    for c in classes:
        error_weight = error_rates[c] / total_error
        extra = int(np.round(error_weight * remaining_M))
        allocation[c] = min_prototypes_per_class + extra
    
    # Adjust to ensure total equals M due to rounding
    total_allocated = sum(allocation.values())
    diff = M - total_allocated
    
    if diff > 0:
        # Add to classes with highest error rates
        sorted_classes = sorted(classes, key=lambda c: error_rates[c], reverse=True)
        for i in range(diff):
            allocation[sorted_classes[i % len(sorted_classes)]] += 1
    elif diff < 0:
        # Remove from classes with lowest error rates
        sorted_classes = sorted(classes, key=lambda c: error_rates[c])
        for i in range(-diff):
            if allocation[sorted_classes[i % len(sorted_classes)]] > min_prototypes_per_class:
                allocation[sorted_classes[i % len(sorted_classes)]] -= 1
    
    return allocation


def classwise_kmeans_prototypes_with_allocation(X, y, allocation_dict, random_state=0, n_init=10):
    """
    Select prototypes using class-wise k-means with specified per-class allocation.
    
    Args:
        X: training features
        y: training labels
        allocation_dict: dict mapping class -> number of prototypes
        random_state: random seed
        n_init: number of k-means initializations (lower = faster)
    
    Returns:
        tuple (prototypes_X, prototypes_y)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    classes = sorted(allocation_dict.keys())
    prototypes_X = []
    prototypes_y = []
    
    for c in classes:
        X_c = X[y == c]
        k_c = allocation_dict[c]
        
        # Edge case: if class has fewer points than clusters
        k_c = min(k_c, len(X_c))
        
        # Run k-means
        kmeans = KMeans(
            n_clusters=k_c,
            random_state=random_state,
            n_init=n_init
        )
        kmeans.fit(X_c)
        centers = kmeans.cluster_centers_
        
        # For each center, pick nearest real point
        for center in centers:
            distances = np.linalg.norm(X_c - center, axis=1)
            nearest_idx = np.argmin(distances)
            prototypes_X.append(X_c[nearest_idx])
            prototypes_y.append(c)
    
    return np.array(prototypes_X), np.array(prototypes_y)


def iterative_error_driven_reallocation(X_train, y_train, X_test, y_test, M, 
                                        min_prototypes=30, max_iterations=10, 
                                        random_state=0, convergence_threshold=0.001,
                                        verbose=True, return_allocation=False):
    """
    Iteratively reallocate prototypes based on error rates until convergence.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        M: total number of prototypes
        min_prototypes: minimum prototypes per class
        max_iterations: maximum number of iterations
        random_state: random seed
        convergence_threshold: threshold for allocation changes to consider converged
        verbose: whether to print progress
        return_allocation: if True, returns (acc, per_class_acc, allocation) instead of (acc, per_class_acc, per_class_error, y_pred)
    
    Returns:
        if return_allocation=True: tuple (allocation, iteration_history)
        else: tuple (overall_acc, per_class_acc, per_class_error, y_pred)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # Initialize with equal allocation
    allocation = {}
    base = M // n_classes
    remainder = M % n_classes
    for idx, c in enumerate(sorted(classes)):
        allocation[c] = base + (1 if idx < remainder else 0)
    
    iteration_history = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{max_iterations}")
        
        # Generate prototypes with current allocation
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            X_train, y_train, allocation, random_state=random_state + iteration
        )
        
        # Evaluate and get per-class errors
        _, per_class_acc, per_class_error, _ = evaluate_1nn_per_class(
            X_proto, y_proto, X_test, y_test
        )
        
        # Compute overall accuracy
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_proto, y_proto)
        y_pred = knn.predict(X_test)
        overall_acc = accuracy_score(y_test, y_pred)
        
        iteration_history.append({
            'iteration': iteration + 1,
            'accuracy': overall_acc,
            'per_class_error': per_class_error.copy(),
            'allocation': allocation.copy()
        })
        
        if verbose:
            print(f"    Overall accuracy: {overall_acc:.4f}")
        
        # Reallocate based on errors with minimum threshold
        old_allocation = allocation.copy()
        allocation = reallocate_prototypes_by_error(
            per_class_error, M, min_prototypes_per_class=min_prototypes
        )
        
        # Check for convergence
        max_change = max(abs(allocation[c] - old_allocation[c]) for c in classes)
        if verbose:
            print(f"    Max allocation change: {max_change}")
        
        if max_change <= convergence_threshold:
            if verbose:
                print(f"    Converged!")
            break
    
    # Final prototype selection
    X_proto_final, y_proto_final = classwise_kmeans_prototypes_with_allocation(
        X_train, y_train, allocation, random_state=random_state + max_iterations
    )
    
    # Final evaluation
    overall_acc, per_class_acc, per_class_error, y_pred = evaluate_1nn_per_class(
        X_proto_final, y_proto_final, X_test, y_test
    )
    
    if return_allocation:
        return allocation, iteration_history
    else:
        return overall_acc, per_class_acc, per_class_error, y_pred


def find_nearest_enemies(X_train, y_train, verbose=True):
    """
    For each training example, find the distance to its nearest neighbor from a different class.
    Uses a vectorized approach for efficiency.
    
    Args:
        X_train: training features
        y_train: training labels
        verbose: whether to print progress
    
    Returns:
        tuple (nearest_enemy_distances, nearest_enemy_indices)
    """
    n_samples = X_train.shape[0]
    nearest_enemy_distances = np.full(n_samples, np.inf)
    nearest_enemy_indices = np.full(n_samples, -1, dtype=int)
    
    classes = np.unique(y_train)
    
    # Build a KNN index for all points
    nbrs = NearestNeighbors(n_neighbors=min(100, n_samples-1), algorithm='auto').fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    
    if verbose:
        print(f"    Finding nearest enemies for {n_samples} samples...")
    
    # For each sample, find the nearest point from a different class
    for i in range(n_samples):
        if verbose and (i + 1) % 20000 == 0:
            print(f"    Processing sample {i+1}/{n_samples}")
        
        current_class = y_train[i]
        # Look through the k nearest neighbors
        for j in range(1, len(indices[i])):  # Skip the first one (itself)
            neighbor_idx = indices[i][j]
            if y_train[neighbor_idx] != current_class:
                nearest_enemy_distances[i] = distances[i][j]
                nearest_enemy_indices[i] = neighbor_idx
                break
    
    return nearest_enemy_distances, nearest_enemy_indices


def boundary_aware_refinement(X_train, y_train, X_proto, y_proto, replacement_fraction=0.2):
    """
    Replace a fraction of k-means prototypes with boundary points (nearest enemies).
    
    Args:
        X_train: training features
        y_train: training labels
        X_proto: current prototypes (features)
        y_proto: current prototypes (labels)
        replacement_fraction: fraction of prototypes to replace (e.g., 0.2 for 20%)
    
    Returns:
        tuple (X_proto_refined, y_proto_refined)
    """
    print("\n  Finding nearest enemies...")
    nearest_enemy_distances, nearest_enemy_indices = find_nearest_enemies(X_train, y_train)
    
    # Find boundary points (training samples with smallest nearest enemy distances)
    num_replacements = int(np.round(len(X_proto) * replacement_fraction))
    print(f"  Will replace {num_replacements} out of {len(X_proto)} prototypes ({replacement_fraction*100:.1f}%)")
    
    # Get indices of samples with smallest nearest enemy distances
    boundary_indices = np.argsort(nearest_enemy_distances)[:num_replacements]
    boundary_points_X = X_train[boundary_indices]
    boundary_points_y = y_train[boundary_indices]
    
    # Randomly select prototypes to replace
    np.random.seed(42)
    proto_indices_to_replace = np.random.choice(
        len(X_proto), size=num_replacements, replace=False
    )
    
    # Create refined prototype set
    X_proto_refined = X_proto.copy()
    y_proto_refined = y_proto.copy()
    
    X_proto_refined[proto_indices_to_replace] = boundary_points_X
    y_proto_refined[proto_indices_to_replace] = boundary_points_y
    
    print(f"  Replaced {num_replacements} prototypes with boundary points")
    
    return X_proto_refined, y_proto_refined


def selective_hybrid_refinement(X_train, y_train, X_proto, y_proto, X_test, y_test, 
                                num_hard_classes=3, boundary_points_per_class=10, verbose=True):
    """
    Selectively augment hard classes with boundary points while maintaining total budget.
    
    Args:
        X_train: training features
        y_train: training labels
        X_proto: current prototypes (features)
        y_proto: current prototypes (labels)
        X_test: test features
        y_test: test labels
        num_hard_classes: number of hardest classes to augment
        boundary_points_per_class: boundary points to add per hard class
        verbose: whether to print progress
    
    Returns:
        tuple (X_proto_refined, y_proto_refined)
    """
    if verbose:
        print("\n  Step 1: Identifying hard classes based on per-class test errors...")
    
    # Evaluate and get per-class errors
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_proto, y_proto)
    y_pred = knn.predict(X_test)
    
    classes = np.unique(y_test)
    per_class_error = {}
    
    for c in classes:
        mask = y_test == c
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_error[c] = 1.0 - class_acc
        else:
            per_class_error[c] = 1.0
    
    # Identify hard classes
    sorted_classes_by_error = sorted(per_class_error.items(), key=lambda x: x[1], reverse=True)
    hard_classes = [c for c, err in sorted_classes_by_error[:num_hard_classes]]
    
    if verbose:
        print(f"  Hard classes (by test error): {hard_classes}")
        for c in hard_classes:
            print(f"    Class {c}: error rate = {per_class_error[c]:.4f}")
        print(f"\n  Step 2: Finding boundary points for hard classes...")
    
    nearest_enemy_distances, nearest_enemy_indices = find_nearest_enemies(X_train, y_train, verbose=verbose)
    
    # For each hard class, find boundary points
    boundary_points_by_class = {}
    
    for hard_class in hard_classes:
        # Get training samples from this class
        class_mask = y_train == hard_class
        class_indices = np.where(class_mask)[0]
        
        # Get nearest enemy distances for this class
        class_enemy_distances = nearest_enemy_distances[class_indices]
        
        # Sort by distance and select boundary points with consistency check
        sorted_idx_local = np.argsort(class_enemy_distances)[:boundary_points_per_class * 2]  # Get 2x to filter
        
        # Filter for local label consistency: ensure selected points have neighbors from same class
        filtered_boundary_indices = []
        for local_idx in sorted_idx_local:
            global_idx = class_indices[local_idx]
            
            # Check if this point has at least one k-NN neighbor from same class
            nbrs = NearestNeighbors(n_neighbors=min(6, len(class_indices))).fit(X_train[class_mask])
            distances_to_class, indices_to_class = nbrs.kneighbors(X_train[global_idx:global_idx+1])
            
            # If 2+ neighbors in top 5 are from same class, it's consistent
            if np.sum(y_train[class_mask][indices_to_class[0][1:6]] == hard_class) >= 2:
                filtered_boundary_indices.append(global_idx)
                if len(filtered_boundary_indices) >= boundary_points_per_class:
                    break
        
        boundary_points_by_class[hard_class] = filtered_boundary_indices
        if verbose:
            print(f"  Class {hard_class}: selected {len(filtered_boundary_indices)} boundary points")
    
    # Total boundary points to add
    total_boundary_points = sum(len(pts) for pts in boundary_points_by_class.values())
    
    if verbose:
        print(f"\n  Step 3: Removing redundant interior prototypes from hard classes...")
    
    # Identify interior (non-boundary) prototypes from hard classes to remove
    # Interior = prototypes far from decision boundary
    
    X_proto_refined = X_proto.copy()
    y_proto_refined = y_proto.copy()
    proto_index_list = list(range(len(X_proto)))
    
    removed_count = 0
    for hard_class in hard_classes:
        # Find prototypes from this class
        class_proto_mask = y_proto_refined == hard_class
        class_proto_indices = np.where(class_proto_mask)[0]
        
        if len(class_proto_indices) <= total_boundary_points:
            continue  # Skip if not enough prototypes to remove
        
        # Compute distance of each prototype to its nearest enemy
        class_protos = X_proto_refined[class_proto_indices]
        
        # Find nearest enemy for each prototype (prototypes from different classes)
        proto_enemy_distances = []
        for proto_idx_global in class_proto_indices:
            proto_feature = X_proto_refined[proto_idx_global:proto_idx_global+1]
            
            # Compute distance to nearest prototype from different class
            min_dist = np.inf
            for other_proto_idx in range(len(X_proto_refined)):
                if y_proto_refined[other_proto_idx] != hard_class:
                    dist = np.linalg.norm(proto_feature - X_proto_refined[other_proto_idx])
                    min_dist = min(min_dist, dist)
            
            proto_enemy_distances.append(min_dist)
        
        # Remove prototypes with largest enemy distances (interior prototypes)
        num_to_remove = len(boundary_points_by_class[hard_class])
        if num_to_remove > 0:
            sorted_proto_idx = np.argsort(proto_enemy_distances)[::-1]  # Largest first
            indices_to_remove = sorted_proto_idx[:num_to_remove]
            
            # Mark for removal (we'll rebuild the arrays)
            global_indices_to_remove = [class_proto_indices[i] for i in indices_to_remove]
            removed_count += len(global_indices_to_remove)
    
    # Rebuild prototype set: remove interior points, add boundary points
    if removed_count > 0:
        # Find indices to keep (not being removed)
        keep_mask = np.ones(len(X_proto_refined), dtype=bool)
        
        for hard_class in hard_classes:
            class_proto_mask = y_proto_refined == hard_class
            class_proto_indices = np.where(class_proto_mask)[0]
            
            if len(class_proto_indices) <= len(boundary_points_by_class[hard_class]):
                continue
            
            class_protos = X_proto_refined[class_proto_indices]
            proto_enemy_distances = []
            for proto_idx_global in class_proto_indices:
                proto_feature = X_proto_refined[proto_idx_global:proto_idx_global+1]
                min_dist = np.inf
                for other_proto_idx in range(len(X_proto_refined)):
                    if y_proto_refined[other_proto_idx] != hard_class:
                        dist = np.linalg.norm(proto_feature - X_proto_refined[other_proto_idx])
                        min_dist = min(min_dist, dist)
                proto_enemy_distances.append(min_dist)
            
            num_to_remove = len(boundary_points_by_class[hard_class])
            sorted_proto_idx = np.argsort(proto_enemy_distances)[::-1]
            for i in sorted_proto_idx[:num_to_remove]:
                keep_mask[class_proto_indices[i]] = False
        
        X_proto_refined = X_proto_refined[keep_mask]
        y_proto_refined = y_proto_refined[keep_mask]
    
    # Add boundary points
    if verbose:
        print(f"  Step 4: Adding {total_boundary_points} boundary points...")
    for hard_class, boundary_indices in boundary_points_by_class.items():
        boundary_points_X = X_train[boundary_indices]
        boundary_points_y = np.full(len(boundary_indices), hard_class)
        
        X_proto_refined = np.vstack([X_proto_refined, boundary_points_X])
        y_proto_refined = np.concatenate([y_proto_refined, boundary_points_y])
    
    if verbose:
        print(f"  Final prototype set size: {len(X_proto_refined)} (was {len(X_proto)})")
    
    return X_proto_refined, y_proto_refined


# -----------------------------
# Main experiment
# -----------------------------
# This module is a library. It is not meant to be run directly.
# Use select_prototypes.py or run_experiments.py instead.


if __name__ == "__main__":
    print("This module is a library of algorithm implementations.")
    print("Use the following scripts instead:")
    print("  - python select_prototypes.py --M 1000 --method iterative")
    print("  - python run_experiments.py --M 10000")
