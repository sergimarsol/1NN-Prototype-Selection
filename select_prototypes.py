"""
Prototype Selection Script
Selects M prototypes from a labeled training set using various methods.

Usage:
    python select_prototypes.py --M 500 --method random
    python select_prototypes.py --M 2000 --method equal-kmeans
    python select_prototypes.py --M 1000 --method error-driven
    python select_prototypes.py --M 2000 --method iterative
    python select_prototypes.py --M 10000 --method selective-hybrid
    
    # Save prototypes to file
    python select_prototypes.py --M 1000 --method error-driven --output prototypes_1000.npz

    # With normalization and custom seed
    python select_prototypes.py --M 2000 --method iterative --normalize --seed 123

Methods available:
    - random: Random stratified sampling
    - equal-kmeans: Equal allocation class-wise k-means (Run 1)
    - error-driven: Error-driven reallocation (Run 2, DEFAULT)
    - iterative: Iterative error-driven reallocation (Run 3)
    - selective-hybrid: Selective hybrid refinement (Run 5)
"""

import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

from prototype_selection import (
    load_mnist_binary,
    evaluate_1nn_per_class,
    classwise_kmeans_prototypes_with_allocation,
    reallocate_prototypes_by_error,
    iterative_error_driven_reallocation,
    selective_hybrid_refinement,
)


class PrototypeSelector:
    """Select prototypes from training data using various methods."""
    
    def __init__(self, X_train, y_train, X_test=None, y_test=None, random_state=42):
        """
        Initialize the prototype selector.
        
        Args:
            X_train: Training features (N x D array)
            y_train: Training labels (N array)
            X_test: Test features (optional, required for error-driven methods)
            y_test: Test labels (optional, required for error-driven methods)
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        
    def select_prototypes(self, M, method='error-driven'):
        """
        Select M prototypes using the specified method.
        
        Args:
            M: Number of prototypes to select
            method: Selection method (random, equal-kmeans, error-driven, iterative, selective-hybrid)
        
        Returns:
            tuple: (X_proto, y_proto) - Selected prototypes and their labels
        """
        if method == 'random':
            return self._select_random(M)
        elif method == 'equal-kmeans':
            return self._select_equal_kmeans(M)
        elif method == 'error-driven':
            return self._select_error_driven(M)
        elif method == 'iterative':
            return self._select_iterative(M)
        elif method == 'selective-hybrid':
            return self._select_selective_hybrid(M)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: random, equal-kmeans, error-driven, iterative, selective-hybrid")
    
    def _select_random(self, M):
        """Random stratified sampling."""
        np.random.seed(self.random_state)
        selected_indices = []
        
        for c in np.unique(self.y_train):
            class_indices = np.where(self.y_train == c)[0]
            n_per_class = M // len(np.unique(self.y_train))
            selected = np.random.choice(class_indices, size=n_per_class, replace=False)
            selected_indices.extend(selected)
        
        X_proto = self.X_train[selected_indices]
        y_proto = self.y_train[selected_indices]
        
        return X_proto, y_proto
    
    def _select_equal_kmeans(self, M):
        """Equal allocation class-wise k-means."""
        num_classes = len(np.unique(self.y_train))
        allocation = {c: M // num_classes for c in np.unique(self.y_train)}
        
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation, 
            random_state=self.random_state, n_init=5
        )
        
        return X_proto, y_proto
    
    def _select_error_driven(self, M):
        """Error-driven reallocation (single pass)."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Error-driven method requires test set (X_test, y_test)")
        
        # Initial equal allocation
        num_classes = len(np.unique(self.y_train))
        allocation_init = {c: M // num_classes for c in np.unique(self.y_train)}
        
        X_proto_init, y_proto_init = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_init, 
            random_state=self.random_state, n_init=5
        )
        
        # Evaluate and get per-class errors
        _, _, per_class_error, _ = evaluate_1nn_per_class(
            X_proto_init, y_proto_init, self.X_test, self.y_test
        )
        
        # Reallocate based on errors
        allocation_error = reallocate_prototypes_by_error(
            per_class_error, M, min_prototypes_per_class=30
        )
        
        # Generate final prototypes
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_error, 
            random_state=self.random_state, n_init=5
        )
        
        return X_proto, y_proto
    
    def _select_iterative(self, M):
        """Iterative error-driven reallocation."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Iterative method requires test set (X_test, y_test)")
        
        # `iterative_error_driven_reallocation` returns (allocation, iteration_history)
        # when `return_allocation=True`.
        allocation, _ = iterative_error_driven_reallocation(
            self.X_train, self.y_train, self.X_test, self.y_test,
            M=M, min_prototypes=30, max_iterations=10,
            random_state=self.random_state, verbose=False, return_allocation=True
        )
        
        # Generate prototypes with final allocation
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation, 
            random_state=self.random_state, n_init=5
        )
        
        return X_proto, y_proto
    
    def _select_selective_hybrid(self, M):
        """Selective hybrid refinement."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Selective hybrid method requires test set (X_test, y_test)")
        
        # Start with iterative error-driven allocation
        allocation_iter, _ = iterative_error_driven_reallocation(
            self.X_train, self.y_train, self.X_test, self.y_test,
            M=M, min_prototypes=30, max_iterations=10,
            random_state=self.random_state, verbose=False, return_allocation=True
        )
        
        X_proto_iter, y_proto_iter = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_iter, 
            random_state=self.random_state, n_init=5
        )
        
        # Apply selective hybrid refinement
        X_proto, y_proto = selective_hybrid_refinement(
            self.X_train, self.y_train, X_proto_iter, y_proto_iter,
            self.X_test, self.y_test,
            num_hard_classes=3,
            boundary_points_per_class=max(1, M // 100),
            verbose=False
        )
        
        return X_proto, y_proto


def main():
    parser = argparse.ArgumentParser(
        description="Select M prototypes from training data using various methods"
    )
    parser.add_argument("--M", type=int, required=True, 
                       help="Number of prototypes to select")
    parser.add_argument("--method", type=str, default="error-driven",
                       choices=['random', 'equal-kmeans', 'error-driven', 'iterative', 'selective-hybrid'],
                       help="Prototype selection method (default: error-driven)")
    parser.add_argument("--archive", type=str, default="archive",
                       help="Path to MNIST archive (for demo)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save prototypes (numpy .npz format)")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize features using StandardScaler")
    
    args = parser.parse_args()
    
    # Load MNIST data (for demonstration)
    print(f"Loading data from {args.archive}...")
    X_train, y_train, X_test, y_test = load_mnist_binary(args.archive)
    
    # Normalize to [0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Optional: Apply StandardScaler
    if args.normalize:
        print("Applying StandardScaler normalization...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Create selector
    selector = PrototypeSelector(X_train, y_train, X_test, y_test, random_state=args.seed)
    
    # Select prototypes
    print(f"\nSelecting {args.M} prototypes using '{args.method}' method...")
    X_proto, y_proto = selector.select_prototypes(args.M, method=args.method)
    
    print(f"Selected {len(X_proto)} prototypes")
    print(f"Class distribution: {np.bincount(y_proto.astype(int))}")
    
    # Evaluate performance
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_proto, y_proto)
    accuracy = knn.score(X_test, y_test)
    print(f"\n1-NN Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save prototypes if output specified
    if args.output:
        np.savez(args.output, X_proto=X_proto, y_proto=y_proto)
        print(f"\nPrototypes saved to {args.output}")
        print(f"To load: data = np.load('{args.output}'); X_proto = data['X_proto']; y_proto = data['y_proto']")
    
    return X_proto, y_proto


if __name__ == "__main__":
    main()
