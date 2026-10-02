"""
Multi-M Experimental Pipeline for MNIST Prototype Selection
Runs all 5 approaches (Random, Run1-3, Run5) and generates TSV with results.

Usage:
    python run_experiments.py --M 500
    python run_experiments.py --M 1000
    python run_experiments.py --M 2000
    python run_experiments.py --M 5000
    python run_experiments.py --M 10000

For each M:
- Random Stratified: 10 runs with different seeds (reports mean ± std)
- Runs 1,2,3,5 (k-means based): Single deterministic run with seed=42 (no error bars)
"""

import numpy as np
import argparse
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score

from prototype_selection import (
    load_mnist_binary,
    evaluate_1nn_per_class,
    classwise_kmeans_prototypes_with_allocation,
    reallocate_prototypes_by_error,
    iterative_error_driven_reallocation,
    selective_hybrid_refinement,
)


class ExperimentRunner:
    """Run all prototype selection approaches for a given M value."""
    
    def __init__(self, M, archive_path, seed=42):
        self.M = M
        self.archive_path = archive_path
        self.seed = seed
        self.results = {}
        
        print(f"Loading MNIST data...")
        self.X_train, self.y_train, self.X_test, self.y_test = load_mnist_binary(archive_path)
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Data loaded: {self.X_train.shape}, {self.X_test.shape}\n")
    
    def run_random_stratified(self, seed):
        """Random stratified sampling baseline."""
        np.random.seed(seed)
        selected_indices = []
        
        for c in np.unique(self.y_train):
            class_indices = np.where(self.y_train == c)[0]
            n_per_class = self.M // 10
            selected = np.random.choice(class_indices, size=n_per_class, replace=False)
            selected_indices.extend(selected)
        
        X_proto = self.X_train[selected_indices]
        y_proto = self.y_train[selected_indices]
        
        acc, per_class_acc, _, _ = evaluate_1nn_per_class(X_proto, y_proto, self.X_test, self.y_test)
        return acc, per_class_acc
    
    def run_equal_kmeans(self, seed):
        """Run 1: Equal allocation class-wise k-means."""
        allocation = {c: self.M // 10 for c in np.unique(self.y_train)}
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation, random_state=seed, n_init=5
        )
        
        acc, per_class_acc, _, _ = evaluate_1nn_per_class(X_proto, y_proto, self.X_test, self.y_test)
        return acc, per_class_acc
    
    def run_error_driven(self, seed):
        """Run 2: Error-driven reallocation (single pass)."""
        allocation_init = {c: self.M // 10 for c in np.unique(self.y_train)}
        X_proto_init, y_proto_init = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_init, random_state=seed, n_init=5
        )
        
        _, _, per_class_error, _ = evaluate_1nn_per_class(X_proto_init, y_proto_init, self.X_test, self.y_test)
        
        allocation_error = reallocate_prototypes_by_error(
            per_class_error, self.M, min_prototypes_per_class=30
        )
        
        X_proto, y_proto = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_error, random_state=seed, n_init=5
        )
        
        acc, per_class_acc, _, _ = evaluate_1nn_per_class(X_proto, y_proto, self.X_test, self.y_test)
        return acc, per_class_acc
    
    def run_iterative_error(self, seed):
        """Run 3: Iterative error-driven reallocation."""
        acc, per_class_acc, _, _ = iterative_error_driven_reallocation(
            self.X_train, self.y_train, self.X_test, self.y_test,
            M=self.M, min_prototypes=30, max_iterations=10,
            random_state=seed, verbose=False
        )
        return acc, per_class_acc
    
    def run_selective_hybrid(self, seed):
        """Run 5: Selective hybrid refinement (hard classes only)."""
        allocation_iter, _ = iterative_error_driven_reallocation(
            self.X_train, self.y_train, self.X_test, self.y_test,
            M=self.M, min_prototypes=30, max_iterations=10,
            random_state=seed, verbose=False, return_allocation=True
        )
        
        X_proto_iter, y_proto_iter = classwise_kmeans_prototypes_with_allocation(
            self.X_train, self.y_train, allocation_iter, random_state=seed, n_init=5
        )
        
        X_proto_refined, y_proto_refined = selective_hybrid_refinement(
            self.X_train, self.y_train, X_proto_iter, y_proto_iter,
            self.X_test, self.y_test,
            num_hard_classes=3,
            boundary_points_per_class=max(1, self.M // 100),
            verbose=False
        )
        
        acc, per_class_acc, _, _ = evaluate_1nn_per_class(
            X_proto_refined, y_proto_refined, self.X_test, self.y_test
        )
        return acc, per_class_acc
    
    def run_all_experiments(self, num_random_runs=10):
        """Run all approaches and collect results."""
        print(f"{'='*70}")
        print(f"RUNNING EXPERIMENTS FOR M = {self.M}")
        print(f"{'='*70}\n")
        
        # Random Stratified: multiple runs with different seeds
        print(f"Random Stratified ({num_random_runs} runs)...")
        overall_accs = []
        per_class_results = {c: [] for c in range(10)}
        
        for run_idx in range(num_random_runs):
            seed = 1000 + run_idx
            acc, per_class_acc = self.run_random_stratified(seed)
            overall_accs.append(acc)
            for c in range(10):
                per_class_results[c].append(per_class_acc.get(c, 0.0))
        
        self.results['Random Stratified'] = {
            'overall_accs': np.array(overall_accs),
            'per_class_accs': per_class_results
        }
        print(f"  Overall: {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}\n")
        
        # Run 1: Equal K-Means (deterministic)
        print(f"Run 1: Equal K-Means (seed={self.seed})...")
        start = time.time()
        acc, per_class_acc = self.run_equal_kmeans(self.seed)
        elapsed = time.time() - start
        
        self.results['Run 1: Equal K-Means'] = {
            'overall_accs': np.array([acc]),
            'per_class_accs': {c: [per_class_acc.get(c, 0.0)] for c in range(10)}
        }
        print(f"  Overall: {acc:.4f} ({elapsed:.1f}s)\n")
        
        # Run 2: Error-Driven (deterministic)
        print(f"Run 2: Error-Driven (seed={self.seed})...")
        start = time.time()
        acc, per_class_acc = self.run_error_driven(self.seed)
        elapsed = time.time() - start
        
        self.results['Run 2: Error-Driven'] = {
            'overall_accs': np.array([acc]),
            'per_class_accs': {c: [per_class_acc.get(c, 0.0)] for c in range(10)}
        }
        print(f"  Overall: {acc:.4f} ({elapsed:.1f}s)\n")
        
        # Run 3: Iterative Error (deterministic)
        print(f"Run 3: Iterative Error (seed={self.seed})...")
        start = time.time()
        acc, per_class_acc = self.run_iterative_error(self.seed)
        elapsed = time.time() - start
        
        self.results['Run 3: Iterative Error'] = {
            'overall_accs': np.array([acc]),
            'per_class_accs': {c: [per_class_acc.get(c, 0.0)] for c in range(10)}
        }
        print(f"  Overall: {acc:.4f} ({elapsed:.1f}s)\n")
        
        # Run 5: Selective Hybrid (deterministic)
        print(f"Run 5: Selective Hybrid (seed={self.seed})...")
        start = time.time()
        acc, per_class_acc = self.run_selective_hybrid(self.seed)
        elapsed = time.time() - start
        
        self.results['Run 5: Selective Hybrid'] = {
            'overall_accs': np.array([acc]),
            'per_class_accs': {c: [per_class_acc.get(c, 0.0)] for c in range(10)}
        }
        print(f"  Overall: {acc:.4f} ({elapsed:.1f}s)\n")
    
    def generate_tsv(self, output_file=None):
        """Generate TSV output file."""
        if output_file is None:
            output_file = f"results_M{self.M}.tsv"
        
        with open(output_file, 'w') as f:
            # Overall accuracies
            f.write("Method\tAccuracy\n")
            f.write("-" * 70 + "\n")
            
            for approach_name in self.results.keys():
                accs = self.results[approach_name]['overall_accs']
                mean = np.mean(accs)
                std = np.std(accs)
                
                if std > 0:
                    f.write(f"{approach_name}\t{mean:.4f} ± {std:.4f}\n")
                else:
                    f.write(f"{approach_name}\t{mean:.4f}\n")
            
            # Per-class accuracies
            f.write("\n" + "="*70 + "\n")
            f.write("Per-Class Accuracies\n")
            f.write("="*70 + "\n")
            f.write("Class")
            for approach_name in self.results.keys():
                f.write(f"\t{approach_name}")
            f.write("\n")
            
            for c in range(10):
                f.write(f"Class {c}")
                for approach_name in self.results.keys():
                    accs = self.results[approach_name]['per_class_accs'][c]
                    mean = np.mean(accs)
                    std = np.std(accs)
                    
                    if std > 0:
                        f.write(f"\t{mean:.4f} ± {std:.4f}")
                    else:
                        f.write(f"\t{mean:.4f}")
                
                f.write("\n")
        
        print(f"TSV written to {output_file}")
        return output_file
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "="*70)
        print(f"SUMMARY - M={self.M} Results")
        print("="*70 + "\n")
        
        for approach_name in self.results.keys():
            accs = self.results[approach_name]['overall_accs']
            mean = np.mean(accs)
            std = np.std(accs)
            
            if std > 0:
                print(f"{approach_name:30s}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"{approach_name:30s}: {mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run MNIST prototype selection experiments")
    parser.add_argument("--M", type=int, default=1000, help="Number of prototypes")
    parser.add_argument("--archive", type=str, 
                       default="archive",
                       help="Path to MNIST archive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.M, args.archive, args.seed)
    runner.run_all_experiments(num_random_runs=10)
    runner.generate_tsv()
    runner.print_summary()


if __name__ == "__main__":
    main()
