#!/usr/bin/env python3
"""
GO Semantic Similarity vs Diffusion Distance Analysis

Computes GO semantic similarity between samples and correlates it with
diffusion distance to validate that diffusion space captures biological structure.
"""

import argparse
import ast
import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from pronto import Ontology
    PRONTO_AVAILABLE = True
except ImportError:
    PRONTO_AVAILABLE = False
    print("Warning: pronto library not available. GO semantic similarity will use Jaccard similarity instead.")

try:
    from goatools.obo_parser import GODag
    from goatools.semantic import TermCounts, resnik_sim, lin_sim, semantic_similarity
    GOATOOLS_AVAILABLE = True
except ImportError:
    GOATOOLS_AVAILABLE = False


def parse_go_terms(go_string):
    """Parse GO terms from string representation to list."""
    if pd.isna(go_string) or go_string == '' or go_string == '[]':
        return []
    if isinstance(go_string, list):
        return go_string
    if isinstance(go_string, str):
        try:
            parsed = ast.literal_eval(go_string)
            if isinstance(parsed, list):
                return [str(term).strip() for term in parsed if term]
            return []
        except (ValueError, SyntaxError):
            return []
    return []


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets of GO terms."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union if union > 0 else 0.0


def compute_go_semantic_similarity_matrix(go_terms_list, method='jaccard', go_ontology=None):
    """
    Compute pairwise GO semantic similarity matrix.
    
    Args:
        go_terms_list: List of lists, where each inner list contains GO terms for one sample
        method: 'jaccard', 'resnik', 'lin', or 'wang'
        go_ontology: GO ontology object (for semantic methods)
    
    Returns:
        Similarity matrix (n_samples x n_samples)
    """
    n_samples = len(go_terms_list)
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    if method == 'jaccard' or not GOATOOLS_AVAILABLE:
        # Use Jaccard similarity (set overlap)
        print(f"Computing Jaccard similarity for {n_samples} samples...")
        for i in tqdm(range(n_samples), desc="Computing similarities"):
            for j in range(i, n_samples):
                sim = jaccard_similarity(go_terms_list[i], go_terms_list[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    elif method in ['resnik', 'lin', 'wang'] and GOATOOLS_AVAILABLE:
        # Use semantic similarity methods
        print(f"Computing {method} semantic similarity for {n_samples} samples...")
        print("Note: This requires GO ontology file. Using Jaccard as fallback.")
        # For now, fall back to Jaccard
        for i in tqdm(range(n_samples), desc="Computing similarities"):
            for j in range(i, n_samples):
                sim = jaccard_similarity(go_terms_list[i], go_terms_list[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity_matrix


def compute_diffusion_distance_matrix(diffusion_coords, use_euclidean=True, eigenvalues=None):
    """
    Compute pairwise diffusion distance matrix.
    
    Args:
        diffusion_coords: Array of shape (n_samples, n_components) with diffusion coordinates
        use_euclidean: If True, use Euclidean distance in coordinate space
                      If False, use diffusion distance (weighted by eigenvalues)
        eigenvalues: Array of eigenvalues (required for diffusion distance)
    
    Returns:
        Distance matrix (n_samples x n_samples)
    """
    n_samples = diffusion_coords.shape[0]
    
    if use_euclidean:
        # Euclidean distance in diffusion coordinate space
        print(f"Computing Euclidean distance in diffusion space for {n_samples} samples...")
        distances = pdist(diffusion_coords, metric='euclidean')
        distance_matrix = squareform(distances)
    else:
        # Diffusion distance: weighted by 1/λ
        if eigenvalues is None:
            print("Warning: Eigenvalues not provided. Using Euclidean distance.")
            distances = pdist(diffusion_coords, metric='euclidean')
            distance_matrix = squareform(distances)
        else:
            print(f"Computing diffusion distance for {n_samples} samples...")
            n_components = min(len(eigenvalues), diffusion_coords.shape[1])
            eigenvals = eigenvalues[:n_components]
            
            # Diffusion distance: d²(x,y) = Σᵢ (ψᵢ(x) - ψᵢ(y))² / λᵢ
            distance_matrix = np.zeros((n_samples, n_samples))
            for i in tqdm(range(n_samples), desc="Computing diffusion distances"):
                for j in range(i+1, n_samples):
                    diff_vec = diffusion_coords[i, :n_components] - diffusion_coords[j, :n_components]
                    # Weight by 1/λ (higher weight for larger eigenvalues)
                    diff_dist_sq = np.sum((diff_vec ** 2) / eigenvals)
                    distance_matrix[i, j] = np.sqrt(diff_dist_sq)
                    distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix


def analyze_go_vs_diffusion_distance(
    data_frame,
    diffusion_results_file,
    go_term_column='target_biological_processes',
    output_dir=None,
    method='jaccard',
    use_euclidean=True,
    max_samples=None
):
    """
    Main analysis: Correlate GO semantic similarity with diffusion distance.
    
    Args:
        data_frame: DataFrame with alignment data and GO terms
        diffusion_results_file: Path to CSV with diffusion map results
        go_term_column: Column name containing GO terms
        output_dir: Directory to save results
        method: Similarity method ('jaccard', 'resnik', 'lin', 'wang')
        use_euclidean: Use Euclidean distance (True) or diffusion distance (False)
        max_samples: Maximum number of samples to analyze (for speed)
    
    Returns:
        Dictionary with correlation results and statistics
    """
    print("\n" + "="*60)
    print("GO Semantic Similarity vs Diffusion Distance Analysis")
    print("="*60)
    
    # Load diffusion map results
    if not os.path.exists(diffusion_results_file):
        raise FileNotFoundError(f"Diffusion results file not found: {diffusion_results_file}")
    
    diffusion_df = pd.read_csv(diffusion_results_file)
    print(f"Loaded diffusion map results: {len(diffusion_df)} samples")
    
    # Check if GO term column exists in either dataframe
    go_in_data = go_term_column in data_frame.columns
    go_in_diffusion = go_term_column in diffusion_df.columns
    
    if not go_in_data and not go_in_diffusion:
        available_cols = [col for col in data_frame.columns if 'go' in col.lower() or 'target' in col.lower()]
        available_cols_diff = [col for col in diffusion_df.columns if 'go' in col.lower() or 'target' in col.lower()]
        error_msg = f"GO term column '{go_term_column}' not found in either dataframe.\n"
        if available_cols:
            error_msg += f"  Available GO columns in data_frame: {available_cols}\n"
        if available_cols_diff:
            error_msg += f"  Available GO columns in diffusion_df: {available_cols_diff}\n"
        raise ValueError(error_msg)
    
    # Merge with original data to get GO terms
    if go_in_diffusion:
        # GO terms already in diffusion results
        merged_df = diffusion_df.copy()
        print(f"Using GO terms from diffusion results file")
    elif 'query' in data_frame.columns and 'query' in diffusion_df.columns:
        # Merge on query column
        if go_in_data:
            merged_df = diffusion_df.merge(data_frame[['query', go_term_column]], on='query', how='inner')
            print(f"Merged on 'query' column")
        else:
            raise ValueError(f"GO term column '{go_term_column}' not found in data_frame")
    else:
        # Use index-based alignment (risky, but try it)
        if go_in_data:
            merged_df = diffusion_df.copy()
            if len(data_frame) >= len(diffusion_df):
                merged_df[go_term_column] = data_frame[go_term_column].values[:len(merged_df)]
                print(f"Aligned by index (assuming same order)")
            else:
                raise ValueError(f"Cannot align dataframes: data_frame has {len(data_frame)} rows, diffusion_df has {len(diffusion_df)} rows")
        else:
            raise ValueError(f"GO term column '{go_term_column}' not found in data_frame and cannot merge by query")
    
    print(f"After merging: {len(merged_df)} samples with both diffusion coords and GO terms")
    
    # Limit samples if requested
    if max_samples and len(merged_df) > max_samples:
        print(f"Limiting to {max_samples} samples (random sampling)")
        merged_df = merged_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    # Parse GO terms
    print(f"\nParsing GO terms from column: {go_term_column}")
    go_terms_list = merged_df[go_term_column].apply(parse_go_terms).tolist()
    
    # Count samples with GO terms
    samples_with_go = sum(1 for terms in go_terms_list if len(terms) > 0)
    print(f"Samples with GO terms: {samples_with_go}/{len(go_terms_list)} ({100*samples_with_go/len(go_terms_list):.1f}%)")
    
    if samples_with_go < 10:
        print("Warning: Too few samples with GO terms for meaningful analysis")
        return None
    
    # Get diffusion coordinates
    dc_cols = [col for col in merged_df.columns if col.startswith('Diffusion')]
    if not dc_cols:
        raise ValueError("No diffusion coordinate columns found in results file")
    
    dc_cols = sorted(dc_cols, key=lambda x: int(x.replace('Diffusion', '')))
    print(f"Using diffusion coordinates: {dc_cols}")
    
    diffusion_coords = merged_df[dc_cols].values
    
    # Get eigenvalues if available
    eigenvalues = None
    if 'eigenvalue' in merged_df.columns.str.lower().any():
        # Try to extract eigenvalues from metadata or compute from variance
        pass
    
    # Compute GO similarity matrix
    print(f"\nComputing GO semantic similarity matrix (method: {method})...")
    go_similarity_matrix = compute_go_semantic_similarity_matrix(go_terms_list, method=method)
    
    # Compute diffusion distance matrix
    print(f"\nComputing diffusion distance matrix (Euclidean: {use_euclidean})...")
    diffusion_distance_matrix = compute_diffusion_distance_matrix(
        diffusion_coords, 
        use_euclidean=use_euclidean,
        eigenvalues=eigenvalues
    )
    
    # Extract upper triangle (avoid duplicates and self-comparisons)
    n = len(merged_df)
    upper_tri_indices = np.triu_indices(n, k=1)
    
    go_similarities = go_similarity_matrix[upper_tri_indices]
    diffusion_distances = diffusion_distance_matrix[upper_tri_indices]
    
    print(f"\nAnalyzing {len(go_similarities)} pairwise comparisons...")
    
    # Compute correlations
    spearman_corr, spearman_p = spearmanr(go_similarities, diffusion_distances)
    pearson_corr, pearson_p = pearsonr(go_similarities, diffusion_distances)
    
    print(f"\nCorrelation Results:")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    
    # Interpretation
    if spearman_p < 0.05:
        if spearman_corr < -0.1:
            interpretation = "✓ Significant negative correlation: GO structure aligns with diffusion manifold"
        elif spearman_corr > 0.1:
            interpretation = "⚠ Significant positive correlation: Unexpected - similar GO terms are distant"
        else:
            interpretation = "No significant correlation: GO structure does not align with diffusion manifold"
    else:
        interpretation = "No significant correlation (p >= 0.05)"
    
    print(f"  Interpretation: {interpretation}")
    
    # Create visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(diffusion_distances, go_similarities, alpha=0.3, s=10)
        ax.set_xlabel('Diffusion Distance')
        ax.set_ylabel('GO Semantic Similarity')
        ax.set_title(f'GO Similarity vs Diffusion Distance\nSpearman r={spearman_corr:.3f}, p={spearman_p:.3e}')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(diffusion_distances, go_similarities, 1)
        p = np.poly1d(z)
        ax.plot(diffusion_distances, p(diffusion_distances), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.legend()
        
        # Histogram of similarities
        ax = axes[1]
        ax.hist(go_similarities, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('GO Semantic Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of GO Similarities')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'go_similarity_vs_diffusion_distance.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to: {plot_file}")
        plt.close()
        
        # Save results
        results = {
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_p,
            'n_samples': n,
            'n_comparisons': len(go_similarities),
            'method': method,
            'use_euclidean': use_euclidean,
            'interpretation': interpretation
        }
        
        results_df = pd.DataFrame([results])
        results_file = os.path.join(output_dir, 'go_similarity_correlation_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Saved results to: {results_file}")
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'interpretation': interpretation,
        'n_samples': n,
        'n_comparisons': len(go_similarities)
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze GO semantic similarity vs diffusion distance'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input CSV file with alignment data and GO terms'
    )
    parser.add_argument(
        '-d', '--diffusion-results',
        type=str,
        required=True,
        help='Path to CSV file with diffusion map results'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--go-column',
        type=str,
        default='target_biological_processes',
        choices=['target_biological_processes', 'target_molecular_functions', 'target_cellular_components'],
        help='GO term column to analyze'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='jaccard',
        choices=['jaccard', 'resnik', 'lin', 'wang'],
        help='GO similarity method'
    )
    parser.add_argument(
        '--use-euclidean',
        action='store_true',
        default=True,
        help='Use Euclidean distance in diffusion space (default: True)'
    )
    parser.add_argument(
        '--use-diffusion-distance',
        action='store_true',
        help='Use diffusion distance (weighted by eigenvalues) instead of Euclidean'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to analyze (for speed)'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_frame = pd.read_csv(args.input)
    print(f"Loaded {len(data_frame)} samples from {args.input}")
    
    # Determine distance metric
    use_euclidean = not args.use_diffusion_distance
    
    # Run analysis
    results = analyze_go_vs_diffusion_distance(
        data_frame=data_frame,
        diffusion_results_file=args.diffusion_results,
        go_term_column=args.go_column,
        output_dir=args.output_dir,
        method=args.method,
        use_euclidean=use_euclidean,
        max_samples=args.max_samples
    )
    
    if results:
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)


if __name__ == '__main__':
    main()


