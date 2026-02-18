#!/usr/bin/env python3
"""
Script for performing dimensionality reduction and clustering on alignment data.
Manages calling PCA tests with options to toggle semantic clustering.
"""

import argparse
import ast
import os
import pandas as pd
import sys
from stats_modules import dimensionality_reduction_tools
from utilities.go_semantic_similarity import analyze_go_vs_diffusion_distance


def parse_go_term_columns(df, go_term_columns):
    """
    Parse GO term columns from string representations to actual lists.
    Handles empty lists and missing values.
    """
    for col in go_term_columns:
        if col in df.columns:
            def safe_parse(val):
                if pd.isna(val) or val == '' or val == '[]':
                    return []
                if isinstance(val, list):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, list):
                            return parsed
                        return []
                    except (ValueError, SyntaxError):
                        return []
                return []
            
            df[col] = df[col].apply(safe_parse)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Perform dimensionality reduction and clustering on alignment data'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input CSV file with alignment data'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Path to directory for storing PCA/clustering results'
    )
    parser.add_argument(
        '-n', '--organism_name',
        type=str,
        required=True,
        help='Organism name (used for output file naming)'
    )
    parser.add_argument(
        '--use_semantic_clustering',
        action='store_true',
        help='Use semantic clustering based on GO terms (default: use simple label classification)'
    )
    parser.add_argument(
        '--go_term_type',
        type=str,
        default='target_cellular_components',
        choices=['target_cellular_components', 'target_molecular_functions', 'target_biological_processes'],
        help='GO term type to use for semantic clustering (default: target_cellular_components)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=15,
        help='Number of clusters for clustering (default: 15). Ignored if --auto_clusters is used.'
    )
    parser.add_argument(
        '--auto_clusters',
        action='store_true',
        help='Automatically estimate optimal number of clusters using silhouette score or elbow method. Overrides --n_clusters.'
    )
    parser.add_argument(
        '--cluster_estimation_method',
        type=str,
        default='silhouette',
        choices=['silhouette', 'elbow'],
        help='Method for automatic cluster estimation: silhouette (maximize silhouette score) or elbow (minimize WCSS decrease). Default: silhouette.'
    )
    parser.add_argument(
        '--use_tsne',
        action='store_true',
        help='Use t-SNE instead of LDA-PCA'
    )
    parser.add_argument(
        '--use_pca_tsne',
        action='store_true',
        help='Use PCA followed by t-SNE instead of LDA-PCA'
    )
    parser.add_argument(
        '--value_driven',
        action='store_true',
        help='Use purely value-driven clustering (no GO term labels, only alignment features)'
    )
    parser.add_argument(
        '--clustering_method',
        type=str,
        default='kmeans',
        choices=['kmeans', 'agglomerative'],
        help='Clustering method for value-driven clustering (default: kmeans)'
    )
    parser.add_argument(
        '--include_lengths',
        action='store_true',
        help='Include qlen and tlen (absolute lengths) in clustering. Default: exclude them to use only ratio features (0-1 scale)'
    )
    parser.add_argument(
        '--plot_3d',
        action='store_true',
        help='Create 3D visualization (PC1, PC2, PC3) in addition to 2D plot. Can reveal structure hidden in 2D projections.'
    )
    parser.add_argument(
        '--use_umap',
        action='store_true',
        help='Use UMAP instead of PCA for dimensionality reduction. Often provides better cluster separation.'
    )
    parser.add_argument(
        '--umap_n_neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter (default: 15). Lower values focus on local structure, higher on global.'
    )
    parser.add_argument(
        '--umap_min_dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter (default: 0.1). Controls how tightly points are packed (0.0-1.0).'
    )
    parser.add_argument(
        '--include_evolutionary',
        action='store_true',
        help='Include evolutionary features (dnds rates, test_fraction) in clustering. Missing values will be handled automatically.'
    )
    parser.add_argument(
        '--filter_missing_evorates',
        action='store_true',
        help='Filter out rows with missing evolutionary rate data (only use complete cases). Use with --include_evolutionary.'
    )
    parser.add_argument(
        '--impute_evorates',
        action='store_true',
        help='Impute missing evolutionary rate values with median. Use with --include_evolutionary.'
    )
    parser.add_argument(
        '--include_targeting',
        action='store_true',
        help='Include targeting features (mTP probability, SP probability) in clustering'
    )
    parser.add_argument(
        '--plddt_features',
        action='store_true',
        help='Include pLDDT features (plddt_query_region, plddt_target_region) in clustering. These are AlphaFold confidence scores for query and target regions.'
    )
    parser.add_argument(
        '--adjust_tm_score',
        action='store_true',
        help='Replace TM-score with pLDDT-weighted TM-score (score * min_plddt/100). Downweights alignments with low structure confidence. Requires pLDDT columns.'
    )
    parser.add_argument(
        '--include_go_terms',
        action='store_true',
        help='Include GO terms as features (not yet implemented - use semantic clustering instead)'
    )
    parser.add_argument(
        '--exclude_derived_features',
        action='store_true',
        help='Exclude derived features (score_per_length, coverage_balance, identity_coverage) from analysis. Default: include derived features.'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='robust',
        choices=['robust', 'zscore', 'standard', 'minmax'],
        help='Scaling method: robust (default, robust to outliers), zscore/standard (mean=0, std=1), or minmax (0-1 range)'
    )
    parser.add_argument(
        '--use_diffusion',
        action='store_true',
        help='Use Diffusion Map for dimensionality reduction. Reveals underlying manifold structure and can identify trajectories.'
    )
    parser.add_argument(
        '--diffusion_alpha',
        type=float,
        default=1.0,
        help='Diffusion map alpha parameter (default: 1.0). Controls density normalization (0=no normalization, 1=full normalization).'
    )
    parser.add_argument(
        '--diffusion_n_neighbors',
        type=int,
        default=15,
        help='Diffusion map n_neighbors parameter (default: 15). Number of neighbors for affinity graph construction. Use --auto_n_neighbors to automatically determine optimal value.'
    )
    parser.add_argument(
        '--auto_n_neighbors',
        action='store_true',
        help='Automatically determine optimal n_neighbors for diffusion map using heuristics and graph connectivity analysis. Overrides --diffusion_n_neighbors if set.'
    )
    parser.add_argument(
        '--natural_attractor',
        action='store_true',
        help='Use natural attractor (highest local density among all samples) instead of mimic-based attractor. Default: use highest density among known mimics.'
    )
    parser.add_argument(
        '--go_similarity_analysis',
        action='store_true',
        help='Perform GO semantic similarity vs diffusion distance analysis after diffusion map computation. Requires --use_diffusion.'
    )
    parser.add_argument(
        '--go_similarity_column',
        type=str,
        default='target_biological_processes',
        choices=['target_biological_processes', 'target_molecular_functions', 'target_cellular_components'],
        help='GO term column for similarity analysis (default: target_biological_processes)'
    )
    parser.add_argument(
        '--go_similarity_max_samples',
        type=int,
        default=None,
        help='Maximum number of samples for GO similarity analysis (for speed). Default: all samples.'
    )
    parser.add_argument(
        '--system_overlap_analysis',
        action='store_true',
        help='Analyze and visualize manifold overlap between systems. Requires --use_diffusion and multi-system data (system column).'
    )
    parser.add_argument(
        '--save_cloud_definition',
        action='store_true',
        help='Save the mimic cloud definition in feature space to a JSON file. This can be transferred to other datasets.'
    )
    parser.add_argument(
        '--transfer_cloud_from',
        type=str,
        default=None,
        help='Path to JSON file containing mimic cloud definition from another dataset (e.g., legionella). Applies this cloud to the current dataset in feature space.'
    )
    parser.add_argument(
        '--adaptive_epsilon',
        action='store_true',
        help='Use adaptive (local) epsilon/kernel width for diffusion map. Each point uses its own bandwidth based on k-nearest neighbors, improving handling of varying densities.'
    )
    parser.add_argument(
        '--adaptive_k_neighbors',
        type=int,
        default=7,
        help='Number of neighbors for adaptive epsilon estimation (default: 7). Only used with --adaptive_epsilon.'
    )
    parser.add_argument(
        '--epsilon_scale',
        type=float,
        default=1.0,
        help='Scaling factor for adaptive epsilon (default: 1.0). Values < 1.0 reduce epsilon for tighter clustering, > 1.0 increase for better connectivity. Only used with --adaptive_epsilon.'
    )
    parser.add_argument(
        '--spectral_cloud',
        action='store_true',
        help='Define mimic cloud using spectral clustering in addition to distance-based method. Runs alongside distance-based cloud for comparison.'
    )
    parser.add_argument(
        '--spectral_n_clusters',
        type=int,
        default=None,
        help='Number of clusters for spectral clustering. If not specified, will be automatically determined using eigengap heuristic (recommended).'
    )
    parser.add_argument(
        '--spectral_gamma',
        type=float,
        default=None,
        help='RBF kernel parameter (gamma) for spectral clustering. If not specified, will be estimated from median pairwise distance (default: 1/(2*median_dist^2)).'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read input CSV
    print(f"Reading alignment data from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Filter out alignments with likely paralogs
    if 'targets_likely_paralogs' in df.columns:
        initial_count = len(df)
        # Convert to string and handle NaN values, then filter out rows where targets_likely_paralogs is "yes" (case-insensitive)
        paralog_mask = df['targets_likely_paralogs'].astype(str).str.lower().isin(['yes', 'y', 'true', '1'])
        df = df[~paralog_mask]
        filtered_count = len(df)
        removed = initial_count - filtered_count
        if removed > 0:
            print(f"Filtered out {removed} alignments with likely paralogs ({removed/initial_count*100:.1f}%)")
        else:
            print("No alignments with likely paralogs found to filter")
    else:
        print("Warning: 'targets_likely_paralogs' column not found. Skipping paralog filtering.")
    
    # Handle missing evolutionary rate data if requested
    if args.include_evolutionary:
        evo_cols = ['symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg', 'test_fraction']
        available_evo_cols = [col for col in evo_cols if col in df.columns]
        
        if args.filter_missing_evorates:
            initial_count = len(df)
            # Filter to only rows with all evolutionary data present
            df = df.dropna(subset=available_evo_cols)
            filtered_count = len(df)
            removed = initial_count - filtered_count
            if removed > 0:
                print(f"Filtered out {removed} alignments with missing evolutionary data ({removed/initial_count*100:.1f}%)")
                print(f"  Remaining: {filtered_count} alignments")
            else:
                print("All alignments have complete evolutionary data")
        elif args.impute_evorates:
            # Impute missing values with median
            for col in available_evo_cols:
                median_val = df[col].median()
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col].fillna(median_val, inplace=True)
                    print(f"Imputed {missing_count} missing values in {col} with median ({median_val:.6f})")
        else:
            # Default: keep missing values, let the scaler handle them
            # But warn if too many are missing
            for col in available_evo_cols:
                missing = df[col].isna().sum()
                if missing > len(df) * 0.5:
                    print(f"Warning: {missing}/{len(df)} ({missing/len(df)*100:.1f}%) missing in {col}")
                    print(f"  Consider using --filter_missing_evorates or --impute_evorates")
    
    # Check required columns exist
    required_columns = ['query', 'score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}", file=sys.stderr)
        sys.exit(1)
    
    # Run appropriate analysis
    if args.value_driven:
        if args.include_lengths:
            print("Running value-driven PCA clustering (including qlen/tlen)...")
        else:
            print("Running value-driven PCA clustering (ratio features only, excluding qlen/tlen)...")
        # No GO term parsing needed - purely based on alignment feature values
        # This is for discovering new candidates, not validating known ones
        # Check for conflicting dimensionality reduction options
        reduction_methods = sum([args.use_umap, args.use_diffusion])
        if reduction_methods > 1:
            print("Error: Cannot use multiple dimensionality reduction methods. Choose one of --use_umap or --use_diffusion.", file=sys.stderr)
            sys.exit(1)
        
        if args.use_diffusion:
            # Diffusion maps are for trajectory analysis, not clustering
            # Skip clustering parameters
            print("Note: Diffusion maps perform trajectory/pseudotime analysis, not clustering.")
            print("Ignoring --n_clusters and --clustering_method for diffusion map analysis.")
        
        dimensionality_reduction_tools.value_driven_pca_clustering(
            args.output_dir,
            df,
            args.organism_name,
            args.n_clusters,
            args.clustering_method,
            include_lengths=args.include_lengths,
            plot_3d=args.plot_3d,
            use_umap=args.use_umap,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            include_evolutionary=args.include_evolutionary,
            include_targeting=args.include_targeting,
            include_plddt=args.plddt_features,
            adjust_tm_score=args.adjust_tm_score,
            include_go_terms=args.include_go_terms,
            scaler_type=args.scaler,
            use_diffusion=args.use_diffusion,
            diffusion_alpha=args.diffusion_alpha,
            diffusion_n_neighbors=args.diffusion_n_neighbors,
            exclude_derived_features=args.exclude_derived_features,
            auto_clusters=args.auto_clusters,
            cluster_estimation_method=args.cluster_estimation_method,
            use_natural_attractor=args.natural_attractor,
            save_cloud_definition=args.save_cloud_definition,
            transfer_cloud_from=args.transfer_cloud_from,
            use_adaptive_epsilon=args.adaptive_epsilon,
            adaptive_k_neighbors=args.adaptive_k_neighbors,
            auto_n_neighbors=args.auto_n_neighbors,
            use_spectral_cloud=args.spectral_cloud,
            spectral_n_clusters=args.spectral_n_clusters,
            spectral_gamma=args.spectral_gamma,
            epsilon_scale=args.epsilon_scale
        )
        
        # Run GO similarity analysis if requested and diffusion map was used
        if args.go_similarity_analysis:
            if not args.use_diffusion:
                print("Warning: --go_similarity_analysis requires --use_diffusion. Skipping GO similarity analysis.")
            else:
                # Parse GO terms for the analysis
                print(f"\nParsing GO term column: {args.go_similarity_column}...")
                df = parse_go_term_columns(df, [args.go_similarity_column])
                
                # Construct path to diffusion results file
                diffusion_results_file = os.path.join(
                    args.output_dir,
                    f'{args.organism_name}_diffusion_map_results.csv'
                )
                
                if not os.path.exists(diffusion_results_file):
                    print(f"Warning: Diffusion results file not found: {diffusion_results_file}")
                    print("  GO similarity analysis requires diffusion map results. Skipping.")
                else:
                    print(f"\nRunning GO semantic similarity vs diffusion distance analysis...")
                    try:
                        results = analyze_go_vs_diffusion_distance(
                            data_frame=df,
                            diffusion_results_file=diffusion_results_file,
                            go_term_column=args.go_similarity_column,
                            output_dir=args.output_dir,
                            method='jaccard',  # Can be made configurable later
                            use_euclidean=True,  # Can be made configurable later
                            max_samples=args.go_similarity_max_samples
                        )
                        
                        if results:
                            print(f"\n✓ GO similarity analysis complete!")
                            print(f"  Spearman correlation: {results['spearman_correlation']:.4f} (p={results['spearman_pvalue']:.4e})")
                            print(f"  Interpretation: {results['interpretation']}")
                    except Exception as e:
                        print(f"Error in GO similarity analysis: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Run system manifold overlap analysis if requested
        if args.system_overlap_analysis:
            if not args.use_diffusion:
                print("Warning: --system_overlap_analysis requires --use_diffusion. Skipping overlap analysis.")
            elif 'system' not in df.columns:
                print("Warning: 'system' column not found. System overlap analysis requires multi-system data.")
            else:
                # Construct path to diffusion results file
                diffusion_results_file = os.path.join(
                    args.output_dir,
                    f'{args.organism_name}_diffusion_map_results.csv'
                )
                
                if not os.path.exists(diffusion_results_file):
                    print(f"Warning: Diffusion results file not found: {diffusion_results_file}")
                    print("  System overlap analysis requires diffusion map results. Skipping.")
                else:
                    print(f"\nRunning system manifold overlap analysis...")
                    try:
                        dimensionality_reduction_tools.analyze_system_manifold_overlap(
                            diffusion_results_file=diffusion_results_file,
                            output_dir=args.output_dir,
                            n_neighbors=15,
                            use_dc123=True
                        )
                    except Exception as e:
                        print(f"Error in system overlap analysis: {e}")
                        import traceback
                        traceback.print_exc()
    elif args.use_tsne:
        # Parse GO terms for t-SNE (it uses apply_labels which needs target_cellular_components)
        print("Parsing GO term column: target_cellular_components...")
        df = parse_go_term_columns(df, ['target_cellular_components'])
        print("Running t-SNE analysis...")
        dimensionality_reduction_tools.tsne_analysis(args.output_dir, df)
    elif args.use_pca_tsne:
        # Parse GO terms for PCA-t-SNE
        print("Parsing GO term column: target_cellular_components...")
        df = parse_go_term_columns(df, ['target_cellular_components'])
        print("Running PCA-t-SNE analysis...")
        dimensionality_reduction_tools.pca_tsne_analysis(args.output_dir, df)
    else:
        # LDA-PCA methods (supervised, require labels)
        if args.use_semantic_clustering:
            print(f"Running LDA-PCA analysis with semantic clustering ({args.go_term_type}, {args.n_clusters} clusters)...")
            print(f"Parsing GO term column: {args.go_term_type}...")
            df = parse_go_term_columns(df, [args.go_term_type])
        else:
            print("Running LDA-PCA analysis with simple label classification (semantic clustering disabled)...")
            print("Parsing GO term column: target_cellular_components (for simple classification)...")
            df = parse_go_term_columns(df, ['target_cellular_components'])
            args.go_term_type = 'target_cellular_components'  # Still needed for function signature
        
        dimensionality_reduction_tools.lda_pca_analysis(
            args.output_dir,
            df,
            args.organism_name,
            args.go_term_type,
            args.n_clusters,
            use_semantic_clustering=args.use_semantic_clustering
        )
    
    print(f"Analysis complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

