#!/usr/bin/env python3
"""
Script to compare cloud feature statistics across multiple systems.
Takes any number of cloud statistics CSV files and creates comparison plots.
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_statistics_file(csv_file):
    """
    Parse a cloud statistics CSV file and extract feature statistics.
    
    Returns:
    --------
    dict with keys:
        - 'system_name': str
        - 'cloud_type': str ('local' or 'transferred')
        - 'sample_counts': dict
        - 'features': dict of feature_name -> {region -> {stat -> value}}
    """
    # Extract system name and cloud type from filename
    filename = os.path.basename(csv_file)
    # Pattern: {organism_name}_{cloud_type}_cloud_feature_statistics.csv
    match = re.match(r'(.+?)_(local|transferred)_cloud_feature_statistics\.csv', filename)
    if match:
        system_name = match.group(1)
        cloud_type = match.group(2)
    else:
        # Fallback: use filename without extension
        system_name = os.path.splitext(filename)[0]
        cloud_type = 'unknown'
    
    # Read CSV
    stats_df = pd.read_csv(csv_file)
    
    # Read summary if available
    summary_file = csv_file.replace('.csv', '_summary.csv')
    sample_counts = {}
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        for _, row in summary_df.iterrows():
            region = row['region']
            if region != 'total':
                sample_counts[region] = {
                    'count': int(row['count']),
                    'percent': float(row['percent'])
                }
    else:
        # Calculate from stats_df if summary not available
        for region in ['within_cloud', 'around_cloud', 'outside_cloud']:
            region_data = stats_df[stats_df['region'] == region]
            if len(region_data) > 0:
                count = region_data['count'].iloc[0] if 'count' in region_data.columns else 0
                sample_counts[region] = {'count': int(count), 'percent': 0.0}
    
    # Organize features
    features = {}
    for _, row in stats_df.iterrows():
        feature_name = row['feature']
        region = row['region']
        
        if feature_name not in features:
            features[feature_name] = {}
        
        features[feature_name][region] = {
            'mean': float(row['mean']),
            'min': float(row['min']),
            'max': float(row['max']),
            'std': float(row['std']),
            'count': int(row['count']) if 'count' in row else 0
        }
    
    return {
        'system_name': system_name,
        'cloud_type': cloud_type,
        'sample_counts': sample_counts,
        'features': features
    }


def create_comparison_plots(parsed_data_list, output_dir):
    """
    Create comparison plots for feature statistics across systems.
    Focuses on core 5 features: score, tcov, qcov, fident, algn_fraction
    
    Parameters:
    -----------
    parsed_data_list : list of dict
        List of parsed statistics dictionaries
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Core 5 features to focus on
    core_features = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction']
    
    # Get all unique features across all systems, but filter to core features
    all_features = set()
    for data in parsed_data_list:
        all_features.update(data['features'].keys())
    # Filter to only core features that exist
    all_features = sorted([f for f in core_features if f in all_features])
    
    # Get all unique regions
    all_regions = set()
    for data in parsed_data_list:
        for feature_data in data['features'].values():
            all_regions.update(feature_data.keys())
    all_regions = sorted(all_regions)
    
    # Create system labels
    system_labels = []
    for data in parsed_data_list:
        label = f"{data['system_name']} ({data['cloud_type']})"
        system_labels.append(label)
    
    # 1. Sample counts comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Within cloud counts
    within_counts = [data['sample_counts'].get('within_cloud', {}).get('count', 0) for data in parsed_data_list]
    within_percents = [data['sample_counts'].get('within_cloud', {}).get('percent', 0) for data in parsed_data_list]
    
    x_pos = np.arange(len(system_labels))
    axes[0].bar(x_pos, within_counts, alpha=0.7)
    axes[0].set_xlabel('System')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Samples Within Cloud')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(system_labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (count, pct) in enumerate(zip(within_counts, within_percents)):
        axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    # Around cloud counts
    around_counts = [data['sample_counts'].get('around_cloud', {}).get('count', 0) for data in parsed_data_list]
    around_percents = [data['sample_counts'].get('around_cloud', {}).get('percent', 0) for data in parsed_data_list]
    
    axes[1].bar(x_pos, around_counts, alpha=0.7, color='orange')
    axes[1].set_xlabel('System')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Samples Around Cloud (not within)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(system_labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (count, pct) in enumerate(zip(around_counts, around_percents)):
        axes[1].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_counts_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Core features: Grouped bar chart comparing within vs around cloud across systems
    if 'within_cloud' in all_regions and 'around_cloud' in all_regions:
        fig, axes = plt.subplots(1, len(all_features), figsize=(5*len(all_features), 6))
        if len(all_features) == 1:
            axes = [axes]
        
        for feat_idx, feature in enumerate(all_features):
            ax = axes[feat_idx]
            
            # Collect data for within and around cloud
            within_means = []
            within_stds = []
            around_means = []
            around_stds = []
            labels = []
            
            for data in parsed_data_list:
                if feature in data['features']:
                    if 'within_cloud' in data['features'][feature]:
                        within_means.append(data['features'][feature]['within_cloud']['mean'])
                        within_stds.append(data['features'][feature]['within_cloud']['std'])
                    else:
                        within_means.append(np.nan)
                        within_stds.append(0)
                    
                    if 'around_cloud' in data['features'][feature]:
                        around_means.append(data['features'][feature]['around_cloud']['mean'])
                        around_stds.append(data['features'][feature]['around_cloud']['std'])
                    else:
                        around_means.append(np.nan)
                        around_stds.append(0)
                    
                    labels.append(data['system_name'])
            
            if within_means or around_means:
                x = np.arange(len(labels))
                width = 0.35
                
                # Create bars
                bars1 = ax.bar(x - width/2, within_means, width, yerr=within_stds, 
                              label='Within Cloud', alpha=0.8, capsize=5, color='#2E86AB')
                bars2 = ax.bar(x + width/2, around_means, width, yerr=around_stds,
                              label='Around Cloud', alpha=0.8, capsize=5, color='#F18F01')
                
                ax.set_xlabel('System', fontsize=11)
                ax.set_ylabel('Mean ± Std', fontsize=11)
                ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'core_features_within_vs_around_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Core features: Heatmap comparing within vs around across systems
    if 'within_cloud' in all_regions and 'around_cloud' in all_regions:
        # Create data for heatmap: features x (systems * regions)
        heatmap_data = []
        heatmap_labels = []
        
        for feature in all_features:
            row = []
            for data in parsed_data_list:
                if feature in data['features']:
                    if 'within_cloud' in data['features'][feature]:
                        row.append(data['features'][feature]['within_cloud']['mean'])
                    else:
                        row.append(np.nan)
                    if 'around_cloud' in data['features'][feature]:
                        row.append(data['features'][feature]['around_cloud']['mean'])
                    else:
                        row.append(np.nan)
                else:
                    row.extend([np.nan, np.nan])
            heatmap_data.append(row)
        
        # Create column labels
        for data in parsed_data_list:
            heatmap_labels.append(f"{data['system_name']}\n(within)")
            heatmap_labels.append(f"{data['system_name']}\n(around)")
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, index=all_features, columns=heatmap_labels)
            
            fig, axes = plt.subplots(1, 2, figsize=(max(12, len(heatmap_labels)*0.8), max(6, len(all_features)*0.8)))
            
            # Raw values
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='viridis', ax=axes[0],
                       cbar_kws={'label': 'Mean Value'}, linewidths=0.5, linecolor='white')
            axes[0].set_title('Core Features: Within vs Around Cloud\n(Raw Values)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('System & Region', fontsize=11)
            axes[0].set_ylabel('Feature', fontsize=11)
            
            # Normalized by feature (z-score)
            heatmap_df_norm = heatmap_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            sns.heatmap(heatmap_df_norm, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1],
                       cbar_kws={'label': 'Z-score'}, linewidths=0.5, linecolor='white')
            axes[1].set_title('Core Features: Within vs Around Cloud\n(Z-score Normalized)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('System & Region', fontsize=11)
            axes[1].set_ylabel('Feature', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'core_features_heatmap_within_vs_around.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Core features: Range comparison (min-max) for within and around
    if 'within_cloud' in all_regions and 'around_cloud' in all_regions:
        fig, axes = plt.subplots(1, len(all_features), figsize=(5*len(all_features), 6))
        if len(all_features) == 1:
            axes = [axes]
        
        for feat_idx, feature in enumerate(all_features):
            ax = axes[feat_idx]
            
            within_ranges = []
            around_ranges = []
            labels = []
            
            for data in parsed_data_list:
                if feature in data['features']:
                    if 'within_cloud' in data['features'][feature]:
                        feat_data = data['features'][feature]['within_cloud']
                        within_ranges.append((feat_data['min'], feat_data['max'], feat_data['mean']))
                    else:
                        within_ranges.append((np.nan, np.nan, np.nan))
                    
                    if 'around_cloud' in data['features'][feature]:
                        feat_data = data['features'][feature]['around_cloud']
                        around_ranges.append((feat_data['min'], feat_data['max'], feat_data['mean']))
                    else:
                        around_ranges.append((np.nan, np.nan, np.nan))
                    
                    labels.append(data['system_name'])
            
            if within_ranges or around_ranges:
                x = np.arange(len(labels))
                width = 0.35
                
                # Plot range bars for within cloud
                for i, (min_val, max_val, mean_val) in enumerate(within_ranges):
                    if not np.isnan(min_val):
                        ax.plot([x[i] - width/2, x[i] - width/2], [min_val, max_val], 
                               'b-', linewidth=3, alpha=0.7, label='Within' if i == 0 else '')
                        ax.plot(x[i] - width/2, mean_val, 'bo', markersize=8, alpha=0.8)
                
                # Plot range bars for around cloud
                for i, (min_val, max_val, mean_val) in enumerate(around_ranges):
                    if not np.isnan(min_val):
                        ax.plot([x[i] + width/2, x[i] + width/2], [min_val, max_val],
                               'r-', linewidth=3, alpha=0.7, label='Around' if i == 0 else '')
                        ax.plot(x[i] + width/2, mean_val, 'ro', markersize=8, alpha=0.8)
                
                ax.set_xlabel('System', fontsize=11)
                ax.set_ylabel('Value', fontsize=11)
                ax.set_title(f'{feature}\n(Min-Max Range)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'core_features_ranges_within_vs_around.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Feature mean comparison (for within cloud region)
    if 'within_cloud' in all_regions:
        # Select top features by variance across systems
        feature_means = {}
        for feature in all_features:
            means = []
            for data in parsed_data_list:
                if feature in data['features'] and 'within_cloud' in data['features'][feature]:
                    means.append(data['features'][feature]['within_cloud']['mean'])
                else:
                    means.append(np.nan)
            if not all(np.isnan(means)):
                feature_means[feature] = np.nanstd(means)  # Variance across systems
        
        # Sort by variance and take top features
        top_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)[:15]
        top_feature_names = [f[0] for f in top_features]
        
        if top_feature_names:
            n_features = len(top_feature_names)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(top_feature_names):
                ax = axes[idx]
                means = []
                stds = []
                labels = []
                
                for data in parsed_data_list:
                    if feature in data['features'] and 'within_cloud' in data['features'][feature]:
                        means.append(data['features'][feature]['within_cloud']['mean'])
                        stds.append(data['features'][feature]['within_cloud']['std'])
                        labels.append(data['system_name'])
                
                if means:
                    x_pos = np.arange(len(labels))
                    ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
                    ax.set_xlabel('System')
                    ax.set_ylabel('Mean ± Std')
                    ax.set_title(f'{feature}\n(Within Cloud)')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(top_feature_names), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_means_within_cloud_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Summary table
    summary_data = []
    for data in parsed_data_list:
        row = {
            'System': data['system_name'],
            'Cloud Type': data['cloud_type'],
            'Within Cloud Count': data['sample_counts'].get('within_cloud', {}).get('count', 0),
            'Within Cloud %': data['sample_counts'].get('within_cloud', {}).get('percent', 0),
            'Around Cloud Count': data['sample_counts'].get('around_cloud', {}).get('count', 0),
            'Around Cloud %': data['sample_counts'].get('around_cloud', {}).get('percent', 0),
            'N Features': len(data['features'])
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary table to: {summary_file}")
    
    print(f"\nComparison plots saved to: {output_dir}")
    print(f"  - sample_counts_comparison.png")
    print(f"  - core_features_within_vs_around_comparison.png (grouped bars)")
    print(f"  - core_features_heatmap_within_vs_around.png (heatmap)")
    print(f"  - core_features_ranges_within_vs_around.png (min-max ranges)")
    print(f"  - comparison_summary.csv")


def plot_candidates_vs_cloud(parsed_data_list, diffusion_results_files, output_dir, candidate_list=None, validation_list=None):
    """
    Compare candidate and known mimic feature values to cloud ranges.
    
    Parameters:
    -----------
    parsed_data_list : list of dict
        List of parsed statistics dictionaries
    diffusion_results_files : list of str
        Paths to diffusion map results CSV files (should match systems in parsed_data_list)
    output_dir : str
        Directory to save plots
    candidate_list : list of str, optional
        List of candidate query IDs. If None, uses default list.
    validation_list : list of str, optional
        List of known mimic query IDs. If None, tries to detect from data or uses default.
    """
    if candidate_list is None:
        candidate_list = ['Q73HU7', 'Q73IF8', 'Q73GY6', 'Q73GG5', 'Q73HX8', 'P61189', 'O25525', 'O25981']
    
    if validation_list is None:
        # Default validation list (legionella mimics)
        validation_list = ['Q5ZU58', 'Q5ZXE0', 'Q5ZU32', 'Q5ZSB6', 'Q5ZRP9', 'Q5ZT65', 'Q5ZXN6', 'Q5ZVF7', 
                          'Q5ZTE0', 'Q5ZTL7', 'Q5ZSQ2', 'Q5ZTM4', 'Q5ZU83', 'Q5ZVS2', 'Q5ZWA1']
    
    core_features = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction']
    
    # Match diffusion results files to systems
    system_to_results = {}
    for results_file in diffusion_results_files:
        if not os.path.exists(results_file):
            print(f"Warning: Diffusion results file not found: {results_file}")
            continue
        
        # Try to match by system name
        filename = os.path.basename(results_file)
        matched = False
        for data in parsed_data_list:
            if data['system_name'] in filename:
                system_to_results[data['system_name']] = results_file
                print(f"  Matched {results_file} to system: {data['system_name']}")
                matched = True
                break
        
        # If only one results file and one system, match them
        if not matched and len(diffusion_results_files) == 1 and len(parsed_data_list) == 1:
            system_to_results[parsed_data_list[0]['system_name']] = results_file
            print(f"  Matched single results file to system: {parsed_data_list[0]['system_name']}")
        elif not matched:
            print(f"  Warning: Could not match {results_file} to any system")
    
    if len(system_to_results) == 0:
        print("Warning: No matching diffusion results files found. Skipping candidate comparison.")
        return
    
    # Create plots for each system
    for data in parsed_data_list:
        system_name = data['system_name']
        if system_name not in system_to_results:
            continue
        
        results_file = system_to_results[system_name]
        try:
            results_df = pd.read_csv(results_file)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            continue
        
        # Filter to candidates and known mimics
        if 'query' not in results_df.columns:
            print(f"Warning: 'query' column not found in {results_file}")
            continue
        
        candidate_df = results_df[results_df['query'].isin(candidate_list)].copy()
        mimic_df = results_df[results_df['query'].isin(validation_list)].copy()
        
        if len(candidate_df) == 0 and len(mimic_df) == 0:
            print(f"No candidates or known mimics found in {system_name} results")
            continue
        
        # Get cloud statistics for this system
        cloud_stats = data['features']
        
        # Create comparison plot
        fig, axes = plt.subplots(1, len(core_features), figsize=(5*len(core_features), 6))
        if len(core_features) == 1:
            axes = [axes]
        
        for feat_idx, feature in enumerate(core_features):
            ax = axes[feat_idx]
            
            if feature not in cloud_stats or 'within_cloud' not in cloud_stats[feature]:
                ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature)
                continue
            
            # Get cloud ranges
            cloud_data = cloud_stats[feature]['within_cloud']
            cloud_min = cloud_data['min']
            cloud_max = cloud_data['max']
            cloud_mean = cloud_data['mean']
            cloud_std = cloud_data['std']
            
            if feature not in results_df.columns:
                ax.text(0.5, 0.5, f'Feature not\nin data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature)
                continue
            
            # Plot cloud range as shaded region
            ax.axhspan(cloud_min, cloud_max, alpha=0.2, color='green', label='Cloud Range')
            ax.axhline(cloud_mean, color='green', linestyle='--', linewidth=2, label='Cloud Mean')
            ax.axhline(cloud_mean - cloud_std, color='green', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(cloud_mean + cloud_std, color='green', linestyle=':', linewidth=1, alpha=0.5)
            
            # Plot candidates
            candidate_vals = candidate_df[feature].dropna().values if len(candidate_df) > 0 and feature in candidate_df.columns else []
            if len(candidate_vals) > 0:
                x_pos_candidates = np.arange(len(candidate_vals))
                candidate_colors = ['red' if (val < cloud_min or val > cloud_max) else 'blue' for val in candidate_vals]
                ax.scatter(x_pos_candidates, candidate_vals, s=100, c=candidate_colors, alpha=0.7, 
                          edgecolors='black', linewidths=1.5, zorder=5, label='Candidates', marker='o')
                
                # Add candidate labels
                for i, (x, val, query) in enumerate(zip(x_pos_candidates, candidate_vals, candidate_df['query'].values)):
                    ax.text(x, val, query, fontsize=7, ha='center', va='bottom' if val > cloud_mean else 'top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='black'))
            
            # Plot known mimics
            mimic_vals = mimic_df[feature].dropna().values if len(mimic_df) > 0 and feature in mimic_df.columns else []
            if len(mimic_vals) > 0:
                x_pos_mimics = np.arange(len(mimic_vals)) + (len(candidate_vals) if len(candidate_vals) > 0 else 0) + 0.3
                mimic_colors = ['purple' if (val < cloud_min or val > cloud_max) else 'darkblue' for val in mimic_vals]
                ax.scatter(x_pos_mimics, mimic_vals, s=100, c=mimic_colors, alpha=0.7, 
                          edgecolors='black', linewidths=1.5, zorder=5, label='Known Mimics', marker='s')
                
                # Add mimic labels
                for i, (x, val, query) in enumerate(zip(x_pos_mimics, mimic_vals, mimic_df['query'].values)):
                    ax.text(x, val, query, fontsize=7, ha='center', va='bottom' if val > cloud_mean else 'top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='black'))
            
            # Count in/out of range
            candidate_in = sum((candidate_vals >= cloud_min) & (candidate_vals <= cloud_max)) if len(candidate_vals) > 0 else 0
            mimic_in = sum((mimic_vals >= cloud_min) & (mimic_vals <= cloud_max)) if len(mimic_vals) > 0 else 0
            
            title_parts = [f'{feature}', f'{system_name}']
            if len(candidate_vals) > 0:
                title_parts.append(f'Candidates: {candidate_in}/{len(candidate_vals)} in range')
            if len(mimic_vals) > 0:
                title_parts.append(f'Mimics: {mimic_in}/{len(mimic_vals)} in range')
            
            ax.set_xlabel('Index', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title('\n'.join(title_parts), fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'queries_vs_cloud_{system_name}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved query comparison for {system_name}: {plot_file}")
        
        # Create summary tables for both candidates and mimics
        for query_type, query_df, label in [('candidates', candidate_df, 'Candidates'), 
                                            ('mimics', mimic_df, 'Known Mimics')]:
            if len(query_df) == 0:
                continue
            
            summary_rows = []
            for feature in core_features:
                if feature not in query_df.columns:
                    continue
                if feature not in cloud_stats or 'within_cloud' not in cloud_stats[feature]:
                    continue
                
                cloud_data = cloud_stats[feature]['within_cloud']
                query_vals = query_df[feature].dropna()
                
                for query, val in zip(query_df['query'].values, query_df[feature].values):
                    if pd.isna(val):
                        continue
                    
                    in_range = (val >= cloud_data['min']) and (val <= cloud_data['max'])
                    distance_from_mean = abs(val - cloud_data['mean'])
                    z_score = (val - cloud_data['mean']) / cloud_data['std'] if cloud_data['std'] > 0 else 0
                    
                    summary_rows.append({
                        'query': query,
                        'query_type': query_type,
                        'feature': feature,
                        'value': val,
                        'cloud_mean': cloud_data['mean'],
                        'cloud_min': cloud_data['min'],
                        'cloud_max': cloud_data['max'],
                        'cloud_std': cloud_data['std'],
                        'in_range': in_range,
                        'distance_from_mean': distance_from_mean,
                        'z_score': z_score
                    })
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_file = os.path.join(output_dir, f'{query_type}_vs_cloud_{system_name}_summary.csv')
                summary_df.to_csv(summary_file, index=False)
                print(f"  Saved {label.lower()} summary for {system_name}: {summary_file}")


def analyze_cloud_differences(parsed_data_list, diffusion_results_files, output_dir, candidate_list=None, validation_list=None):
    """
    Analyze what's different about alignments in the cloud region between systems
    and why candidates are not being included.
    
    Parameters:
    -----------
    parsed_data_list : list of dict
        List of parsed statistics dictionaries
    diffusion_results_files : list of str
        Paths to diffusion map results CSV files
    output_dir : str
        Directory to save analysis
    candidate_list : list of str, optional
        List of candidate query IDs
    validation_list : list of str, optional
        List of known mimic query IDs
    """
    if candidate_list is None:
        candidate_list = ['Q73HU7', 'Q73IF8', 'Q73GY6', 'Q73GG5', 'Q73HX8', 'P61189', 'O25525', 'O25981']
    
    if validation_list is None:
        validation_list = ['Q5ZU58', 'Q5ZXE0', 'Q5ZU32', 'Q5ZSB6', 'Q5ZRP9', 'Q5ZT65', 'Q5ZXN6', 'Q5ZVF7', 
                          'Q5ZTE0', 'Q5ZTL7', 'Q5ZSQ2', 'Q5ZTM4', 'Q5ZU83', 'Q5ZVS2', 'Q5ZWA1']
    
    core_features = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction']
    
    # Match diffusion results files to systems
    system_to_results = {}
    for results_file in diffusion_results_files:
        if not os.path.exists(results_file):
            continue
        filename = os.path.basename(results_file)
        matched = False
        for data in parsed_data_list:
            if data['system_name'] in filename:
                system_to_results[data['system_name']] = results_file
                matched = True
                break
        if not matched and len(diffusion_results_files) == 1 and len(parsed_data_list) == 1:
            system_to_results[parsed_data_list[0]['system_name']] = results_file
    
    if len(system_to_results) == 0:
        print("Warning: No matching diffusion results files found.")
        return
    
    # Collect data from all systems
    all_cloud_data = []  # Samples within cloud
    all_candidate_data = []  # Candidate samples
    all_mimic_data = []  # Known mimics
    
    for data in parsed_data_list:
        system_name = data['system_name']
        if system_name not in system_to_results:
            continue
        
        results_file = system_to_results[system_name]
        try:
            results_df = pd.read_csv(results_file)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            continue
        
        # Determine cloud membership column
        if data['cloud_type'] == 'transferred':
            cloud_within_col = 'transferred_within_cloud'
        else:
            cloud_within_col = 'feature_space_within_cloud'
        
        if cloud_within_col not in results_df.columns:
            print(f"Warning: Cloud column {cloud_within_col} not found in {results_file}")
            continue
        
        # Get samples within cloud
        cloud_mask = results_df[cloud_within_col] == True
        cloud_samples = results_df[cloud_mask].copy()
        cloud_samples['system'] = system_name
        cloud_samples['cloud_type'] = data['cloud_type']
        all_cloud_data.append(cloud_samples)
        
        # Get candidates
        candidate_samples = results_df[results_df['query'].isin(candidate_list)].copy()
        candidate_samples['system'] = system_name
        all_candidate_data.append(candidate_samples)
        
        # Get known mimics
        mimic_samples = results_df[results_df['query'].isin(validation_list)].copy()
        mimic_samples['system'] = system_name
        all_mimic_data.append(mimic_samples)
    
    if len(all_cloud_data) == 0:
        print("No cloud data found")
        return
    
    # Combine all data
    cloud_df = pd.concat(all_cloud_data, ignore_index=True)
    candidate_df = pd.concat(all_candidate_data, ignore_index=True) if len(all_candidate_data) > 0 else pd.DataFrame()
    mimic_df = pd.concat(all_mimic_data, ignore_index=True) if len(all_mimic_data) > 0 else pd.DataFrame()
    
    # 1. Compare cloud samples between systems
    print(f"\n{'='*60}")
    print("Comparing cloud samples between systems")
    print(f"{'='*60}")
    
    comparison_rows = []
    for feature in core_features:
        if feature not in cloud_df.columns:
            continue
        
        for system in cloud_df['system'].unique():
            system_cloud = cloud_df[(cloud_df['system'] == system) & (cloud_df[feature].notna())]
            if len(system_cloud) == 0:
                continue
            
            comparison_rows.append({
                'feature': feature,
                'system': system,
                'mean': system_cloud[feature].mean(),
                'median': system_cloud[feature].median(),
                'std': system_cloud[feature].std(),
                'min': system_cloud[feature].min(),
                'max': system_cloud[feature].max(),
                'q25': system_cloud[feature].quantile(0.25),
                'q75': system_cloud[feature].quantile(0.75),
                'n_samples': len(system_cloud)
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_file = os.path.join(output_dir, 'cloud_samples_between_systems_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved cloud samples comparison: {comparison_file}")
    
    # 2. Compare candidates to cloud samples
    if len(candidate_df) > 0:
        print(f"\n{'='*60}")
        print("Comparing candidates to cloud samples")
        print(f"{'='*60}")
        
        candidate_comparison_rows = []
        for feature in core_features:
            if feature not in candidate_df.columns or feature not in cloud_df.columns:
                continue
            
            # Overall cloud statistics (across all systems)
            cloud_vals = cloud_df[feature].dropna()
            candidate_vals = candidate_df[feature].dropna()
            
            if len(cloud_vals) == 0 or len(candidate_vals) == 0:
                continue
            
            # Statistical test (Mann-Whitney U test)
            from scipy.stats import mannwhitneyu
            try:
                stat, pval = mannwhitneyu(candidate_vals, cloud_vals, alternative='two-sided')
            except:
                stat, pval = np.nan, np.nan
            
            candidate_comparison_rows.append({
                'feature': feature,
                'candidate_mean': candidate_vals.mean(),
                'candidate_median': candidate_vals.median(),
                'candidate_std': candidate_vals.std(),
                'candidate_min': candidate_vals.min(),
                'candidate_max': candidate_vals.max(),
                'cloud_mean': cloud_vals.mean(),
                'cloud_median': cloud_vals.median(),
                'cloud_std': cloud_vals.std(),
                'cloud_min': cloud_vals.min(),
                'cloud_max': cloud_vals.max(),
                'mean_difference': candidate_vals.mean() - cloud_vals.mean(),
                'median_difference': candidate_vals.median() - cloud_vals.median(),
                'mean_difference_pct': ((candidate_vals.mean() - cloud_vals.mean()) / cloud_vals.mean() * 100) if cloud_vals.mean() != 0 else np.nan,
                'mannwhitney_stat': stat,
                'mannwhitney_pvalue': pval,
                'n_candidates': len(candidate_vals),
                'n_cloud_samples': len(cloud_vals)
            })
        
        candidate_comparison_df = pd.DataFrame(candidate_comparison_rows)
        candidate_comparison_file = os.path.join(output_dir, 'candidates_vs_cloud_samples_detailed.csv')
        candidate_comparison_df.to_csv(candidate_comparison_file, index=False)
        print(f"Saved candidate comparison: {candidate_comparison_file}")
        
        # 3. Create visualization: Candidates vs Cloud samples (violin/box plots)
        fig, axes = plt.subplots(1, len(core_features), figsize=(5*len(core_features), 6))
        if len(core_features) == 1:
            axes = [axes]
        
        for feat_idx, feature in enumerate(core_features):
            ax = axes[feat_idx]
            
            if feature not in candidate_df.columns or feature not in cloud_df.columns:
                ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature)
                continue
            
            cloud_vals = cloud_df[feature].dropna()
            candidate_vals = candidate_df[feature].dropna()
            
            if len(cloud_vals) == 0 or len(candidate_vals) == 0:
                ax.text(0.5, 0.5, f'No data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(feature)
                continue
            
            # Create data for plotting
            plot_data = []
            plot_labels = []
            
            plot_data.extend(cloud_vals.values)
            plot_labels.extend(['Cloud'] * len(cloud_vals))
            
            plot_data.extend(candidate_vals.values)
            plot_labels.extend(['Candidates'] * len(candidate_vals))
            
            plot_df = pd.DataFrame({'value': plot_data, 'group': plot_labels})
            
            # Violin plot
            parts = ax.violinplot([cloud_vals.values, candidate_vals.values], 
                                  positions=[0, 1], widths=0.6, showmeans=True, showmedians=True)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Box plot overlay
            bp = ax.boxplot([cloud_vals.values, candidate_vals.values], 
                           positions=[0, 1], widths=0.3, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['blue', 'red']):
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Cloud\n(n={})'.format(len(cloud_vals)), 
                               'Candidates\n(n={})'.format(len(candidate_vals))])
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'{feature}\n(Cloud vs Candidates)', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add statistical test result
            if feature in candidate_comparison_df['feature'].values:
                row = candidate_comparison_df[candidate_comparison_df['feature'] == feature].iloc[0]
                if not pd.isna(row['mannwhitney_pvalue']):
                    sig = '***' if row['mannwhitney_pvalue'] < 0.001 else '**' if row['mannwhitney_pvalue'] < 0.01 else '*' if row['mannwhitney_pvalue'] < 0.05 else 'ns'
                    ax.text(0.5, 0.95, f"p={row['mannwhitney_pvalue']:.3e} {sig}", 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'candidates_vs_cloud_distributions.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution comparison: {plot_file}")
        
        # 4. Feature-by-feature comparison across systems
        fig, axes = plt.subplots(len(core_features), 1, figsize=(10, 4*len(core_features)))
        if len(core_features) == 1:
            axes = [axes]
        
        for feat_idx, feature in enumerate(core_features):
            ax = axes[feat_idx]
            
            if feature not in cloud_df.columns:
                continue
            
            # Plot cloud samples by system
            system_data = []
            system_labels = []
            for system in sorted(cloud_df['system'].unique()):
                system_vals = cloud_df[(cloud_df['system'] == system) & (cloud_df[feature].notna())][feature]
                if len(system_vals) > 0:
                    system_data.append(system_vals.values)
                    system_labels.append(f"{system}\n(n={len(system_vals)})")
            
            if len(system_data) > 0:
                bp = ax.boxplot(system_data, labels=system_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
            
            # Overlay candidates
            if feature in candidate_df.columns:
                candidate_vals = candidate_df[feature].dropna()
                if len(candidate_vals) > 0:
                    # Plot candidates as scatter points
                    x_positions = []
                    y_positions = []
                    for i, system in enumerate(sorted(cloud_df['system'].unique())):
                        system_candidates = candidate_df[(candidate_df['system'] == system) & 
                                                         (candidate_df[feature].notna())]
                        if len(system_candidates) > 0:
                            x_positions.extend([i+1] * len(system_candidates))
                            y_positions.extend(system_candidates[feature].values)
                    
                    if len(x_positions) > 0:
                        ax.scatter(x_positions, y_positions, color='red', s=50, alpha=0.7, 
                                 zorder=10, label='Candidates', marker='x')
            
            ax.set_ylabel(feature, fontsize=11, fontweight='bold')
            ax.set_xlabel('System', fontsize=11)
            ax.set_title(f'{feature}: Cloud samples by system (with candidates overlaid)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'cloud_samples_by_system_with_candidates.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved system-by-system comparison: {plot_file}")
    
    print(f"\nAnalysis complete. Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare cloud feature statistics across multiple systems'
    )
    parser.add_argument(
        'stat_files',
        nargs='+',
        help='Paths to cloud statistics CSV files (*_cloud_feature_statistics.csv)'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='cloud_statistics_comparison',
        help='Output directory for comparison plots (default: cloud_statistics_comparison)'
    )
    parser.add_argument(
        '--diffusion_results',
        nargs='+',
        default=None,
        help='Paths to diffusion map results CSV files (for candidate comparison). Should match systems in stat_files.'
    )
    parser.add_argument(
        '--candidates',
        nargs='+',
        default=None,
        help='List of candidate query IDs. If not provided, uses default list.'
    )
    parser.add_argument(
        '--mimics',
        nargs='+',
        default=None,
        help='List of known mimic query IDs. If not provided, uses default validation list.'
    )
    
    args = parser.parse_args()
    
    # Parse all statistics files
    print(f"Parsing {len(args.stat_files)} statistics files...")
    parsed_data_list = []
    for stat_file in args.stat_files:
        if not os.path.exists(stat_file):
            print(f"Warning: File not found: {stat_file}")
            continue
        try:
            parsed = parse_statistics_file(stat_file)
            parsed_data_list.append(parsed)
            print(f"  ✓ Parsed: {stat_file} -> {parsed['system_name']} ({parsed['cloud_type']} cloud, {len(parsed['features'])} features)")
        except Exception as e:
            print(f"  ✗ Error parsing {stat_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(parsed_data_list) == 0:
        print("Error: No valid statistics files found")
        return
    
    print(f"\nFound {len(parsed_data_list)} valid statistics files")
    
    # Create comparison plots
    create_comparison_plots(parsed_data_list, args.output_dir)
    
    # Create candidate and mimic comparison plots if diffusion results provided
    if args.diffusion_results:
        print(f"\n{'='*60}")
        print("Creating candidate and known mimic vs cloud comparison plots")
        print(f"{'='*60}")
        plot_candidates_vs_cloud(parsed_data_list, args.diffusion_results, args.output_dir, 
                                candidate_list=args.candidates, validation_list=args.mimics)
        
        # Detailed analysis of cloud differences
        print(f"\n{'='*60}")
        print("Analyzing cloud differences between systems and candidate exclusion")
        print(f"{'='*60}")
        analyze_cloud_differences(parsed_data_list, args.diffusion_results, args.output_dir,
                                 candidate_list=args.candidates, validation_list=args.mimics)
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()
