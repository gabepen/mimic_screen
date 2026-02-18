import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import SpectralEmbedding
    from scipy.sparse import csgraph
    from scipy.spatial.distance import pdist, squareform
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from gensim.models import Word2Vec
import argparse
import re
import pdb

candidate_list = ['Q73HU7', 'Q73IF8', 'Q73GY6', 'Q73GG5', 'Q73HX8', 'P61189', 'O25525', 'O25981']


'''
validation_list = [
    'Q5ZVD8', 'Q5ZTI6', 'Q5ZUA2', 'Q5ZSI9', 'Q5ZUX1', 'Q5ZWA1', 
    'Q5ZU58', 'Q5ZU32', 'Q5ZUS4', 'Q5ZRQ0', 'Q5ZSZ6', 'Q5ZVF7', 
    'Q5ZYU9', 'Q5ZTM4', 'Q5ZSB6', 'Q5F2U4', 'Q5ZW23'
]
'''
#validation_list = ['Q5ZRQ0', 'Q5ZSJ8', 'Q5ZUS4', 'Q5ZQL5', 'Q5ZQL4', 'Q5ZQL3', 'Q5ZWD4', 'Q5ZQV8', 'Q5ZPT1', 'Q5ZXN1', 'Q5ZYU2', 'Q5ZSQ7']
validation_list = ['Q5ZU58', 'Q5ZXE0', 'Q5ZU32', 'Q5ZSB6', 'Q5ZRP9', 'Q5ZT65', 'Q5ZXN6', 'Q5ZVF7', 'Q5ZTE0', 'Q5ZTL7', 'Q5ZSQ2', 'Q5ZTM4', 'Q5ZU83', 'Q5ZVS2', 'Q5ZWA1']


def generate_label_color_dict(labels):
    
    unique_labels = labels.unique()
    label_counts = labels.value_counts()
    label_rank_dict = {label: rank for rank, label in enumerate(label_counts.index, 1)}
    '''
    colors = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
        'magenta', 'yellow', 'black', 'teal', 'lime', 'navy', 'maroon', 'gold', 'silver', 'violet'
    ]
    '''
    # Use a combination of tab20, tab20b, and tab20c colormaps for more colors
    colors = list(plt.cm.get_cmap('tab20', 20).colors) + \
             list(plt.cm.get_cmap('tab20b', 20).colors) + \
             list(plt.cm.get_cmap('tab20c', 20).colors)
             
    if len(unique_labels) > len(colors):
        raise ValueError("Number of unique labels exceeds the number of available colors.")
    label_color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
    return label_color_dict, label_rank_dict

def semantic_clustering_word2vec(data_frame, term, n_clusters=10):
    
    data_frame['target_molecular_functions'].apply(lambda terms: [re.sub(r'\W+', ' ', term.lower()) for term in terms])
    
    #Train Word2Vec model
    sentences = data_frame['target_molecular_functions']
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Get the vector representation of each unique term
    unique_terms = list(set(term for sublist in sentences for term in sublist))
    term_vectors = np.array([word2vec_model.wv[term] for term in unique_terms])

    # Create a mapping from terms to their vectors
    term_vector_map = {term: vector for term, vector in zip(unique_terms, term_vectors)}
    
    # Cluster the term vectors into 10-20 groups
    n_clusters = 10  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(term_vectors)

    # Map terms to their clusters
    term_cluster_map = dict(zip(unique_terms, clusters))

    # Find the representative term for each cluster
    representative_terms = []
    for i in range(n_clusters):
        cluster_terms = [term for term, cluster in term_cluster_map.items() if cluster == i]
        if cluster_terms:
            centroid = kmeans.cluster_centers_[i]
            term_vectors = np.array([term_vector_map[term] for term in cluster_terms])
            closest_term_index = np.argmin(np.linalg.norm(term_vectors - centroid, axis=1))
            representative_terms.append(cluster_terms[closest_term_index])

    print("word2vec Representative terms:", representative_terms)

def semantic_clustering(data_frame, term, n_clusters=15):

    def assign_protein_to_group(annotations):
        
        cluster_counts = {i: 0 for i in range(n_clusters)}
        for annotation in annotations:
            cluster = term_cluster_map.get(annotation, -1)
            if cluster != -1:
                cluster_counts[cluster] += 1

        rep_term_index = max(cluster_counts, key=cluster_counts.get)
        return representative_terms[rep_term_index]

    data_frame[term] = data_frame[term].apply(lambda terms: [re.sub(r'\W+', ' ', term.lower()) for term in terms])
    
    # flatten list for vectorization 
    all_annotations = [term for sublist in data_frame[term] for term in sublist]
    
    # Convert annotations to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_annotations)
    
    # Cluster the annotations into 10-20 groups
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=12, max_iter=600)
    clusters = kmeans.fit_predict(X)

    # Map terms to their clusters
    term_cluster_map = dict(zip(all_annotations, clusters))
    
    import numpy as np

    # Find the representative term for each cluster
    representative_terms = []
    for i in range(n_clusters):
        cluster_terms = [term for term, cluster in term_cluster_map.items() if cluster == i]
        if cluster_terms:
            centroid = kmeans.cluster_centers_[i]
            term_vectors = vectorizer.transform(cluster_terms)
            closest_term_index = np.argmin(np.linalg.norm(term_vectors.toarray() - centroid, axis=1))
            representative_terms.append(cluster_terms[closest_term_index])

    data_frame['simplified_labels'] = data_frame[term].apply(assign_protein_to_group)
    
    labels = data_frame['simplified_labels']
   
    # Select specific columns for analysis
    selected_columns = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen' ]  # Replace with your column names
    test_data_frame = data_frame[selected_columns]

    
    return test_data_frame, labels, selected_columns

def generate_candidate_list(data_frame, algn_threshold, fident_threshold):
    
    candidate_list = []
    for index, row in data_frame.iterrows():
        if float(row[6]) < algn_threshold and float(row[5]) < fident_threshold:
            candidate_list.append(row[0])
    return candidate_list

def classify_label_by_cell_loc(x):
    
    x = ''.join(x).lower()
    
    if 'mitochondrial small ribosomal subunit' in x or 'mitochondrial large ribosomal subunit' in x:
        return 'mitchondrial ribosomal'
    elif 'mitochondrial inner membrane' in x or 'mitochondrial outer membrane' in x:
        return 'mitochondrial membrane'
    elif 'mitochondrial respiratory' in x:
        return 'mitochondrial respiratory chain'
    elif 'peroxisome' in x:
        return 'peroxisomal'
    elif 'mitochon' in x:
        return 'mitochondrial'
    elif 'endoplasmic' in x:
        return 'ER'
    elif 'golgi' in x:
        return 'GA'
    elif 'extracellular' in x:
        return 'extracellular'
    elif len(x) == 0:
        return 'missing'
    else:
        return 'other'

def classify_label_by_enzyme_func(x):
    
    x = ''.join(x).lower()
    pass
    
def apply_labels(data_frame):
    
     # Store the categorical value in a separate variable
    data_frame['simplified_labels'] = data_frame['target_cellular_components'].apply(classify_label_by_cell_loc)
    
    labels = data_frame['simplified_labels']
    
    # Select specific columns for analysis
    selected_columns = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen' ]  # Replace with your column names
    test_data_frame = data_frame[selected_columns]
    
    return test_data_frame, labels, selected_columns
    
def lda_pca_analysis(pca_test_dir, data_frame, organism_name, go_term_type, n_clusters=20, use_semantic_clustering=True):
    
    #go_term_type = 'target_cellular_components'
    #n_clusters = 20
    if use_semantic_clustering:
        test_data_frame, labels, selected_columns = semantic_clustering(data_frame, go_term_type, n_clusters)
    else:
        test_data_frame, labels, selected_columns = apply_labels(data_frame)
    
    #candidate_list = generate_candidate_list(data_frame, 0.1, 0.3)
    print('Candidates within threshold:', len(candidate_list))
     # label data by groups 
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(test_data_frame)
    
    # Set the number of components for LDA
    n_components = min(len(selected_columns),len(np.unique(labels)) - 1)
    
    # Run LDA
    lda = LDA(n_components=n_components)
    lda_data = lda.fit_transform(scaled_data, labels)
    
    # Run PCA
    pca = PCA()
    pca.fit(lda_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    print(f"Number of components to retain 95% of the variance: {n_components}")
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(lda_data)
    
       
    # Create a DataFrame for the PCA results
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)
    pca_df['query'] = data_frame['query'] # Assuming 'query' is the index or a column in the original DataFrame
    pca_df['label'] = labels.values
    
    # Change the label for rows whose query_id matches a value in candidate_list
    pca_df['label'] = pca_df.apply(
        lambda row: 'candidate' if row['query'] in candidate_list else row['label'], axis=1
    )
    
    pca_df['label'] = pca_df.apply(
        lambda row: 'mimic' if row['query'] in validation_list else row['label'], axis=1
    )
    
    
    color_dict, rank_dict = generate_label_color_dict(pca_df['label'])
    color_dict['candidate'] = 'cyan'
    color_dict['mimic'] = 'mediumspringgreen'
    rank_dict['candidate'] = n_clusters + 3
    rank_dict['mimic'] = n_clusters + 2
    
    if go_term_type == 'target_cellular_components':
        go_term_out = 'cellcomp'
    elif go_term_type == 'target_molecular_functions':
        go_term_out = 'molfunc'
    else:
        go_term_out = 'bioproc'

    # Save pca_df to a CSV file
    pca_df.to_csv(pca_test_dir + f'/{organism_name}_pca_results_{go_term_out}_{n_clusters}.csv', index=False)
    
    plt.clf()
    
    # Add labels to each point
    '''
    color_dict = {
        'mitchondrial ribosomal': 'red',
        'mitochondrial membrane': 'orange',
        'mitochondrial respiratory chain': 'yellow',
        'peroxisomal': 'blue',
        'mitochondrial': 'purple',
        'ER': 'green',
        'GA': 'brown',
        'extracellular': 'pink',
        'missing': 'grey',
        'other': 'black',
        'candidate': 'cyan', 
        'mimic': 'mediumspringgreen'
    }
    '''
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    colors = color_dict
    for i, row in pca_df.iterrows():
        x = row['PC1']
        y = row['PC2'] if 'PC2' in pca_df.columns else 0  # Use the first two components for the 2D plot
        
        
        if row['label'] == 'candidate':
            plt.scatter(x, y, c=colors[row['label']], edgecolor='k', s=35, zorder=rank_dict[row['label']])
        elif row['label'] == 'mimic':
            plt.scatter(x, y, c=colors[row['label']], edgecolor='k', s=20, zorder=rank_dict[row['label']])
        elif row['label'] == 'missing' or row['label'] == 'other':
            plt.scatter(x, y, c=colors[row['label']], s=4, zorder=rank_dict[row['label']])
        else:
            plt.scatter(x, y, c=colors[row['label']], s=10, zorder=rank_dict[row['label']])
        
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in colors.items()]
    plt.legend(handles=handles, title=' '.join(go_term_type.split('_')[1:]).capitalize(), title_fontsize='8', fontsize='6',  bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'LDA-PCA of Alignment Features {organism_name.capitalize()}')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
    plt.savefig(pca_test_dir + f'/lda_pca_viz_{organism_name}_{go_term_out}_{n_clusters}.png', dpi=300)
    plt.close()
    
def tsne_analysis(pca_test_dir, data_frame):
    
    # label data by groups 
    test_data_frame, labels, selected_columns = apply_labels(data_frame)
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(test_data_frame)
   
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(scaled_data)
    print(tsne.kl_divergence_) # want this low
    
     # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['query'] = data_frame['query']  # Assuming 'query' is the index or a column in the original DataFrame
    tsne_df['label'] = labels.values
    
    # Change the label for rows whose query_id matches a value in candidate_list
    tsne_df['label'] = tsne_df.apply(
        lambda row: 'candidate' if row['query'] in candidate_list else row['label'], axis=1
    )
    
     # Add labels to each point
    colors = {
        'mitchondrial ribosomal': 'red',
        'mitochondrial membrane': 'orange',
        'mitochondrial respiratory chain': 'yellow',
        'peroxisomal': 'blue',
        'mitochondrial': 'purple',
        'ER': 'green',
        'GA': 'brown',
        'extracellular': 'pink',
        'missing': 'grey',
        'other': 'black',
        'candidate': 'cyan'
    }
    plt.clf()
    separate_plots = False
    if separate_plots:
        unique_labels = tsne_df['label'].unique()
        for label in unique_labels:
            plt.clf()
            subset_df = tsne_df[tsne_df['label'] == label]
            for i, row in subset_df.iterrows():
                x = row['TSNE1']
                y = row['TSNE2']
                plt.scatter(x, y, c=colors[row['label']], edgecolor='k', s=50)
            plt.title(f't-SNE Visualization for {label}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(f'{pca_test_dir}/tsne_viz_{label}.png')
            plt.close()
    
    else:
        for i, row in tsne_df.iterrows():
            x = row['TSNE1']
            y = row['TSNE2']
            if row['label'] == 'candidate':
                plt.scatter(x, y, c=colors[row['label']], edgecolor='k', s=35, zorder=10)
            elif row['label'] == 'missing' or row['label'] == 'other':
                plt.scatter(x, y, c=colors[row['label']], s=4)
            else:
                plt.scatter(x, y, c=colors[row['label']], s=10)
        
    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in colors.items()]
    #plt.legend(handles=handles, title='Cell Location', title_fontsize='8', fontsize='6')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of Alignment Features Legionella')
    plt.savefig(pca_test_dir + '/tsne_viz_lp.png')
    plt.close()

def pca_tsne_analysis(pca_test_dir, data_frame):
    
     # label data by groups 
    test_data_frame, labels, selected_columns = apply_labels(data_frame)
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(test_data_frame)
     
    # Run PCA
    pca = PCA()
    pca.fit(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(test_data_frame)
    print(tsne.kl_divergence_) # want this low
    
     # Create a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['query'] = data_frame['query']  # Assuming 'query' is the index or a column in the original DataFrame
    tsne_df['label'] = labels.values
    
    # Change the label for rows whose query_id matches a value in candidate_list
    tsne_df['label'] = tsne_df.apply(
        lambda row: 'candidate' if row['query'] in candidate_list else row['label'], axis=1
    )
    
     # Add labels to each point
    colors = {
        'mitchondrial ribosomal': 'red',
        'mitochondrial membrane': 'orange',
        'mitochondrial respiratory chain': 'yellow',
        'peroxisomal': 'blue',
        'mitochondrial': 'purple',
        'ER': 'green',
        'GA': 'brown',
        'extracellular': 'pink',
        'missing': 'grey',
        'other': 'black',
        'candidate': 'cyan'
    }
    
    plt.clf()
    for i, row in tsne_df.iterrows():
        x = row['TSNE1']
        y = row['TSNE2']
        if row['label'] == 'candidate':
            plt.scatter(x, y, c=colors[row['label']], edgecolor='k', s=35, zorder=10)
            print(row)
        elif row['label'] == 'missing' or row['label'] == 'other':
            plt.scatter(x, y, c=colors[row['label']], s=4)
        else:
            plt.scatter(x, y, c=colors[row['label']], s=10)
    
    # Create a custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in colors.items()]
    #plt.legend(handles=handles, title='Cell Location', title_fontsize='8', fontsize='6')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('PCA/t-SNE of Alignment Features ')
    plt.savefig(pca_test_dir + '/tsne_pca_viz_hp.png')
    plt.close()
    
def compute_robust_axis_limits(data, percentile_low=1.0, percentile_high=99.0, padding_factor=0.05):
    """
    Compute robust axis limits based on percentiles to handle outliers.
    
    Args:
        data: Array of values
        percentile_low: Lower percentile (default: 1.0)
        percentile_high: Upper percentile (default: 99.0)
        padding_factor: Additional padding as fraction of range (default: 0.05)
    
    Returns:
        (min_limit, max_limit) tuple
    """
    if len(data) == 0:
        return (0, 1)
    
    low = np.percentile(data, percentile_low)
    high = np.percentile(data, percentile_high)
    
    # Add padding
    range_val = high - low
    if range_val == 0:
        # If all values are the same, add small padding
        padding = abs(low) * 0.1 if low != 0 else 0.1
        return (low - padding, high + padding)
    
    padding = range_val * padding_factor
    return (low - padding, high + padding)


def define_mimic_cloud_in_feature_space(scaled_data, data_frame, mimic_queries, output_file=None, feature_columns=None):
    """
    Define mimic cloud in feature space (before diffusion mapping).
    This cloud definition can be transferred to other datasets.
    
    With adaptive epsilon, different systems may have different scalers. This function
    saves both scaled and raw feature values to enable proper transfer across systems.
    
    Parameters:
    -----------
    scaled_data : array-like
        Scaled feature matrix (n_samples, n_features)
    data_frame : DataFrame
        DataFrame with 'query' column and feature columns (for raw values)
    mimic_queries : list
        List of query IDs that are known mimics
    output_file : str, optional
        Path to save cloud definition (JSON format)
    feature_columns : list, optional
        List of feature column names used. If None, will try to infer from data_frame.
        This is needed to extract raw feature values for transfer.
    
    Returns:
    --------
    cloud_definition : dict
        Dictionary containing:
        - 'core_mimic_features': array of core mimic feature vectors (scaled)
        - 'core_mimic_features_raw': array of core mimic feature vectors (raw, unscaled)
        - 'core_mimic_queries': list of query IDs for core mimics
        - 'threshold_within': distance threshold for "within cloud" (in scaled space)
        - 'threshold_around': distance threshold for "around cloud" (in scaled space)
        - 'feature_columns': list of feature column names
        - 'all_mimic_features': array of all mimic feature vectors (scaled, for reference)
        - 'all_mimic_queries': list of all mimic query IDs
    """
    import json
    from scipy.spatial.distance import cdist
    from scipy.sparse import csgraph
    
    # Find mimic indices
    mimic_mask = data_frame['query'].isin(mimic_queries).values
    mimic_indices = np.where(mimic_mask)[0]
    
    if len(mimic_indices) == 0:
        print("Warning: No mimics found in dataset")
        return None
    
    print(f"\nDefining mimic cloud in feature space...")
    print(f"  Found {len(mimic_indices)} mimic alignments")
    
    # Get mimic features (scaled)
    mimic_features = scaled_data[mimic_indices]
    mimic_queries_found = data_frame.iloc[mimic_indices]['query'].values
    
    # Get raw (unscaled) feature values for transfer across systems
    # This allows proper scaling with target system's scaler
    if feature_columns is None:
        # Try to infer feature columns from data_frame
        # Exclude non-feature columns
        exclude_cols = ['query', 'target', 'system', 'label', 'cluster', 'go_term']
        feature_columns = [col for col in data_frame.columns 
                         if col not in exclude_cols and 
                         data_frame[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        if len(feature_columns) != scaled_data.shape[1]:
            print(f"  Warning: Could not infer feature columns. Found {len(feature_columns)} columns but scaled_data has {scaled_data.shape[1]} features.")
            print(f"  Using scaled features only (may not transfer well across systems with different scalers).")
            feature_columns = None
    
    if feature_columns is not None and len(feature_columns) == scaled_data.shape[1]:
        mimic_features_raw = data_frame.iloc[mimic_indices][feature_columns].values
        print(f"  Saving raw feature values for cross-system transfer (columns: {feature_columns[:5]}...)")
    else:
        mimic_features_raw = None
        feature_columns = None
        print(f"  Warning: Raw features not available. Cloud transfer may not work correctly across systems with different scalers.")
    
    # Compute pairwise distances between mimics in feature space
    mimic_pairwise_distances = cdist(mimic_features, mimic_features, metric='euclidean')
    upper_triangle = np.triu(mimic_pairwise_distances, k=1)
    mimic_pairwise_flat = upper_triangle[upper_triangle > 0]
    
    if len(mimic_pairwise_flat) == 0:
        # Only one unique mimic
        print("  Only one unique mimic position, using all mimics")
        core_mimic_indices = list(range(len(mimic_indices)))
        cluster_threshold = 0.0
    else:
        # Calculate distance statistics
        q25_pairwise_dist = np.percentile(mimic_pairwise_flat, 25)
        q50_pairwise_dist = np.median(mimic_pairwise_flat)
        q75_pairwise_dist = np.percentile(mimic_pairwise_flat, 75)
        max_pairwise_dist = np.max(mimic_pairwise_flat)
        
        print(f"  Mimic pairwise distance statistics:")
        print(f"    25th percentile: {q25_pairwise_dist:.6f}")
        print(f"    Median: {q50_pairwise_dist:.6f}")
        print(f"    75th percentile: {q75_pairwise_dist:.6f}")
        print(f"    Max: {max_pairwise_dist:.6f}")
        
        # Try progressively tighter thresholds to find main cluster
        thresholds_to_try = [
            q25_pairwise_dist * 0.75,  # Tightest
            q25_pairwise_dist,        # 25th percentile
            q50_pairwise_dist,        # Median
        ]
        
        best_cluster = None
        best_threshold = None
        min_cluster_size = max(2, len(mimic_indices) // 3)
        
        for cluster_threshold in thresholds_to_try:
            # Build adjacency graph
            adjacency = (mimic_pairwise_distances <= cluster_threshold).astype(int)
            np.fill_diagonal(adjacency, 0)
            
            # Find connected components
            n_components, labels = csgraph.connected_components(
                csgraph=adjacency, directed=False, return_labels=True
            )
            
            if n_components > 0:
                component_sizes = [np.sum(labels == i) for i in range(n_components)]
                component_sizes_sorted = sorted(component_sizes, reverse=True)
                largest_component_size = component_sizes_sorted[0]
                largest_component_idx = np.argmax(component_sizes)
                largest_component = np.where(labels == largest_component_idx)[0]
                
                # Check if this is a good cluster
                is_good_cluster = (
                    largest_component_size >= min_cluster_size and
                    largest_component_size < len(mimic_indices)  # Excludes at least one
                )
                
                if len(component_sizes_sorted) > 1:
                    second_largest = component_sizes_sorted[1]
                    is_dominant = largest_component_size >= second_largest * 2
                    is_good_cluster = is_good_cluster and is_dominant
                
                if is_good_cluster:
                    best_cluster = largest_component
                    best_threshold = cluster_threshold
                    print(f"  Using threshold: {cluster_threshold:.6f}")
                    print(f"  Found main cluster with {len(best_cluster)} mimics (excludes {len(mimic_indices) - len(best_cluster)} outlier(s))")
                    break
        
        # Use best cluster or fallback
        if best_cluster is not None:
            core_mimic_indices = best_cluster.tolist()
            cluster_threshold = best_threshold
        else:
            # Fallback: use tight threshold
            cluster_threshold = q25_pairwise_dist * 0.5
            print(f"  No ideal cluster found, using tight threshold ({cluster_threshold:.6f})")
            
            adjacency = (mimic_pairwise_distances <= cluster_threshold).astype(int)
            np.fill_diagonal(adjacency, 0)
            n_components, labels = csgraph.connected_components(
                csgraph=adjacency, directed=False, return_labels=True
            )
            
            if n_components > 0:
                component_sizes = [np.sum(labels == i) for i in range(n_components)]
                largest_component_idx = np.argmax(component_sizes)
                core_mimic_indices = np.where(labels == largest_component_idx)[0].tolist()
            else:
                # Last resort: use all mimics
                core_mimic_indices = list(range(len(mimic_indices)))
    
    # Get core mimic features (scaled)
    core_mimic_features = mimic_features[core_mimic_indices]
    core_mimic_queries = mimic_queries_found[core_mimic_indices].tolist()
    
    # Get core mimic features (raw, if available)
    if mimic_features_raw is not None:
        core_mimic_features_raw = mimic_features_raw[core_mimic_indices]
    else:
        core_mimic_features_raw = None
    
    # Calculate thresholds for cloud definition
    if len(core_mimic_indices) > 1:
        core_pairwise_distances = cdist(core_mimic_features, core_mimic_features, metric='euclidean')
        upper_triangle_core = np.triu(core_pairwise_distances, k=1)
        core_pairwise_flat = upper_triangle_core[upper_triangle_core > 0]
        
        if len(core_pairwise_flat) > 0:
            q75_core = np.percentile(core_pairwise_flat, 75)
            q90_core = np.percentile(core_pairwise_flat, 90)
            threshold_within = q75_core
            threshold_around = q90_core * 1.25  # 90th percentile + 25% buffer
        else:
            threshold_within = 0.0
            threshold_around = 0.0
    else:
        threshold_within = 0.0
        threshold_around = 0.0
    
    print(f"  Cloud thresholds:")
    print(f"    Within cloud: {threshold_within:.6f}")
    print(f"    Around cloud: {threshold_around:.6f}")
    
    # Create cloud definition (keep as numpy arrays for immediate use)
    cloud_definition = {
        'core_mimic_features': core_mimic_features,  # Scaled features (for immediate use)
        'core_mimic_queries': core_mimic_queries,
        'all_mimic_features': mimic_features,  # Scaled features (for reference)
        'all_mimic_queries': mimic_queries_found.tolist(),
        'threshold_within': float(threshold_within),
        'threshold_around': float(threshold_around),
        'n_core_mimics': len(core_mimic_indices),
        'n_total_mimics': len(mimic_indices),
        'feature_columns': feature_columns,  # Column names for raw features
        'core_mimic_features_raw': core_mimic_features_raw,  # Raw features (for cross-system transfer)
    }
    
    # Save to file if specified (convert to lists for JSON)
    if output_file:
        cloud_definition_for_json = cloud_definition.copy()
        cloud_definition_for_json['core_mimic_features'] = core_mimic_features.tolist()
        cloud_definition_for_json['all_mimic_features'] = mimic_features.tolist()
        if core_mimic_features_raw is not None:
            cloud_definition_for_json['core_mimic_features_raw'] = core_mimic_features_raw.tolist()
        else:
            cloud_definition_for_json['core_mimic_features_raw'] = None
        with open(output_file, 'w') as f:
            json.dump(cloud_definition_for_json, f, indent=2)
        print(f"  Saved cloud definition to: {output_file}")
        if feature_columns is not None:
            print(f"  Includes raw features for cross-system transfer")
    
    return cloud_definition


def load_mimic_cloud_definition(cloud_file):
    """
    Load a saved mimic cloud definition from JSON file.
    
    Parameters:
    -----------
    cloud_file : str
        Path to JSON file containing cloud definition
    
    Returns:
    --------
    cloud_definition : dict
        Cloud definition dictionary
    """
    import json
    
    with open(cloud_file, 'r') as f:
        cloud_definition = json.load(f)
    
    # Convert lists back to numpy arrays
    cloud_definition['core_mimic_features'] = np.array(cloud_definition['core_mimic_features'])
    cloud_definition['all_mimic_features'] = np.array(cloud_definition['all_mimic_features'])
    
    # Convert raw features if present
    if 'core_mimic_features_raw' in cloud_definition and cloud_definition['core_mimic_features_raw'] is not None:
        cloud_definition['core_mimic_features_raw'] = np.array(cloud_definition['core_mimic_features_raw'])
    else:
        cloud_definition['core_mimic_features_raw'] = None
    
    return cloud_definition


def report_cloud_feature_statistics(data_frame, cloud_results, cloud_name="cloud", output_file=None):
    """
    Report feature statistics (mean, min, max) for alignments within, around, and outside the cloud.
    
    Parameters:
    -----------
    data_frame : DataFrame
        DataFrame with feature columns
    cloud_results : dict
        Dictionary with 'within_cloud' and 'around_cloud' boolean arrays
    cloud_name : str
        Name of the cloud for reporting (e.g., "transferred cloud", "local cloud")
    output_file : str, optional
        Path to file to save statistics. If None, only prints to console.
    """
    within_cloud = cloud_results['within_cloud']
    around_cloud = cloud_results['around_cloud']
    outside_cloud = ~around_cloud
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['query', 'target', 'system', 'label', 'cluster', 'go_term', 
                     'target_cellular_components', 'target_molecular_functions', 
                     'target_biological_processes', 'targets_likely_paralogs']
    
    # Also exclude result columns that might have been added
    exclude_cols.extend([col for col in data_frame.columns 
                        if any(x in col.lower() for x in ['diffusion', 'pseudotime', 'cloud', 'distance', 'trajectory'])])
    
    feature_cols = [col for col in data_frame.columns 
                   if col not in exclude_cols and 
                   data_frame[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    # Filter to numeric columns only
    feature_cols = [col for col in feature_cols if data_frame[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    if len(feature_cols) == 0:
        print(f"  Warning: No feature columns found for statistics")
        return
    
    # Prepare output (both print and file)
    output_lines = []
    
    def add_line(line):
        output_lines.append(line)
        print(line)
    
    add_line(f"\n{'='*60}")
    add_line(f"Feature Statistics for {cloud_name}")
    add_line(f"{'='*60}")
    
    # Get data for each region
    within_data = data_frame.loc[within_cloud, feature_cols]
    around_not_within = around_cloud & ~within_cloud
    around_data = data_frame.loc[around_not_within, feature_cols] if around_not_within.sum() > 0 else pd.DataFrame()
    outside_data = data_frame.loc[outside_cloud, feature_cols]
    
    n_within = within_cloud.sum()
    n_around = around_not_within.sum()
    n_outside = outside_cloud.sum()
    n_total = len(data_frame)
    
    add_line(f"\nSample counts:")
    add_line(f"  Within cloud: {n_within} ({n_within/n_total*100:.1f}%)")
    add_line(f"  Around cloud (not within): {n_around} ({n_around/n_total*100:.1f}%)")
    add_line(f"  Outside cloud: {n_outside} ({n_outside/n_total*100:.1f}%)")
    add_line(f"  Total: {n_total}")
    
    if n_within == 0:
        add_line(f"\n  Warning: No samples within cloud - cannot compute statistics")
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_lines))
        return
    
    # Compute statistics for each region
    add_line(f"\n{'Feature':<30} {'Region':<20} {'Mean':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    add_line(f"{'-'*30} {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for col in feature_cols:
        # Within cloud
        if n_within > 0:
            within_vals = within_data[col].dropna()
            if len(within_vals) > 0:
                add_line(f"{col[:29]:<30} {'Within cloud':<20} "
                      f"{within_vals.mean():>12.4f} {within_vals.min():>12.4f} "
                      f"{within_vals.max():>12.4f} {within_vals.std():>12.4f}")
        
        # Around cloud (not within)
        if n_around > 0:
            around_vals = around_data[col].dropna() if len(around_data) > 0 else pd.Series(dtype=float)
            if len(around_vals) > 0:
                add_line(f"{col[:29]:<30} {'Around cloud':<20} "
                      f"{around_vals.mean():>12.4f} {around_vals.min():>12.4f} "
                      f"{around_vals.max():>12.4f} {around_vals.std():>12.4f}")
        
        # Outside cloud
        if n_outside > 0:
            outside_vals = outside_data[col].dropna()
            if len(outside_vals) > 0:
                add_line(f"{col[:29]:<30} {'Outside cloud':<20} "
                      f"{outside_vals.mean():>12.4f} {outside_vals.min():>12.4f} "
                      f"{outside_vals.max():>12.4f} {outside_vals.std():>12.4f}")
        
        # Add separator between features
        if col != feature_cols[-1]:
            add_line("")
    
    # Summary comparison
    add_line(f"\n{'='*60}")
    add_line(f"Summary: Feature Ranges")
    add_line(f"{'='*60}")
    add_line(f"{'Feature':<30} {'Within Cloud Range':<30} {'Outside Cloud Range':<30}")
    add_line(f"{'-'*30} {'-'*30} {'-'*30}")
    
    for col in feature_cols[:15]:  # Show top 15 features to avoid too much output
        within_vals = within_data[col].dropna()
        outside_vals = outside_data[col].dropna()
        
        if len(within_vals) > 0 and len(outside_vals) > 0:
            within_range = f"[{within_vals.min():.3f}, {within_vals.max():.3f}]"
            outside_range = f"[{outside_vals.min():.3f}, {outside_vals.max():.3f}]"
            add_line(f"{col[:29]:<30} {within_range:<30} {outside_range:<30}")
    
    if len(feature_cols) > 15:
        add_line(f"\n  ... and {len(feature_cols) - 15} more features")
    
    add_line(f"{'='*60}\n")
    
    # Save to file if specified
    if output_file:
        # Save as CSV for easier analysis
        csv_file = output_file.replace('.txt', '.csv')
        
        # Create DataFrame with statistics
        stats_rows = []
        for col in feature_cols:
            # Within cloud
            if n_within > 0:
                within_vals = within_data[col].dropna()
                if len(within_vals) > 0:
                    stats_rows.append({
                        'feature': col,
                        'region': 'within_cloud',
                        'mean': within_vals.mean(),
                        'min': within_vals.min(),
                        'max': within_vals.max(),
                        'std': within_vals.std(),
                        'count': len(within_vals)
                    })
            
            # Around cloud (not within)
            if n_around > 0:
                around_vals = around_data[col].dropna() if len(around_data) > 0 else pd.Series(dtype=float)
                if len(around_vals) > 0:
                    stats_rows.append({
                        'feature': col,
                        'region': 'around_cloud',
                        'mean': around_vals.mean(),
                        'min': around_vals.min(),
                        'max': around_vals.max(),
                        'std': around_vals.std(),
                        'count': len(around_vals)
                    })
            
            # Outside cloud
            if n_outside > 0:
                outside_vals = outside_data[col].dropna()
                if len(outside_vals) > 0:
                    stats_rows.append({
                        'feature': col,
                        'region': 'outside_cloud',
                        'mean': outside_vals.mean(),
                        'min': outside_vals.min(),
                        'max': outside_vals.max(),
                        'std': outside_vals.std(),
                        'count': len(outside_vals)
                    })
        
        stats_df = pd.DataFrame(stats_rows)
        
        # Add sample count summary as metadata
        summary_df = pd.DataFrame([{
            'region': 'within_cloud',
            'count': n_within,
            'percent': n_within/n_total*100
        }, {
            'region': 'around_cloud',
            'count': n_around,
            'percent': n_around/n_total*100
        }, {
            'region': 'outside_cloud',
            'count': n_outside,
            'percent': n_outside/n_total*100
        }, {
            'region': 'total',
            'count': n_total,
            'percent': 100.0
        }])
        
        # Save CSV
        stats_df.to_csv(csv_file, index=False)
        print(f"  Saved feature statistics (CSV) to: {csv_file}")
        
        # Also save summary
        summary_file = csv_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Keep text file for backward compatibility
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"  Saved feature statistics (text) to: {output_file}")


def apply_mimic_cloud_to_dataset(scaled_data, cloud_definition, data_frame=None, scaler=None):
    """
    Apply a mimic cloud definition (from another dataset) to a new dataset.
    
    With adaptive epsilon, different systems may have different scalers. This function
    can use raw features from the cloud definition and scale them with the target system's scaler.
    
    Parameters:
    -----------
    scaled_data : array-like
        Scaled feature matrix for new dataset (n_samples, n_features)
    cloud_definition : dict
        Cloud definition from define_mimic_cloud_in_feature_space or load_mimic_cloud_definition
    data_frame : DataFrame, optional
        DataFrame with raw feature columns. Required if using raw features for transfer.
    scaler : sklearn scaler, optional
        Scaler fitted on the target system. Required if using raw features for transfer.
        If provided along with raw features, will scale the cloud definition's raw features
        using this scaler instead of using the pre-scaled features.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'within_cloud': boolean array indicating samples within cloud
        - 'around_cloud': boolean array indicating samples around cloud
        - 'min_distances': array of minimum distances to core mimics
        - 'distances_to_center': array of distances to cloud center
    """
    from scipy.spatial.distance import cdist
    
    # Check if we should use raw features (better for cross-system transfer)
    use_raw_features = (
        'core_mimic_features_raw' in cloud_definition and 
        cloud_definition['core_mimic_features_raw'] is not None and
        'feature_columns' in cloud_definition and
        cloud_definition['feature_columns'] is not None and
        scaler is not None and
        data_frame is not None
    )
    
    if use_raw_features:
        # Use raw features and scale with target system's scaler
        print("  Using raw features from cloud definition, scaling with target system's scaler")
        feature_columns = cloud_definition['feature_columns']
        
        # Check that feature columns exist in target data_frame
        missing_cols = [col for col in feature_columns if col not in data_frame.columns]
        if missing_cols:
            print(f"  Warning: Missing feature columns in target data: {missing_cols}")
            print(f"  Falling back to pre-scaled features (may not transfer well)")
            use_raw_features = False
        else:
            # Extract raw features from cloud definition and scale with target scaler
            core_mimic_features_raw = cloud_definition['core_mimic_features_raw']
            core_mimic_features = scaler.transform(core_mimic_features_raw)
            
            # Recalculate thresholds in the target system's scaled space
            # This is approximate - we scale the raw features and recompute distances
            core_pairwise_distances = cdist(core_mimic_features, core_mimic_features, metric='euclidean')
            upper_triangle = np.triu(core_pairwise_distances, k=1)
            core_pairwise_flat = upper_triangle[upper_triangle > 0]
            
            if len(core_pairwise_flat) > 0:
                q75_core = np.percentile(core_pairwise_flat, 75)
                q90_core = np.percentile(core_pairwise_flat, 90)
                threshold_within = q75_core
                threshold_around = q90_core * 1.25
            else:
                threshold_within = cloud_definition['threshold_within']
                threshold_around = cloud_definition['threshold_around']
            
            print(f"  Recalculated thresholds in target system's scaled space:")
            print(f"    Within cloud: {threshold_within:.6f} (original: {cloud_definition['threshold_within']:.6f})")
            print(f"    Around cloud: {threshold_around:.6f} (original: {cloud_definition['threshold_around']:.6f})")
    
    if not use_raw_features:
        # Use pre-scaled features (original behavior)
        core_mimic_features = cloud_definition['core_mimic_features']
        threshold_within = cloud_definition['threshold_within']
        threshold_around = cloud_definition['threshold_around']
        print("  Using pre-scaled features from cloud definition")
        print("  Note: This assumes source and target systems use similar scaling")
    
    # Check feature dimensions match
    if scaled_data.shape[1] != core_mimic_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: new data has {scaled_data.shape[1]} features, "
            f"cloud definition expects {core_mimic_features.shape[1]} features"
        )
    
    # Compute distances from all samples to core mimics
    distances_to_mimics = cdist(scaled_data, core_mimic_features, metric='euclidean')
    min_distances = np.min(distances_to_mimics, axis=1)
    
    # Apply thresholds
    within_cloud = min_distances <= threshold_within
    around_cloud = min_distances <= threshold_around
    
    # Compute distance to cloud center
    cloud_center = np.mean(core_mimic_features, axis=0)
    distances_to_center = np.sqrt(np.sum((scaled_data - cloud_center)**2, axis=1))
    
    results = {
        'within_cloud': within_cloud,
        'around_cloud': around_cloud,
        'min_distances': min_distances,
        'distances_to_center': distances_to_center,
        'cloud_center': cloud_center
    }
    
    return results


def define_mimic_cloud_spectral_clustering(scaled_data, data_frame, mimic_queries, n_clusters=None, gamma=None, output_file=None):
    """
    Define mimic cloud using spectral clustering.
    This identifies the cluster containing known mimics and uses it as the cloud.
    The number of clusters is automatically determined using the eigengap heuristic,
    which finds the largest gap in the graph Laplacian eigenvalues.
    
    Parameters:
    -----------
    scaled_data : array-like
        Scaled feature matrix (n_samples, n_features)
    data_frame : DataFrame
        DataFrame with 'query' column to identify mimics
    mimic_queries : list
        List of query IDs that are known mimics
    n_clusters : int, optional
        Number of clusters for spectral clustering. If None (recommended), will be 
        automatically determined using eigengap heuristic on the graph Laplacian.
    gamma : float, optional
        RBF kernel parameter. If None, will use median distance heuristic.
    output_file : str, optional
        Path to save cloud definition (JSON format)
    
    Returns:
    --------
    cloud_results : dict
        Dictionary containing:
        - 'within_cloud': boolean array indicating samples within mimic cluster
        - 'around_cloud': boolean array (same as within_cloud for spectral clustering)
        - 'cluster_labels': array of cluster assignments for all samples
        - 'mimic_cluster_id': ID of the cluster containing mimics
        - 'cluster_sizes': dict mapping cluster ID to size
        - 'mimic_queries': list of mimic query IDs found
    """
    import json
    from scipy.spatial.distance import cdist, pdist, squareform
    
    # Find mimic indices
    mimic_mask = data_frame['query'].isin(mimic_queries).values
    mimic_indices = np.where(mimic_mask)[0]
    
    if len(mimic_indices) == 0:
        print("Warning: No mimics found in dataset for spectral clustering")
        return None
    
    print(f"\nDefining mimic cloud using spectral clustering...")
    print(f"  Found {len(mimic_indices)} mimic alignments")
    
    n_samples = scaled_data.shape[0]
    
    # Estimate gamma (RBF kernel parameter) if not provided
    if gamma is None:
        # Use median pairwise distance as heuristic
        sample_indices = np.random.choice(n_samples, min(1000, n_samples), replace=False)
        sample_distances = pdist(scaled_data[sample_indices], metric='euclidean')
        median_dist = np.median(sample_distances)
        gamma = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0
        print(f"  Estimated gamma (RBF parameter): {gamma:.6f} (median distance: {median_dist:.6f})")
    else:
        print(f"  Using gamma: {gamma:.6f}")
    
    # Automatically determine number of clusters using eigengap heuristic
    if n_clusters is None:
        print("  Automatically determining optimal number of clusters using eigengap heuristic...")
        try:
            from sklearn.metrics.pairwise import rbf_kernel
            from scipy.sparse import csgraph
            from scipy.linalg import eigh
            
            # Build affinity matrix using RBF kernel
            # For large datasets, use a sample
            if n_samples > 5000:
                print(f"    Large dataset ({n_samples} samples), using subsample for eigengap analysis...")
                sample_size = 5000
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_data = scaled_data[sample_indices]
                affinity_matrix = rbf_kernel(sample_data, gamma=gamma)
            else:
                affinity_matrix = rbf_kernel(scaled_data, gamma=gamma)
            
            # Compute normalized Laplacian
            degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
            laplacian = degree_matrix - affinity_matrix
            # Normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(affinity_matrix, axis=1) + 1e-10))
            laplacian_norm = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
            
            # Compute eigenvalues (only need first k_max+1)
            k_max = min(20, n_samples - 1)  # Check up to 20 clusters
            eigenvalues, _ = eigh(laplacian_norm, subset_by_index=[0, k_max])
            eigenvalues = np.sort(eigenvalues)
            
            # Find eigengap (largest gap between consecutive eigenvalues)
            eigengaps = np.diff(eigenvalues[:k_max])
            optimal_k_idx = np.argmax(eigengaps)
            n_clusters = optimal_k_idx + 1  # +1 because gap is between eigenvalues
            
            # Ensure reasonable bounds
            n_clusters = max(2, min(n_clusters, min(20, n_samples // 10)))
            
            print(f"    Eigengap analysis: optimal k = {n_clusters}")
            print(f"    Top 5 eigenvalues: {eigenvalues[:5]}")
            print(f"    Largest eigengap: {eigengaps[optimal_k_idx]:.6f} (between λ{n_clusters-1} and λ{n_clusters})")
            
        except Exception as e:
            print(f"    Error in eigengap analysis: {e}")
            print(f"    Falling back to heuristic: sqrt(n_samples/2)")
            # Fallback heuristic
            n_clusters = max(2, min(20, int(np.sqrt(n_samples / 2))))
            print(f"    Using n_clusters: {n_clusters}")
    else:
        print(f"  Using specified n_clusters: {n_clusters}")
    
    # Perform spectral clustering
    print(f"  Performing spectral clustering with k={n_clusters}...")
    try:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='rbf',
            gamma=gamma,
            assign_labels='kmeans',
            random_state=42,
            n_jobs=-1
        )
        cluster_labels = spectral.fit_predict(scaled_data)
    except Exception as e:
        print(f"  Error in spectral clustering: {e}")
        print("  Falling back to distance-based method")
        return None
    
    # Identify which cluster contains the mimics
    mimic_cluster_ids = cluster_labels[mimic_indices]
    unique_mimic_clusters, counts = np.unique(mimic_cluster_ids, return_counts=True)
    
    # Find the cluster with the most mimics
    most_common_idx = np.argmax(counts)
    mimic_cluster_id = unique_mimic_clusters[most_common_idx]
    n_mimics_in_cluster = counts[most_common_idx]
    
    print(f"  Mimics found in {len(unique_mimic_clusters)} cluster(s)")
    print(f"  Primary mimic cluster: {mimic_cluster_id} ({n_mimics_in_cluster}/{len(mimic_indices)} mimics)")
    
    # If mimics are split across clusters, we can optionally merge them
    # For now, use the primary cluster
    if len(unique_mimic_clusters) > 1:
        print(f"  Note: {len(unique_mimic_clusters)} clusters contain mimics, using primary cluster")
        for cluster_id, count in zip(unique_mimic_clusters, counts):
            if cluster_id != mimic_cluster_id:
                print(f"    Cluster {cluster_id}: {count} mimics")
    
    # Define cloud membership
    within_cloud = cluster_labels == mimic_cluster_id
    around_cloud = within_cloud.copy()  # For spectral clustering, these are the same
    
    # Calculate cluster sizes
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, label_counts))
    
    print(f"  Cloud statistics:")
    print(f"    Samples in mimic cluster: {within_cloud.sum()}")
    print(f"    Cluster sizes: {cluster_sizes}")
    
    # Get mimic queries found
    mimic_queries_found = data_frame.iloc[mimic_indices]['query'].values.tolist()
    
    # Create results dictionary
    cloud_results = {
        'within_cloud': within_cloud,
        'around_cloud': around_cloud,
        'cluster_labels': cluster_labels,
        'mimic_cluster_id': int(mimic_cluster_id),
        'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
        'mimic_queries': mimic_queries_found,
        'n_clusters': n_clusters,
        'gamma': float(gamma),
        'n_mimics_in_cluster': int(n_mimics_in_cluster),
        'n_total_mimics': int(len(mimic_indices))
    }
    
    # Save to file if specified
    if output_file:
        cloud_results_for_json = cloud_results.copy()
        # Convert numpy arrays to lists for JSON
        cloud_results_for_json['within_cloud'] = within_cloud.tolist()
        cloud_results_for_json['around_cloud'] = around_cloud.tolist()
        cloud_results_for_json['cluster_labels'] = cluster_labels.tolist()
        with open(output_file, 'w') as f:
            json.dump(cloud_results_for_json, f, indent=2)
        print(f"  Saved spectral clustering cloud definition to: {output_file}")
    
    return cloud_results


def suggest_optimal_n_neighbors(scaled_data, n_samples=None):
    """
    Suggest optimal n_neighbors value for diffusion map based on heuristics and graph connectivity.
    
    Parameters:
    -----------
    scaled_data : array-like
        Scaled feature matrix (n_samples, n_features)
    n_samples : int, optional
        Number of samples (if None, will use scaled_data.shape[0])
    
    Returns:
    --------
    suggested_k : int
        Suggested n_neighbors value
    recommendations : dict
        Dictionary with heuristic values and explanations
    """
    if n_samples is None:
        n_samples = scaled_data.shape[0]
    
    # Heuristic 1: sqrt(n_samples) - common rule of thumb
    heuristic_sqrt = max(5, int(np.sqrt(n_samples)))
    
    # Heuristic 2: log(n_samples) - for larger datasets
    heuristic_log = max(5, int(np.log(n_samples) * 2))
    
    # Heuristic 3: n_samples / 10 - ensures connectivity but not too dense
    heuristic_frac = max(5, min(50, int(n_samples / 10)))
    
    # Heuristic 4: Fixed range based on dataset size
    if n_samples < 100:
        heuristic_size = max(3, min(10, n_samples // 5))
    elif n_samples < 1000:
        heuristic_size = 15
    elif n_samples < 5000:
        heuristic_size = 20
    else:
        heuristic_size = 30
    
    # Test graph connectivity for a few candidate values
    from sklearn.neighbors import kneighbors_graph
    
    candidates = sorted(set([heuristic_sqrt, heuristic_log, heuristic_frac, heuristic_size, 10, 15, 20, 30]))
    candidates = [c for c in candidates if 3 <= c < n_samples]
    
    connectivity_results = {}
    for k in candidates[:5]:  # Test up to 5 candidates
        try:
            knn_graph = kneighbors_graph(scaled_data, n_neighbors=k, mode='connectivity', 
                                        metric='euclidean', include_self=False)
            # Check if graph is connected (or has few components)
            from scipy.sparse.csgraph import connected_components
            n_components, _ = connected_components(csgraph=knn_graph, directed=False)
            
            # Calculate average degree
            degrees = np.array(knn_graph.sum(axis=1)).flatten()
            avg_degree = np.mean(degrees)
            
            connectivity_results[k] = {
                'n_components': n_components,
                'avg_degree': avg_degree,
                'is_well_connected': n_components <= max(1, n_samples // 100)  # Allow some small disconnected components
            }
        except:
            pass
    
    # Choose best candidate: well-connected with reasonable degree
    best_k = heuristic_size  # Default
    if connectivity_results:
        well_connected = {k: v for k, v in connectivity_results.items() 
                         if v['is_well_connected']}
        if well_connected:
            # Prefer moderate average degree (not too sparse, not too dense)
            best_k = min(well_connected.keys(), 
                        key=lambda k: abs(well_connected[k]['avg_degree'] - 15))
        else:
            # If none are well-connected, use the one with fewest components
            best_k = min(connectivity_results.keys(),
                        key=lambda k: connectivity_results[k]['n_components'])
    
    recommendations = {
        'suggested': best_k,
        'heuristics': {
            'sqrt(n)': heuristic_sqrt,
            'log(n)*2': heuristic_log,
            'n/10': heuristic_frac,
            'size_based': heuristic_size
        },
        'connectivity_analysis': connectivity_results,
        'n_samples': n_samples
    }
    
    return best_k, recommendations


def compute_diffusion_map_analysis(scaled_data, data_frame, pca_test_dir, organism_name, diffusion_alpha=1.0, diffusion_n_neighbors=15, plot_3d=False, use_natural_attractor=False, transfer_cloud_from=None, save_cloud_definition=False, use_adaptive_epsilon=False, adaptive_k_neighbors=7, auto_n_neighbors=False, use_spectral_cloud=False, spectral_n_clusters=None, spectral_gamma=None, scaler=None, epsilon_scale=1.0):
    """
    Compute diffusion map and diffusion pseudotime analysis.
    This is for trajectory analysis, not clustering.
    
    Parameters:
    -----------
    transfer_cloud_from : str, optional
        Path to JSON file containing mimic cloud definition from another dataset.
        If provided, this cloud will be applied in feature space and visualized.
    save_cloud_definition : bool, optional
        If True, save the mimic cloud definition to a JSON file for transfer to other datasets.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse import csr_matrix, diags
    from scipy.linalg import eigh
    from scipy.sparse.csgraph import shortest_path
    import os
    
    # Check for multi-system mode (joint diffusion embedding)
    is_multi_system = 'system' in data_frame.columns
    if is_multi_system:
        systems = data_frame['system'].unique()
        print(f"\n{'='*60}")
        print(f"Multi-system mode detected: {len(systems)} systems found")
        print(f"Systems: {', '.join(sorted(systems))}")
        print(f"System value counts:")
        print(data_frame['system'].value_counts())
        print(f"{'='*60}")
        print(f"Using JOINT diffusion embedding across all systems...")
        print(f"  This creates a unified manifold where mimic cloud regions are comparable across systems.")
    else:
        print(f"Using Diffusion Map for trajectory analysis (alpha={diffusion_alpha}, n_neighbors={diffusion_n_neighbors})...")
    
    # Auto-determine n_neighbors if requested
    if auto_n_neighbors:
        print("\n" + "="*60)
        print("Auto-determining optimal n_neighbors for diffusion map...")
        print("="*60)
        suggested_k, recommendations = suggest_optimal_n_neighbors(scaled_data)
        print(f"  Suggested n_neighbors: {suggested_k}")
        print(f"  Heuristic values:")
        for name, value in recommendations['heuristics'].items():
            print(f"    {name}: {value}")
        if recommendations['connectivity_analysis']:
            print(f"  Connectivity analysis:")
            for k, info in sorted(recommendations['connectivity_analysis'].items()):
                print(f"    k={k}: {info['n_components']} components, avg_degree={info['avg_degree']:.1f}, "
                      f"well_connected={info['is_well_connected']}")
        diffusion_n_neighbors = suggested_k
        print(f"  Using n_neighbors={diffusion_n_neighbors}")
        print("="*60 + "\n")
    
    # Create k-nearest neighbors graph with distances
    knn_graph = kneighbors_graph(scaled_data, n_neighbors=diffusion_n_neighbors, mode='distance', metric='euclidean', include_self=True)
    
    # Convert to symmetric affinity matrix (Gaussian kernel)
    # Use adaptive epsilon (bandwidth) if requested
    if use_adaptive_epsilon:
        print(f"  Using adaptive epsilon (local kernel width) with k={adaptive_k_neighbors} neighbors...")
        from sklearn.neighbors import NearestNeighbors
        
        # Compute local scales for each point based on k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=adaptive_k_neighbors + 1)  # +1 because includes self
        nn.fit(scaled_data)
        distances_nn, indices_nn = nn.kneighbors(scaled_data)
        
        # Local scale (epsilon) for each point: distance to k-th neighbor (excluding self)
        local_epsilons = distances_nn[:, adaptive_k_neighbors]  # k-th neighbor (0-indexed, excluding self)
        # Avoid zero scales
        local_epsilons = np.maximum(local_epsilons, np.percentile(local_epsilons, 1))
        
        # Apply scaling factor if provided
        if epsilon_scale != 1.0:
            local_epsilons = local_epsilons * epsilon_scale
            print(f"    Applied epsilon scaling factor: {epsilon_scale}")
        
        print(f"    Local epsilon statistics:")
        print(f"      Min: {np.min(local_epsilons):.6f}")
        print(f"      Median: {np.median(local_epsilons):.6f}")
        print(f"      Max: {np.max(local_epsilons):.6f}")
        
        # Create symmetric affinity matrix with adaptive bandwidths
        knn_graph_sym = 0.5 * (knn_graph + knn_graph.T)  # Make symmetric
        
        # For each edge (i, j), use max of local epsilons (conservative approach)
        n_samples = scaled_data.shape[0]
        affinities_list = []
        indices_list = []
        indptr_list = [0]
        
        for i in range(n_samples):
            row_start = knn_graph_sym.indptr[i]
            row_end = knn_graph_sym.indptr[i + 1]
            row_indices = knn_graph_sym.indices[row_start:row_end]
            row_distances = knn_graph_sym.data[row_start:row_end]
            
            row_affinities = []
            row_indices_out = []
            
            for j_idx, j in enumerate(row_indices):
                if i == j:
                    # Self-affinity is 1
                    row_affinities.append(1.0)
                    row_indices_out.append(j)
                else:
                    # Use max of local epsilons for this pair (conservative approach)
                    # For tighter clustering, could use min or mean instead
                    epsilon_ij = max(local_epsilons[i], local_epsilons[j])
                    # Optional: scale down epsilon for tighter clustering (uncomment if needed)
                    # epsilon_ij = epsilon_ij * 0.7  # 30% reduction for tighter clusters
                    dist_ij = row_distances[j_idx]
                    affinity_ij = np.exp(-dist_ij**2 / (2 * epsilon_ij**2))
                    row_affinities.append(affinity_ij)
                    row_indices_out.append(j)
            
            affinities_list.extend(row_affinities)
            indices_list.extend(row_indices_out)
            indptr_list.append(len(affinities_list))
        
        affinity_matrix = csr_matrix((affinities_list, indices_list, indptr_list), shape=(n_samples, n_samples))
    else:
        # Fixed global bandwidth (original approach)
        distances = knn_graph.data
        if len(distances) > 0:
            bandwidth = np.median(distances[distances > 0])
        else:
            bandwidth = 1.0
        print(f"  Using fixed epsilon (bandwidth): {bandwidth:.6f}")
        
        # Create Gaussian affinity matrix
        knn_graph_sym = 0.5 * (knn_graph + knn_graph.T)  # Make symmetric
        distances_sym = knn_graph_sym.data
        affinities = np.exp(-distances_sym**2 / (2 * bandwidth**2))
        affinity_matrix = csr_matrix((affinities, knn_graph_sym.indices, knn_graph_sym.indptr), shape=knn_graph_sym.shape)
    
    # Compute row sums for normalization
    row_sums = np.array(affinity_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    
    # Apply alpha parameter
    if diffusion_alpha != 1.0:
        row_sums_alpha = np.power(row_sums, -diffusion_alpha)
        D_alpha_inv = diags(row_sums_alpha, format='csr')
        affinity_matrix = D_alpha_inv.dot(affinity_matrix).dot(D_alpha_inv)
        row_sums = np.array(affinity_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
    
    # Create Markov transition matrix
    D_inv = diags(1.0 / row_sums, format='csr')
    transition_matrix = D_inv.dot(affinity_matrix)
    
    # Compute diffusion map coordinates
    n_samples = transition_matrix.shape[0]
    n_diffusion_components = min(10, scaled_data.shape[1] - 1, n_samples - 1)
    
    if n_samples < 5000:
        transition_dense = transition_matrix.toarray()
        eigenvals, eigenvecs = eigh(transition_dense)
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        diffusion_coords = eigenvecs[:, 1:n_diffusion_components+1]
        eigenvalues = eigenvals[1:n_diffusion_components+1]
    else:
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components=n_diffusion_components, 
                                     affinity='precomputed',
                                     random_state=42)
        diffusion_coords = embedding.fit_transform(transition_matrix)
        eigenvalues = None  # Not available from SpectralEmbedding
    
    print(f"Diffusion map computed. Using {n_diffusion_components} components.")
    
    # Analyze eigenvalues to determine intrinsic dimensionality
    print("\n" + "="*60)
    print("Eigenvalue Analysis (Intrinsic Dimensionality Check)")
    print("="*60)
    if eigenvalues is not None:
        print(f"Eigenvalues: {eigenvalues[:min(10, len(eigenvalues))]}")  # Show first 10
        
        # Check eigenvalue ratios to determine structure
        if len(eigenvalues) >= 3:
            lambda1 = eigenvalues[0]
            lambda2 = eigenvalues[1]
            lambda3 = eigenvalues[2]
            
            ratio_1_2 = lambda1 / lambda2 if lambda2 > 0 else np.inf
            ratio_2_3 = lambda2 / lambda3 if lambda3 > 0 else np.inf
            
            print(f"\nEigenvalue Ratios:")
            print(f"  λ₁/λ₂ = {ratio_1_2:.4f}")
            print(f"  λ₂/λ₃ = {ratio_2_3:.4f}")
            
            if len(eigenvalues) >= 4:
                lambda4 = eigenvalues[3]
                ratio_3_4 = lambda3 / lambda4 if lambda4 > 0 else np.inf
                print(f"  λ₃/λ₄ = {ratio_3_4:.4f}")
                
                # More detailed analysis for DC3 and DC4
                # Check how much information DC3 and DC4 contain relative to DC1 and DC2
                lambda_sum_12 = lambda1 + lambda2
                lambda_sum_34 = lambda3 + lambda4
                ratio_12_34 = lambda_sum_12 / lambda_sum_34 if lambda_sum_34 > 0 else np.inf
                
                # Check individual contributions
                lambda3_contribution = lambda3 / lambda1 if lambda1 > 0 else 0
                lambda4_contribution = lambda4 / lambda1 if lambda1 > 0 else 0
                
                print(f"\nDC3 and DC4 Information Content:")
                print(f"  λ₃ contribution (relative to λ₁): {lambda3_contribution:.4f} ({lambda3_contribution*100:.2f}%)")
                print(f"  λ₄ contribution (relative to λ₁): {lambda4_contribution:.4f} ({lambda4_contribution*100:.2f}%)")
                print(f"  (λ₁+λ₂)/(λ₃+λ₄) ratio: {ratio_12_34:.4f}")
                
                if ratio_3_4 > 3.0:
                    print(f"  ✓ DC3 contains significant signal (λ₃/λ₄ > 3)")
                    print(f"    → DC3 likely captures meaningful structure")
                elif ratio_3_4 > 2.0:
                    print(f"  → DC3 may contain signal (λ₃/λ₄ > 2)")
                    print(f"    → DC3 worth investigating")
                elif ratio_3_4 > 1.5:
                    print(f"  → DC3 and DC4 have similar magnitude (λ₃/λ₄ ≈ {ratio_3_4:.2f})")
                    print(f"    → Both may contain some signal or both may be noise")
                else:
                    print(f"  → DC3 and DC4 likely noise (λ₃/λ₄ ≈ 1)")
                    print(f"    → DC4 may be capturing noise floor")
                
                # Check if DC3+DC4 together contain substantial information
                if lambda3_contribution > 0.1:  # DC3 is >10% of DC1
                    print(f"  ✓ DC3 has substantial information ({lambda3_contribution*100:.1f}% of DC1)")
                if lambda4_contribution > 0.05:  # DC4 is >5% of DC1
                    print(f"  → DC4 has some information ({lambda4_contribution*100:.1f}% of DC1)")
                
                if ratio_12_34 < 3.0:
                    print(f"  ⚠ DC3+DC4 together contain significant information")
                    print(f"    → Consider 3D or 4D analysis (DC1-DC2-DC3 or DC1-DC2-DC3-DC4)")
                elif ratio_12_34 < 5.0:
                    print(f"  → DC3+DC4 may contain some information")
                    print(f"    → Worth checking 3D visualizations")
                else:
                    print(f"  → DC3+DC4 contain minimal information relative to DC1+DC2")
            
            print(f"\nStructure Interpretation:")
            if ratio_1_2 > 5 and ratio_2_3 > 5:
                print(f"  ✓ Strong 1D structure detected (λ₁ ≫ λ₂ ≫ λ₃)")
                print(f"    → Intrinsic dimensionality is ~1")
                print(f"    → DC1 captures the main trajectory")
                print(f"    → DC2 and higher likely represent noise correction")
            elif ratio_1_2 > 2 and ratio_2_3 > 2:
                print(f"  → Moderate 1D structure (λ₁ > λ₂ > λ₃)")
                print(f"    → DC1 is primary, but DC2 may contain signal")
            elif ratio_1_2 < 1.5:
                print(f"  ⚠ Warning: λ₂ ≈ λ₁ (ratio = {ratio_1_2:.4f})")
                print(f"    → Structure may have curvature or branching")
                print(f"    → DC2 may contain important information")
                print(f"    → Consider 2D analysis")
            else:
                print(f"  → Mixed structure: DC1 dominant but DC2 may be informative")
        else:
            print("  (Not enough eigenvalues for ratio analysis)")
    else:
        print("  (Eigenvalues not available from SpectralEmbedding approximation)")
        print("  (Consider using smaller dataset for exact eigenvalue computation)")
    
    # Use DC1 as the primary pseudotime/progression axis
    # Root point selection: prefer densest region (attractor) over minimum DC1
    # This is more robust when many points cluster together (e.g., all alignments)
    
    # Compute local density to find attractor
    from sklearn.neighbors import NearestNeighbors
    n_neighbors_density = min(50, len(diffusion_coords) // 10)
    if n_neighbors_density < 5:
        n_neighbors_density = min(5, len(diffusion_coords) - 1)
    
    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors_density + 1)
        nn.fit(diffusion_coords[:, :2])  # Use DC1 and DC2 for density
        distances, _ = nn.kneighbors(diffusion_coords[:, :2])
        # Local density = inverse of mean distance to neighbors
        local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
        attractor_idx = np.argmax(local_density)
        root_idx = attractor_idx
        print(f"\nRoot point (attractor/densest region): index {root_idx}")
        print(f"  DC1: {diffusion_coords[root_idx, 0]:.4f}, DC2: {diffusion_coords[root_idx, 1]:.4f}")
        print(f"  Local density: {local_density[root_idx]:.4f}")
    except:
        # Fallback to minimum DC1 if density calculation fails
        root_idx = np.argmin(diffusion_coords[:, 0])
        print(f"\nRoot point (minimum DC1, fallback): index {root_idx}")
    
    # Check DC1 variance to diagnose pseudotime issues
    dc1_values = diffusion_coords[:, 0]
    dc1_range = dc1_values.max() - dc1_values.min()
    dc1_std = dc1_values.std()
    print(f"DC1 statistics:")
    print(f"  Range: {dc1_range:.6f}")
    print(f"  Std dev: {dc1_std:.6f}")
    print(f"  Min: {dc1_values.min():.6f}, Max: {dc1_values.max():.6f}")
    
    # DC1-based pseudotime (normalized to start at 0)
    dc1_pseudotime = diffusion_coords[:, 0] - diffusion_coords[root_idx, 0]
    dc1_pseudotime = dc1_pseudotime - dc1_pseudotime.min()  # Normalize to start at 0
    
    # Check pseudotime variance
    pseudotime_range = dc1_pseudotime.max() - dc1_pseudotime.min()
    pseudotime_std = dc1_pseudotime.std()
    print(f"DC1 pseudotime statistics:")
    print(f"  Range: {pseudotime_range:.6f}")
    print(f"  Std dev: {pseudotime_std:.6f}")
    
    if pseudotime_range < 1e-6:
        print("  WARNING: DC1 pseudotime has near-zero range - all points are essentially identical!")
        print("  This suggests DC1 has very low variance, possibly due to:")
        print("    - Many duplicate or near-duplicate alignments")
        print("    - Tight clustering in diffusion space")
        print("  Consider using geodesic distances instead for better resolution.")
    
    # Also compute traditional diffusion pseudotime for comparison
    diffusion_pseudotime = dc1_pseudotime.copy()
    
    # Alternative: compute shortest path distances on the graph
    # Convert affinity to distance (higher affinity = lower distance)
    distance_graph = affinity_matrix.copy()
    distance_graph.data = -np.log(distance_graph.data + 1e-10)  # Avoid log(0)
    distance_graph.data[distance_graph.data < 0] = 0  # Remove negative values
    
    # Compute shortest paths from root
    geodesic_distances = None
    try:
        geodesic_distances = shortest_path(distance_graph, 
                                          directed=False, 
                                          indices=root_idx, 
                                          method='auto')
        geodesic_distances = geodesic_distances.flatten()
        geodesic_distances[np.isinf(geodesic_distances)] = np.nan
        
        # Check geodesic distance statistics
        valid_geodesic = geodesic_distances[~np.isnan(geodesic_distances)]
        if len(valid_geodesic) > 0:
            geodesic_range = valid_geodesic.max() - valid_geodesic.min()
            geodesic_std = valid_geodesic.std()
            print(f"Geodesic distance statistics:")
            print(f"  Range: {geodesic_range:.6f}")
            print(f"  Std dev: {geodesic_std:.6f}")
            print(f"  Valid distances: {len(valid_geodesic)}/{len(geodesic_distances)}")
        
        print(f"Computed geodesic distances from root point")
    except Exception as e:
        print(f"Warning: Could not compute geodesic distances: {e}")
        print("  Using diffusion coordinate ordering")
    
    # Use geodesic distances as primary pseudotime if DC1 variance is too low
    # or if geodesic distances have better resolution
    use_geodesic_as_primary = False
    if geodesic_distances is not None:
        valid_geodesic = geodesic_distances[~np.isnan(geodesic_distances)]
        if len(valid_geodesic) > 0:
            geodesic_range = valid_geodesic.max() - valid_geodesic.min()
            # Use geodesic if DC1 range is very small OR if geodesic has better range
            if pseudotime_range < 1e-4 or (geodesic_range > 0 and geodesic_range > pseudotime_range * 2):
                use_geodesic_as_primary = True
                print(f"\nUsing geodesic distances as primary pseudotime (better resolution)")
                # Normalize geodesic distances to start at 0
                diffusion_pseudotime = geodesic_distances.copy()
                diffusion_pseudotime[np.isnan(diffusion_pseudotime)] = np.nanmax(geodesic_distances) + 1
                diffusion_pseudotime = diffusion_pseudotime - diffusion_pseudotime.min()
            else:
                print(f"\nUsing DC1-based pseudotime (sufficient resolution)")
    
    # Create results DataFrame - reset index to ensure alignment with diffusion_coords
    results_df = data_frame.reset_index(drop=True).copy()
    # Save up to DC4 (or as many as available)
    for i in range(min(4, n_diffusion_components)):
        results_df[f'Diffusion{i+1}'] = diffusion_coords[:, i]
    results_df['dc1_pseudotime'] = dc1_pseudotime
    results_df['diffusion_pseudotime'] = diffusion_pseudotime
    if geodesic_distances is not None:
        results_df['geodesic_distance'] = geodesic_distances
        if use_geodesic_as_primary:
            results_df['trajectory_order'] = np.argsort(geodesic_distances)
        else:
            results_df['trajectory_order'] = np.argsort(dc1_pseudotime)
    else:
        results_df['trajectory_order'] = np.argsort(dc1_pseudotime)
    
    # Feature-space cloud definition and transfer
    transferred_cloud_results = None
    local_cloud_definition = None
    spectral_cloud_results = None
    
    # Define spectral clustering cloud if requested (runs alongside distance-based cloud)
    if use_spectral_cloud:
        print("\n" + "="*60)
        print("Defining mimic cloud using spectral clustering (parallel to distance-based method)")
        print("="*60)
        spectral_output_file = os.path.join(pca_test_dir, f'{organism_name}_spectral_cloud_definition.json')
        spectral_cloud_results = define_mimic_cloud_spectral_clustering(
            scaled_data, data_frame, validation_list,
            n_clusters=spectral_n_clusters,
            gamma=spectral_gamma,
            output_file=spectral_output_file
        )
        if spectral_cloud_results:
            # Add spectral cloud results to DataFrame
            results_df['spectral_within_cloud'] = spectral_cloud_results['within_cloud']
            results_df['spectral_around_cloud'] = spectral_cloud_results['around_cloud']
            results_df['spectral_cluster_label'] = spectral_cloud_results['cluster_labels']
            
            # Calculate and store mimic distances from cluster centroid for analysis
            if 'query' in results_df.columns:
                mimic_mask = results_df['query'].isin(validation_list).values
                mimic_indices = np.where(mimic_mask)[0]
                if len(mimic_indices) > 0:
                    dc_cols = [col for col in results_df.columns if col.startswith('Diffusion')]
                    if len(dc_cols) >= 3:
                        all_coords_3d = results_df[dc_cols[:3]].values
                        mimic_coords_3d = all_coords_3d[mimic_indices]
                        mimic_center = np.mean(mimic_coords_3d, axis=0)
                        mimic_distances = np.linalg.norm(mimic_coords_3d - mimic_center, axis=1)
                        
                        # Initialize distance column for all samples
                        results_df['spectral_mimic_distance'] = np.nan
                        results_df.loc[mimic_mask, 'spectral_mimic_distance'] = mimic_distances
                        
                        # Classify mimics
                        median_dist = np.median(mimic_distances)
                        q75_dist = np.percentile(mimic_distances, 75)
                        results_df['spectral_mimic_type'] = 'non_mimic'
                        results_df.loc[mimic_mask & (results_df['spectral_mimic_distance'] <= median_dist), 'spectral_mimic_type'] = 'core'
                        results_df.loc[mimic_mask & (results_df['spectral_mimic_distance'] > q75_dist), 'spectral_mimic_type'] = 'outlier'
                        results_df.loc[mimic_mask & (results_df['spectral_mimic_type'] == 'non_mimic'), 'spectral_mimic_type'] = 'intermediate'
            
            print(f"  Applied spectral clustering cloud to current dataset:")
            print(f"    Within cloud: {spectral_cloud_results['within_cloud'].sum()} samples")
            print(f"    Mimic cluster ID: {spectral_cloud_results['mimic_cluster_id']}")
            print(f"    Total clusters: {spectral_cloud_results['n_clusters']}")
    
    # Define and save cloud in feature space if requested
    if save_cloud_definition:
        print("\n" + "="*60)
        print("Defining mimic cloud in feature space (for transfer to other datasets)")
        print("="*60)
        cloud_output_file = os.path.join(pca_test_dir, f'{organism_name}_mimic_cloud_definition.json')
        # Get feature columns from data_frame (exclude non-feature columns)
        exclude_cols = ['query', 'target', 'system', 'label', 'cluster', 'go_term']
        feature_columns = [col for col in data_frame.columns 
                          if col not in exclude_cols and 
                          data_frame[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        # Match to scaled_data dimensions
        if len(feature_columns) == scaled_data.shape[1]:
            local_cloud_definition = define_mimic_cloud_in_feature_space(
                scaled_data, data_frame, validation_list, output_file=cloud_output_file,
                feature_columns=feature_columns
            )
        else:
            local_cloud_definition = define_mimic_cloud_in_feature_space(
                scaled_data, data_frame, validation_list, output_file=cloud_output_file
            )
        if local_cloud_definition:
            # Apply to current dataset and add to results
            local_cloud_results = apply_mimic_cloud_to_dataset(scaled_data, local_cloud_definition)
            results_df['feature_space_within_cloud'] = local_cloud_results['within_cloud']
            results_df['feature_space_around_cloud'] = local_cloud_results['around_cloud']
            results_df['feature_space_min_dist_to_mimic'] = local_cloud_results['min_distances']
            results_df['feature_space_dist_to_cloud_center'] = local_cloud_results['distances_to_center']
            print(f"  Applied feature-space cloud to current dataset:")
            print(f"    Within cloud: {local_cloud_results['within_cloud'].sum()} samples")
            print(f"    Around cloud: {local_cloud_results['around_cloud'].sum()} samples")
            
            # Report feature statistics for local cloud
            local_stats_file = os.path.join(pca_test_dir, f'{organism_name}_local_cloud_feature_statistics.txt')
            report_cloud_feature_statistics(data_frame, local_cloud_results, cloud_name="local cloud", output_file=local_stats_file)
    
    # Load and apply transferred cloud if provided
    if transfer_cloud_from:
        print("\n" + "="*60)
        print(f"Applying transferred mimic cloud from: {transfer_cloud_from}")
        print("="*60)
        try:
            transferred_cloud_definition = load_mimic_cloud_definition(transfer_cloud_from)
            transferred_cloud_results = apply_mimic_cloud_to_dataset(
                scaled_data, transferred_cloud_definition, 
                data_frame=data_frame, scaler=scaler
            )
            results_df['transferred_within_cloud'] = transferred_cloud_results['within_cloud']
            results_df['transferred_around_cloud'] = transferred_cloud_results['around_cloud']
            results_df['transferred_min_dist_to_mimic'] = transferred_cloud_results['min_distances']
            results_df['transferred_dist_to_cloud_center'] = transferred_cloud_results['distances_to_center']
            print(f"  Transferred cloud applied successfully:")
            print(f"    Within cloud: {transferred_cloud_results['within_cloud'].sum()} samples")
            print(f"    Around cloud: {transferred_cloud_results['around_cloud'].sum()} samples")
            print(f"    Cloud from: {len(transferred_cloud_definition['core_mimic_queries'])} core mimics")
            
            # Report feature statistics for transferred cloud
            transferred_stats_file = os.path.join(pca_test_dir, f'{organism_name}_transferred_cloud_feature_statistics.txt')
            report_cloud_feature_statistics(data_frame, transferred_cloud_results, cloud_name="transferred cloud", output_file=transferred_stats_file)
        except Exception as e:
            print(f"  Error loading/applying transferred cloud: {e}")
            import traceback
            traceback.print_exc()
    
    # Identify best alignments for known mimics and candidates
    alignment_quality = (data_frame['score'].values * 
                        data_frame['fident'].values * 
                        data_frame['algn_fraction'].values)
    
    # Find best mimic alignments (one per query)
    mimic_indices = []
    mimic_mask = data_frame['query'].isin(validation_list).values
    if mimic_mask.sum() > 0:
        mimic_df = pd.DataFrame({
            'query': data_frame['query'].values,
            'quality': alignment_quality,
            'row_idx': range(len(data_frame))
        })
        mimic_df = mimic_df[mimic_mask]
        best_mimic_indices = mimic_df.groupby('query')['quality'].idxmax().values
        # Map back to original DataFrame indices using row_idx
        mimic_indices = mimic_df.loc[best_mimic_indices, 'row_idx'].values.tolist()
        print(f"Found {len(mimic_indices)} best mimic alignments to highlight")
    
    # Find best candidate alignments (one per query)
    candidate_indices = []
    candidate_mask = data_frame['query'].isin(candidate_list).values
    if candidate_mask.sum() > 0:
        candidate_df = pd.DataFrame({
            'query': data_frame['query'].values,
            'quality': alignment_quality,
            'row_idx': range(len(data_frame))
        })
        candidate_df = candidate_df[candidate_mask]
        best_candidate_indices = candidate_df.groupby('query')['quality'].idxmax().values
        # Map back to original DataFrame indices using row_idx
        candidate_indices = candidate_df.loc[best_candidate_indices, 'row_idx'].values.tolist()
        print(f"Found {len(candidate_indices)} best candidate alignments to highlight")
    
    # Save results
    # results_df already has all columns from data_frame.copy() above (with reset index)
    # Include query column and GO term columns if available for downstream analysis
    if 'query' in data_frame.columns and 'query' not in results_df.columns:
        results_df['query'] = data_frame['query'].values
    # Include system column if present (for multi-system mode)
    if 'system' in data_frame.columns and 'system' not in results_df.columns:
        results_df['system'] = data_frame['system'].values
    # Include GO term columns if they exist
    go_term_cols = ['target_biological_processes', 'target_molecular_functions', 'target_cellular_components']
    for col in go_term_cols:
        if col in data_frame.columns:
            results_df[col] = data_frame[col].values
    
    output_file = os.path.join(pca_test_dir, f'{organism_name}_diffusion_map_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Saved diffusion map results to {output_file}")
    
    # Create visualizations
    # 2D plot: Diffusion1 vs Diffusion2, colored by pseudotime
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot all points colored by pseudotime
    scatter = ax.scatter(diffusion_coords[:, 0], diffusion_coords[:, 1], 
                        c=diffusion_pseudotime, cmap='viridis', s=20, alpha=0.5, zorder=1)
    
    # Highlight root point
    ax.scatter(diffusion_coords[root_idx, 0], diffusion_coords[root_idx, 1], 
              c='red', s=150, marker='*', label='Root', zorder=5, edgecolors='black', linewidths=1)
    
    # Highlight mimic points
    if len(mimic_indices) > 0:
        mimic_coords = diffusion_coords[mimic_indices, :]
        ax.scatter(mimic_coords[:, 0], mimic_coords[:, 1],
                  c='lime', s=200, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                  zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
        # Add text labels for mimics
        for idx in mimic_indices:
            query_id = data_frame.iloc[idx]['query']
            ax.annotate(query_id, 
                       (diffusion_coords[idx, 0], diffusion_coords[idx, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='lime',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='lime'),
                       zorder=6)
    
    # Highlight candidate points
    if len(candidate_indices) > 0:
        candidate_coords = diffusion_coords[candidate_indices, :]
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                  c='cyan', s=200, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                  zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
        # Add text labels for candidates
        for idx in candidate_indices:
            query_id = data_frame.iloc[idx]['query']
            ax.annotate(query_id, 
                       (diffusion_coords[idx, 0], diffusion_coords[idx, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='cyan',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='cyan'),
                       zorder=6)
    
    # Set robust axis limits to handle outliers (use 1st-99th percentile)
    xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
    ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 1], percentile_low=1.0, percentile_high=99.0)
    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(ylim_low, ylim_high)
    
    ax.set_xlabel('Diffusion Coordinate 1')
    ax.set_ylabel('Diffusion Coordinate 2')
    ax.set_title(f'Diffusion Map - {organism_name.capitalize()}\n(Colored by Pseudotime, Mimics/Candidates Labeled)\n(Axis limits: 1st-99th percentile)')
    plt.colorbar(scatter, ax=ax, label='Diffusion Pseudotime')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_2d.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()
    
    if plot_3d:
        # Separate 3D plot with multiple viewing angles
        from mpl_toolkits.mplot3d import Axes3D
        
        # Define multiple viewing angles: (elevation, azimuth, title_suffix)
        view_angles = [
            (30, 45, 'standard'),
            (0, 0, 'xy_plane'),      # Top-down view (looking down Z-axis)
            (90, 0, 'xz_plane'),     # Side view (looking along Y-axis)
            (0, 90, 'yz_plane'),     # Side view (looking along X-axis)
            (30, 135, 'diagonal1'),
            (30, 225, 'diagonal2'),
            (60, 45, 'elevated'),
        ]
        
        for elev, azim, suffix in view_angles:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all points colored by pseudotime
            scatter = ax.scatter(diffusion_coords[:, 0], diffusion_coords[:, 1], diffusion_coords[:, 2],
                               c=diffusion_pseudotime, cmap='viridis', s=20, alpha=0.5)
            
            # Root point
            ax.scatter([diffusion_coords[root_idx, 0]], [diffusion_coords[root_idx, 1]], [diffusion_coords[root_idx, 2]],
                      c='red', s=200, marker='*', label='Root')
            
            # Mimic points
            if len(mimic_indices) > 0:
                mimic_coords = diffusion_coords[mimic_indices, :]
                ax.scatter(mimic_coords[:, 0], mimic_coords[:, 1], mimic_coords[:, 2],
                          c='lime', s=300, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                          edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # Candidate points
            if len(candidate_indices) > 0:
                candidate_coords = diffusion_coords[candidate_indices, :]
                ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2],
                          c='cyan', s=300, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                          edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # Set robust axis limits for 3D plot
            xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
            ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 1], percentile_low=1.0, percentile_high=99.0)
            zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
            ax.set_xlim(xlim_low, xlim_high)
            ax.set_ylim(ylim_low, ylim_high)
            ax.set_zlim(zlim_low, zlim_high)
            
            ax.set_xlabel('Diffusion Coordinate 1')
            ax.set_ylabel('Diffusion Coordinate 2')
            ax.set_zlabel('Diffusion Coordinate 3')
            ax.set_title(f'Diffusion Map 3D - {organism_name.capitalize()}\n(Colored by Pseudotime, View: elev={elev}°, azim={azim}°)\n(Axis limits: 1st-99th percentile)')
            ax.view_init(elev=elev, azim=azim)
            plt.colorbar(scatter, ax=ax, label='Diffusion Pseudotime')
            ax.legend(loc='best')
            plot_file_3d = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_{suffix}.png')
            plt.savefig(plot_file_3d, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot ({suffix}) to {plot_file_3d}")
            plt.close()
    
    # ============================================================
    # 3D Visualizations with Transferred Cloud Membership Labels
    # ============================================================
    if plot_3d and transferred_cloud_results is not None and n_diffusion_components >= 3:
        print("\n" + "="*60)
        print("Creating 3D visualizations with transferred cloud membership labels")
        print("="*60)
        
        transferred_within = transferred_cloud_results['within_cloud']
        transferred_around = transferred_cloud_results['around_cloud']
        transferred_outside = ~transferred_around
        
        # Define multiple viewing angles
        view_angles = [
            (30, 45, 'standard'),
            (0, 0, 'xy_plane'),
            (90, 0, 'xz_plane'),
            (0, 90, 'yz_plane'),
            (30, 135, 'diagonal1'),
            (30, 225, 'diagonal2'),
            (60, 45, 'elevated'),
        ]
        
        for elev, azim, suffix in view_angles:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points outside cloud (gray, small, low alpha)
            if transferred_outside.sum() > 0:
                ax.scatter(diffusion_coords[transferred_outside, 0], 
                          diffusion_coords[transferred_outside, 1], 
                          diffusion_coords[transferred_outside, 2],
                          c='lightgray', s=15, alpha=0.2, 
                          label=f'Outside cloud ({transferred_outside.sum()})', zorder=1)
            
            # Plot points around cloud but not within (orange)
            transferred_around_not_within = transferred_around & ~transferred_within
            if transferred_around_not_within.sum() > 0:
                ax.scatter(diffusion_coords[transferred_around_not_within, 0], 
                          diffusion_coords[transferred_around_not_within, 1], 
                          diffusion_coords[transferred_around_not_within, 2],
                          c='orange', s=40, alpha=0.6, 
                          label=f'Around cloud ({transferred_around_not_within.sum()})', zorder=2)
            
            # Plot points within cloud (purple/blue, larger, higher alpha)
            if transferred_within.sum() > 0:
                ax.scatter(diffusion_coords[transferred_within, 0], 
                          diffusion_coords[transferred_within, 1], 
                          diffusion_coords[transferred_within, 2],
                          c='purple', s=60, alpha=0.8, 
                          label=f'Within cloud ({transferred_within.sum()})', zorder=3)
            
            # Highlight root point
            ax.scatter([diffusion_coords[root_idx, 0]], 
                      [diffusion_coords[root_idx, 1]], 
                      [diffusion_coords[root_idx, 2]],
                      c='red', s=300, marker='*', label='Root', 
                      zorder=5, edgecolors='black', linewidths=1.5)
            
            # Highlight mimic points if available
            if len(mimic_indices) > 0:
                mimic_coords = diffusion_coords[mimic_indices, :]
                ax.scatter(mimic_coords[:, 0], mimic_coords[:, 1], mimic_coords[:, 2],
                          c='lime', s=400, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                          zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
            
            # Highlight candidate points if available
            if len(candidate_indices) > 0:
                candidate_coords = diffusion_coords[candidate_indices, :]
                ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2],
                          c='cyan', s=400, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                          zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
            
            # Set robust axis limits
            xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
            ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 1], percentile_low=1.0, percentile_high=99.0)
            zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
            ax.set_xlim(xlim_low, xlim_high)
            ax.set_ylim(ylim_low, ylim_high)
            ax.set_zlim(zlim_low, zlim_high)
            
            ax.set_xlabel('Diffusion Coordinate 1', fontsize=12)
            ax.set_ylabel('Diffusion Coordinate 2', fontsize=12)
            ax.set_zlabel('Diffusion Coordinate 3', fontsize=12)
            ax.set_title(f'Diffusion Map 3D - Transferred Cloud Membership\n{organism_name.capitalize()} (View: elev={elev}°, azim={azim}°)\n'
                        f'Within: {transferred_within.sum()} | Around: {transferred_around_not_within.sum()} | Outside: {transferred_outside.sum()}', 
                        fontsize=11)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
            ax.grid(True, alpha=0.3)
            
            plot_file_3d_cloud = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_cloud_membership_{suffix}.png')
            plt.savefig(plot_file_3d_cloud, dpi=300, bbox_inches='tight')
            print(f"Saved 3D cloud membership plot ({suffix}) to {plot_file_3d_cloud}")
            plt.close()
    
    # ============================================================
    # DC3 and DC4 Visualizations
    # ============================================================
    if n_diffusion_components >= 3:
        print("\n" + "="*60)
        print("DC3 and DC4 Visualizations")
        print("="*60)
        
        # Create 2D plots for DC3 and DC4
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_configs = [
            (0, 2, 'DC1 vs DC3', 'Diffusion Coordinate 1', 'Diffusion Coordinate 3'),
            (1, 2, 'DC2 vs DC3', 'Diffusion Coordinate 2', 'Diffusion Coordinate 3'),
        ]
        
        if n_diffusion_components >= 4:
            plot_configs.extend([
                (0, 3, 'DC1 vs DC4', 'Diffusion Coordinate 1', 'Diffusion Coordinate 4'),
                (1, 3, 'DC2 vs DC4', 'Diffusion Coordinate 2', 'Diffusion Coordinate 4'),
                (2, 3, 'DC3 vs DC4', 'Diffusion Coordinate 3', 'Diffusion Coordinate 4'),
            ])
        else:
            # If no DC4, just show DC3 plots
            plot_configs.append((None, None, None, None, None))
        
        plot_idx = 0
        for i, j, title, xlabel, ylabel in plot_configs:
            if i is None:
                axes[plot_idx].axis('off')
                plot_idx += 1
                continue
                
            ax = axes[plot_idx]
            
            # Plot all points colored by pseudotime
            scatter = ax.scatter(diffusion_coords[:, i], diffusion_coords[:, j], 
                               c=diffusion_pseudotime, cmap='viridis', s=20, alpha=0.5, zorder=1)
            
            # Highlight root point
            ax.scatter(diffusion_coords[root_idx, i], diffusion_coords[root_idx, j],
                      c='red', s=150, marker='*', label='Root', zorder=5, 
                      edgecolors='black', linewidths=1)
            
            # Highlight mimic points
            if len(mimic_indices) > 0:
                mimic_coords = diffusion_coords[mimic_indices, :]
                ax.scatter(mimic_coords[:, i], mimic_coords[:, j],
                          c='lime', s=200, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                          zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # Highlight candidate points
            if len(candidate_indices) > 0:
                candidate_coords = diffusion_coords[candidate_indices, :]
                ax.scatter(candidate_coords[:, i], candidate_coords[:, j],
                          c='cyan', s=200, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                          zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # Set robust axis limits
            xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, i], percentile_low=1.0, percentile_high=99.0)
            ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, j], percentile_low=1.0, percentile_high=99.0)
            ax.set_xlim(xlim_low, xlim_high)
            ax.set_ylim(ylim_low, ylim_high)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.colorbar(scatter, ax=ax, label='Diffusion Pseudotime')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_file_dc34 = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_dc3_dc4.png')
        plt.savefig(plot_file_dc34, dpi=300, bbox_inches='tight')
        print(f"Saved DC3/DC4 plots to {plot_file_dc34}")
        plt.close()
        
        # 3D plots with DC3 and DC4 (multiple angles)
        if n_diffusion_components >= 4:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Define viewing angles for DC3/DC4 plots
            view_angles = [
                (30, 45, 'standard'),
                (0, 0, 'xy_plane'),
                (90, 0, 'xz_plane'),
                (0, 90, 'yz_plane'),
                (30, 135, 'diagonal1'),
                (30, 225, 'diagonal2'),
            ]
            
            # DC1, DC2, DC3 - multiple angles
            for elev, azim, suffix in view_angles:
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(diffusion_coords[:, 0], diffusion_coords[:, 1], diffusion_coords[:, 2],
                                   c=diffusion_pseudotime, cmap='viridis', s=20, alpha=0.5)
                ax.scatter([diffusion_coords[root_idx, 0]], [diffusion_coords[root_idx, 1]], [diffusion_coords[root_idx, 2]],
                          c='red', s=200, marker='*', label='Root')
                if len(mimic_indices) > 0:
                    mimic_coords = diffusion_coords[mimic_indices, :]
                    ax.scatter(mimic_coords[:, 0], mimic_coords[:, 1], mimic_coords[:, 2],
                              c='lime', s=300, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9)
                if len(candidate_indices) > 0:
                    candidate_coords = diffusion_coords[candidate_indices, :]
                    ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2],
                              c='cyan', s=300, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9)
                # Set robust axis limits
                xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
                ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 1], percentile_low=1.0, percentile_high=99.0)
                zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
                ax.set_xlim(xlim_low, xlim_high)
                ax.set_ylim(ylim_low, ylim_high)
                ax.set_zlim(zlim_low, zlim_high)
                
                ax.set_xlabel('DC1')
                ax.set_ylabel('DC2')
                ax.set_zlabel('DC3')
                ax.set_title(f'Diffusion Map 3D: DC1-DC2-DC3\n(View: elev={elev}°, azim={azim}°)\n(Axis limits: 1st-99th percentile)')
                ax.view_init(elev=elev, azim=azim)
                plt.colorbar(scatter, ax=ax, label='Diffusion Pseudotime')
                ax.legend(loc='best')
                plot_file_3d_dc123 = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_dc123_{suffix}.png')
                plt.savefig(plot_file_3d_dc123, dpi=300, bbox_inches='tight')
                print(f"Saved 3D plot (DC1-DC2-DC3, {suffix}) to {plot_file_3d_dc123}")
                plt.close()
            
            # DC1, DC3, DC4 - multiple angles
            for elev, azim, suffix in view_angles:
                fig = plt.figure(figsize=(12, 9))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(diffusion_coords[:, 0], diffusion_coords[:, 2], diffusion_coords[:, 3],
                                   c=diffusion_pseudotime, cmap='viridis', s=20, alpha=0.5)
                ax.scatter([diffusion_coords[root_idx, 0]], [diffusion_coords[root_idx, 2]], [diffusion_coords[root_idx, 3]],
                          c='red', s=200, marker='*', label='Root')
                if len(mimic_indices) > 0:
                    mimic_coords = diffusion_coords[mimic_indices, :]
                    ax.scatter(mimic_coords[:, 0], mimic_coords[:, 2], mimic_coords[:, 3],
                              c='lime', s=300, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9)
                if len(candidate_indices) > 0:
                    candidate_coords = diffusion_coords[candidate_indices, :]
                    ax.scatter(candidate_coords[:, 0], candidate_coords[:, 2], candidate_coords[:, 3],
                              c='cyan', s=300, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9)
                # Set robust axis limits
                xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
                ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
                zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 3], percentile_low=1.0, percentile_high=99.0)
                ax.set_xlim(xlim_low, xlim_high)
                ax.set_ylim(ylim_low, ylim_high)
                ax.set_zlim(zlim_low, zlim_high)
                
                ax.set_xlabel('DC1')
                ax.set_ylabel('DC3')
                ax.set_zlabel('DC4')
                ax.set_title(f'Diffusion Map 3D: DC1-DC3-DC4\n(View: elev={elev}°, azim={azim}°)\n(Axis limits: 1st-99th percentile)')
                ax.view_init(elev=elev, azim=azim)
                plt.colorbar(scatter, ax=ax, label='Diffusion Pseudotime')
                ax.legend(loc='best')
                plot_file_3d_dc134 = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_dc134_{suffix}.png')
                plt.savefig(plot_file_3d_dc134, dpi=300, bbox_inches='tight')
                print(f"Saved 3D plot (DC1-DC3-DC4, {suffix}) to {plot_file_3d_dc134}")
                plt.close()
            
            # Cloud membership visualizations for DC1-DC2-DC3 and DC1-DC3-DC4
            if transferred_cloud_results is not None:
                transferred_within = transferred_cloud_results['within_cloud']
                transferred_around = transferred_cloud_results['around_cloud']
                transferred_outside = ~transferred_around
                transferred_around_not_within = transferred_around & ~transferred_within
                
                # DC1-DC2-DC3 with cloud membership
                for elev, azim, suffix in view_angles:
                    fig = plt.figure(figsize=(14, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Plot by cloud membership
                    if transferred_outside.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_outside, 0], 
                                  diffusion_coords[transferred_outside, 1], 
                                  diffusion_coords[transferred_outside, 2],
                                  c='lightgray', s=15, alpha=0.2, 
                                  label=f'Outside cloud ({transferred_outside.sum()})', zorder=1)
                    if transferred_around_not_within.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_around_not_within, 0], 
                                  diffusion_coords[transferred_around_not_within, 1], 
                                  diffusion_coords[transferred_around_not_within, 2],
                                  c='orange', s=40, alpha=0.6, 
                                  label=f'Around cloud ({transferred_around_not_within.sum()})', zorder=2)
                    if transferred_within.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_within, 0], 
                                  diffusion_coords[transferred_within, 1], 
                                  diffusion_coords[transferred_within, 2],
                                  c='purple', s=60, alpha=0.8, 
                                  label=f'Within cloud ({transferred_within.sum()})', zorder=3)
                    
                    # Root and highlights
                    ax.scatter([diffusion_coords[root_idx, 0]], 
                              [diffusion_coords[root_idx, 1]], 
                              [diffusion_coords[root_idx, 2]],
                              c='red', s=300, marker='*', label='Root', 
                              zorder=5, edgecolors='black', linewidths=1.5)
                    if len(mimic_indices) > 0:
                        mimic_coords = diffusion_coords[mimic_indices, :]
                        ax.scatter(mimic_coords[:, 0], mimic_coords[:, 1], mimic_coords[:, 2],
                                  c='lime', s=400, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                                  zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
                    if len(candidate_indices) > 0:
                        candidate_coords = diffusion_coords[candidate_indices, :]
                        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2],
                                  c='cyan', s=400, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                                  zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
                    
                    # Set robust axis limits
                    xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
                    ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 1], percentile_low=1.0, percentile_high=99.0)
                    zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
                    ax.set_xlim(xlim_low, xlim_high)
                    ax.set_ylim(ylim_low, ylim_high)
                    ax.set_zlim(zlim_low, zlim_high)
                    
                    ax.set_xlabel('DC1', fontsize=12)
                    ax.set_ylabel('DC2', fontsize=12)
                    ax.set_zlabel('DC3', fontsize=12)
                    ax.set_title(f'Diffusion Map 3D: DC1-DC2-DC3 - Transferred Cloud Membership\n'
                                f'(View: elev={elev}°, azim={azim}°)\n'
                                f'Within: {transferred_within.sum()} | Around: {transferred_around_not_within.sum()} | Outside: {transferred_outside.sum()}', 
                                fontsize=11)
                    ax.view_init(elev=elev, azim=azim)
                    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                    ax.grid(True, alpha=0.3)
                    
                    plot_file_3d_dc123_cloud = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_dc123_cloud_membership_{suffix}.png')
                    plt.savefig(plot_file_3d_dc123_cloud, dpi=300, bbox_inches='tight')
                    print(f"Saved 3D DC1-DC2-DC3 cloud membership plot ({suffix}) to {plot_file_3d_dc123_cloud}")
                    plt.close()
                
                # DC1-DC3-DC4 with cloud membership
                for elev, azim, suffix in view_angles:
                    fig = plt.figure(figsize=(14, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Plot by cloud membership
                    if transferred_outside.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_outside, 0], 
                                  diffusion_coords[transferred_outside, 2], 
                                  diffusion_coords[transferred_outside, 3],
                                  c='lightgray', s=15, alpha=0.2, 
                                  label=f'Outside cloud ({transferred_outside.sum()})', zorder=1)
                    if transferred_around_not_within.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_around_not_within, 0], 
                                  diffusion_coords[transferred_around_not_within, 2], 
                                  diffusion_coords[transferred_around_not_within, 3],
                                  c='orange', s=40, alpha=0.6, 
                                  label=f'Around cloud ({transferred_around_not_within.sum()})', zorder=2)
                    if transferred_within.sum() > 0:
                        ax.scatter(diffusion_coords[transferred_within, 0], 
                                  diffusion_coords[transferred_within, 2], 
                                  diffusion_coords[transferred_within, 3],
                                  c='purple', s=60, alpha=0.8, 
                                  label=f'Within cloud ({transferred_within.sum()})', zorder=3)
                    
                    # Root and highlights
                    ax.scatter([diffusion_coords[root_idx, 0]], 
                              [diffusion_coords[root_idx, 2]], 
                              [diffusion_coords[root_idx, 3]],
                              c='red', s=300, marker='*', label='Root', 
                              zorder=5, edgecolors='black', linewidths=1.5)
                    if len(mimic_indices) > 0:
                        mimic_coords = diffusion_coords[mimic_indices, :]
                        ax.scatter(mimic_coords[:, 0], mimic_coords[:, 2], mimic_coords[:, 3],
                                  c='lime', s=400, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                                  zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
                    if len(candidate_indices) > 0:
                        candidate_coords = diffusion_coords[candidate_indices, :]
                        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 2], candidate_coords[:, 3],
                                  c='cyan', s=400, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                                  zorder=4, edgecolors='black', linewidths=2, alpha=0.9)
                    
                    # Set robust axis limits
                    xlim_low, xlim_high = compute_robust_axis_limits(diffusion_coords[:, 0], percentile_low=1.0, percentile_high=99.0)
                    ylim_low, ylim_high = compute_robust_axis_limits(diffusion_coords[:, 2], percentile_low=1.0, percentile_high=99.0)
                    zlim_low, zlim_high = compute_robust_axis_limits(diffusion_coords[:, 3], percentile_low=1.0, percentile_high=99.0)
                    ax.set_xlim(xlim_low, xlim_high)
                    ax.set_ylim(ylim_low, ylim_high)
                    ax.set_zlim(zlim_low, zlim_high)
                    
                    ax.set_xlabel('DC1', fontsize=12)
                    ax.set_ylabel('DC3', fontsize=12)
                    ax.set_zlabel('DC4', fontsize=12)
                    ax.set_title(f'Diffusion Map 3D: DC1-DC3-DC4 - Transferred Cloud Membership\n'
                                f'(View: elev={elev}°, azim={azim}°)\n'
                                f'Within: {transferred_within.sum()} | Around: {transferred_around_not_within.sum()} | Outside: {transferred_outside.sum()}', 
                                fontsize=11)
                    ax.view_init(elev=elev, azim=azim)
                    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                    ax.grid(True, alpha=0.3)
                    
                    plot_file_3d_dc134_cloud = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_3d_dc134_cloud_membership_{suffix}.png')
                    plt.savefig(plot_file_3d_dc134_cloud, dpi=300, bbox_inches='tight')
                    print(f"Saved 3D DC1-DC3-DC4 cloud membership plot ({suffix}) to {plot_file_3d_dc134_cloud}")
                    plt.close()
    
    # ============================================================
    # 1D Structure Analysis: Project onto DC1 alone
    # ============================================================
    print("\n" + "="*60)
    print("1D Structure Analysis (DC1 Projection)")
    print("="*60)
    
    # 1. Plot histogram of DC1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of DC1
    ax = axes[0, 0]
    ax.hist(diffusion_coords[:, 0], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(diffusion_coords[root_idx, 0], color='red', linestyle='--', linewidth=2, label='Root')
    ax.set_xlabel('Diffusion Coordinate 1 (DC1)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of DC1 (1D Projection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram of DC1 pseudotime
    ax = axes[0, 1]
    ax.hist(dc1_pseudotime, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('DC1 Pseudotime')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of DC1 Pseudotime')
    ax.grid(True, alpha=0.3)
    
    # Plot key features vs DC1
    feature_cols = ['score', 'fident', 'algn_fraction', 'tcov', 'qcov']
    available_features = [col for col in feature_cols if col in data_frame.columns]
    
    if len(available_features) >= 2:
        # Plot first 2 features
        for i, feat in enumerate(available_features[:2]):
            ax = axes[1, i]
            ax.scatter(diffusion_coords[:, 0], data_frame[feat].values, 
                      alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
            # Highlight mimics
            if len(mimic_indices) > 0:
                ax.scatter(diffusion_coords[mimic_indices, 0], 
                          data_frame[feat].values[mimic_indices],
                          c='lime', s=100, marker='o', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
            # Highlight candidates
            if len(candidate_indices) > 0:
                ax.scatter(diffusion_coords[candidate_indices, 0], 
                          data_frame[feat].values[candidate_indices],
                          c='cyan', s=100, marker='s', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
            ax.set_xlabel('Diffusion Coordinate 1 (DC1)')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs DC1')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    elif len(available_features) == 1:
        # Plot single feature
        ax = axes[1, 0]
        ax.scatter(diffusion_coords[:, 0], data_frame[available_features[0]].values, 
                  alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
        if len(mimic_indices) > 0:
            ax.scatter(diffusion_coords[mimic_indices, 0], 
                      data_frame[available_features[0]].values[mimic_indices],
                      c='lime', s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
        if len(candidate_indices) > 0:
            ax.scatter(diffusion_coords[candidate_indices, 0], 
                      data_frame[available_features[0]].values[candidate_indices],
                      c='cyan', s=100, marker='s', edgecolors='black', 
                      linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
        ax.set_xlabel('Diffusion Coordinate 1 (DC1)')
        ax.set_ylabel(available_features[0])
        ax.set_title(f'{available_features[0]} vs DC1')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        axes[1, 1].axis('off')
    else:
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_file_1d = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_1d_analysis.png')
    plt.savefig(plot_file_1d, dpi=300, bbox_inches='tight')
    print(f"Saved 1D analysis plots to {plot_file_1d}")
    plt.close()
    
    # Plot all available features vs DC1 in a grid
    if len(available_features) > 2:
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, feat in enumerate(available_features):
            ax = axes[i]
            ax.scatter(diffusion_coords[:, 0], data_frame[feat].values, 
                      alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
            # Highlight mimics
            if len(mimic_indices) > 0:
                ax.scatter(diffusion_coords[mimic_indices, 0], 
                          data_frame[feat].values[mimic_indices],
                          c='lime', s=100, marker='o', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
            # Highlight candidates
            if len(candidate_indices) > 0:
                ax.scatter(diffusion_coords[candidate_indices, 0], 
                          data_frame[feat].values[candidate_indices],
                          c='cyan', s=100, marker='s', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
            ax.set_xlabel('Diffusion Coordinate 1 (DC1)')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs DC1')
            if i == 0:
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plot_file_features = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_features_vs_dc1.png')
        plt.savefig(plot_file_features, dpi=300, bbox_inches='tight')
        print(f"Saved feature vs DC1 plots to {plot_file_features}")
        plt.close()
    
    # Feature vs DC3 and DC4 plots (if available)
    if n_diffusion_components >= 3 and len(available_features) > 0:
        # Feature vs DC3
        n_features = len(available_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feat in enumerate(available_features):
            ax = axes[i]
            feat_values = data_frame[feat].values
            
            ax.scatter(diffusion_coords[:, 2], feat_values, 
                      alpha=0.3, s=10, c='gray', label='All points')
            
            if len(mimic_indices) > 0:
                ax.scatter(diffusion_coords[mimic_indices, 2], feat_values[mimic_indices],
                          c='lime', s=200, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                          edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
            
            if len(candidate_indices) > 0:
                ax.scatter(diffusion_coords[candidate_indices, 2], feat_values[candidate_indices],
                          c='cyan', s=200, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                          edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
            
            ax.set_xlabel('Diffusion Coordinate 3 (DC3)')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs DC3')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plot_file_features_dc3 = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_features_vs_dc3.png')
        plt.savefig(plot_file_features_dc3, dpi=300, bbox_inches='tight')
        print(f"Saved feature vs DC3 plots to {plot_file_features_dc3}")
        plt.close()
        
        # Feature vs DC4 (if available)
        if n_diffusion_components >= 4:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, feat in enumerate(available_features):
                ax = axes[i]
                feat_values = data_frame[feat].values
                
                ax.scatter(diffusion_coords[:, 3], feat_values, 
                          alpha=0.3, s=10, c='gray', label='All points')
                
                if len(mimic_indices) > 0:
                    ax.scatter(diffusion_coords[mimic_indices, 3], feat_values[mimic_indices],
                              c='lime', s=200, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
                
                if len(candidate_indices) > 0:
                    ax.scatter(diffusion_coords[candidate_indices, 3], feat_values[candidate_indices],
                              c='cyan', s=200, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
                
                ax.set_xlabel('Diffusion Coordinate 4 (DC4)')
                ax.set_ylabel(feat)
                ax.set_title(f'{feat} vs DC4')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plot_file_features_dc4 = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_features_vs_dc4.png')
            plt.savefig(plot_file_features_dc4, dpi=300, bbox_inches='tight')
            print(f"Saved feature vs DC4 plots to {plot_file_features_dc4}")
            plt.close()
    
    # Summary statistics
    print(f"\nDC1 Statistics:")
    print(f"  Mean: {np.mean(diffusion_coords[:, 0]):.4f}")
    print(f"  Std: {np.std(diffusion_coords[:, 0]):.4f}")
    print(f"  Min: {np.min(diffusion_coords[:, 0]):.4f}")
    print(f"  Max: {np.max(diffusion_coords[:, 0]):.4f}")
    print(f"  Range: {np.max(diffusion_coords[:, 0]) - np.min(diffusion_coords[:, 0]):.4f}")
    
    if len(eigenvalues) >= 2 if eigenvalues is not None else False:
        dc1_variance = np.var(diffusion_coords[:, 0])
        dc2_variance = np.var(diffusion_coords[:, 1])
        variance_ratio = dc1_variance / dc2_variance if dc2_variance > 0 else np.inf
        print(f"\nVariance Analysis:")
        print(f"  DC1 variance: {dc1_variance:.4f}")
        print(f"  DC2 variance: {dc2_variance:.4f}")
        print(f"  DC1/DC2 variance ratio: {variance_ratio:.4f}")
        if variance_ratio > 5:
            print(f"  → DC1 dominates; DC2 likely noise correction")
        elif variance_ratio < 2:
            print(f"  → DC2 contains significant signal; consider 2D analysis")
    
    print("\n" + "="*60)
    
    # ============================================================
    # Distance from Attractor Analysis
    # ============================================================
    print("\n" + "="*60)
    print("Distance from Attractor Analysis")
    print("="*60)
    
    # Find reference point in dense region (attractor)
    # Default: use highest local density among known mimics
    # With --natural_attractor flag: use highest local density among all samples
    if use_natural_attractor:
        # Use natural attractor (highest density among all samples)
        from sklearn.neighbors import NearestNeighbors
        n_neighbors_density = min(50, len(diffusion_coords) // 10)
        nn = NearestNeighbors(n_neighbors=n_neighbors_density)
        nn.fit(diffusion_coords[:, :2])  # Use DC1 and DC2
        distances_to_neighbors, _ = nn.kneighbors(diffusion_coords[:, :2])
        local_density = 1.0 / (distances_to_neighbors.mean(axis=1) + 1e-10)
        attractor_idx = np.argmax(local_density)
        attractor_coords = diffusion_coords[attractor_idx, :2]
        print(f"Attractor point (natural/highest density among all samples): index {attractor_idx}")
        print(f"  DC1: {attractor_coords[0]:.4f}, DC2: {attractor_coords[1]:.4f}")
        print(f"  Local density: {local_density[attractor_idx]:.4f}")
    elif len(mimic_indices) > 0:
        # Use mimic-based attractor (highest density among known mimics) - DEFAULT
        # This ensures the attractor represents where mimics cluster, not just where all data clusters
        # Get mimic coordinates in 2D diffusion space
        mimic_coords_2d = diffusion_coords[mimic_indices, :2]
        
        # Compute local density among mimics only
        from sklearn.neighbors import NearestNeighbors
        n_neighbors_density = min(len(mimic_indices) - 1, 10)  # Use fewer neighbors for mimics
        if n_neighbors_density < 1:
            n_neighbors_density = 1
        
        if len(mimic_indices) > 1:
            nn = NearestNeighbors(n_neighbors=n_neighbors_density + 1)  # +1 to exclude self
            nn.fit(mimic_coords_2d)
            distances_to_neighbors, _ = nn.kneighbors(mimic_coords_2d)
            # Local density = inverse of mean distance to neighbors (exclude self, distance 0)
            local_density_mimics = 1.0 / (distances_to_neighbors[:, 1:].mean(axis=1) + 1e-10)
            
            # Find mimic with highest density (attractor)
            mimic_attractor_idx = np.argmax(local_density_mimics)
            attractor_idx = mimic_indices[mimic_attractor_idx]
            attractor_coords = diffusion_coords[attractor_idx, :2]
            
            print(f"Attractor point (highest density among known mimics): index {attractor_idx}")
            print(f"  Mimic query: {data_frame.iloc[attractor_idx]['query']}")
            print(f"  DC1: {attractor_coords[0]:.4f}, DC2: {attractor_coords[1]:.4f}")
            print(f"  Local density (among mimics): {local_density_mimics[mimic_attractor_idx]:.4f}")
            print(f"  Total mimics used: {len(mimic_indices)}")
        else:
            # Only one mimic, use it as attractor
            attractor_idx = mimic_indices[0]
            attractor_coords = diffusion_coords[attractor_idx, :2]
            print(f"Attractor point (only one known mimic): index {attractor_idx}")
            print(f"  Mimic query: {data_frame.iloc[attractor_idx]['query']}")
            print(f"  DC1: {attractor_coords[0]:.4f}, DC2: {attractor_coords[1]:.4f}")
    else:
        # Fallback: no mimics found, use highest density among all samples
        print("Warning: No known mimics found. Using highest density among all samples as attractor.")
        from sklearn.neighbors import NearestNeighbors
        n_neighbors_density = min(50, len(diffusion_coords) // 10)
        nn = NearestNeighbors(n_neighbors=n_neighbors_density)
        nn.fit(diffusion_coords[:, :2])
        distances_to_neighbors, _ = nn.kneighbors(diffusion_coords[:, :2])
        local_density = 1.0 / (distances_to_neighbors.mean(axis=1) + 1e-10)
        attractor_idx = np.argmax(local_density)
        attractor_coords = diffusion_coords[attractor_idx, :2]
        print(f"Attractor point (highest density among all samples): index {attractor_idx}")
        print(f"  DC1: {attractor_coords[0]:.4f}, DC2: {attractor_coords[1]:.4f}")
        print(f"  Local density: {local_density[attractor_idx]:.4f}")
    
    # Compute Euclidean distance in (DC1, DC2) space from attractor
    distances_from_attractor = np.sqrt(
        (diffusion_coords[:, 0] - attractor_coords[0])**2 + 
        (diffusion_coords[:, 1] - attractor_coords[1])**2
    )
    
    # Also compute diffusion distance if we have the transition matrix
    # Diffusion distance: d²(x,y) = Σᵢ (ψᵢ(x) - ψᵢ(y))² / λᵢ
    # where ψᵢ are eigenvectors and λᵢ are eigenvalues
    # Note: eigenvalues array already starts from index 1 (DC1), so index 0 in eigenvalues = DC1
    diffusion_distances = None
    if eigenvalues is not None and len(eigenvalues) >= 1:
        # Use first few components for diffusion distance
        # eigenvalues already excludes DC0, so we can use indices 0, 1, 2, ...
        # eigenvecs[:, 1:] corresponds to DC1, DC2, ... (skipping DC0 at index 0)
        max_available_eigenvals = len(eigenvalues)
        if eigenvecs is not None:
            max_available_eigenvecs = eigenvecs.shape[1] - 1  # -1 because we skip eigenvector 0 (DC0)
            max_available = min(max_available_eigenvals, max_available_eigenvecs)
        else:
            max_available = max_available_eigenvals
        
        n_components_dist = min(5, max_available)
        if n_components_dist > 0 and n_samples < 5000:  # Only if we computed exact eigenvectors
            # Ensure we have enough eigenvalues and eigenvectors
            if len(eigenvalues) >= n_components_dist and eigenvecs is not None and eigenvecs.shape[1] > n_components_dist:
                # eigenvecs[:, 1:n_components_dist+1] gives DC1, DC2, ..., DC(n_components_dist)
                # eigenvalues[0:n_components_dist] gives λ1, λ2, ..., λ(n_components_dist)
                attractor_eigenvec = eigenvecs[attractor_idx, 1:n_components_dist+1]
                all_eigenvecs = eigenvecs[:, 1:n_components_dist+1]
                eigenvals_dist = eigenvalues[0:n_components_dist]
                
                # Ensure shapes match
                if all_eigenvecs.shape[1] == len(eigenvals_dist):
                    # Diffusion distance: weighted by 1/λ (higher weight for larger eigenvalues)
                    diff_vec = all_eigenvecs - attractor_eigenvec
                    diffusion_distances = np.sqrt(np.sum((diff_vec**2) / eigenvals_dist[np.newaxis, :], axis=1))
                    print(f"Computed diffusion distances using {n_components_dist} components")
                else:
                    print(f"  (Shape mismatch: eigenvecs has {all_eigenvecs.shape[1]} components, eigenvalues has {len(eigenvals_dist)}. Skipping diffusion distance.)")
            else:
                print(f"  (Not enough eigenvalues/eigenvectors: have {len(eigenvalues)} eigenvalues, {eigenvecs.shape[1]-1 if eigenvecs is not None else 0} eigenvectors. Skipping diffusion distance.)")
        else:
            print("  (Diffusion distance not computed - using SpectralEmbedding approximation)")
    
    # Add to results
    results_df['distance_from_attractor_euclidean'] = distances_from_attractor
    if diffusion_distances is not None:
        results_df['distance_from_attractor_diffusion'] = diffusion_distances
    
    # Use diffusion distance for plateau analysis if available, otherwise fall back to Euclidean
    if diffusion_distances is not None:
        distances_for_plateau = diffusion_distances
        distance_type = "Diffusion"
        print(f"\nUsing DIFFUSION DISTANCE for plateau analysis")
    else:
        distances_for_plateau = distances_from_attractor
        distance_type = "Euclidean"
        print(f"\nUsing EUCLIDEAN DISTANCE for plateau analysis (diffusion distance not available)")
    
    print(f"\nDistance Statistics ({distance_type} distance):")
    print(f"  Mean: {np.mean(distances_for_plateau):.4f}")
    print(f"  Std: {np.std(distances_for_plateau):.4f}")
    print(f"  Min: {np.min(distances_for_plateau):.4f}")
    print(f"  Max: {np.max(distances_for_plateau):.4f}")
    
    # Also report Euclidean for reference
    print(f"\nDistance Statistics (Euclidean in DC1,DC2 space - for reference):")
    print(f"  Mean: {np.mean(distances_from_attractor):.4f}")
    print(f"  Std: {np.std(distances_from_attractor):.4f}")
    print(f"  Min: {np.min(distances_from_attractor):.4f}")
    print(f"  Max: {np.max(distances_from_attractor):.4f}")
    
    # ============================================================
    # Feature Dynamics Analysis
    # ============================================================
    print("\n" + "="*60)
    print("Feature Dynamics Along Diffusion Space")
    print("="*60)
    
    # Plot features against DC1, DC2, and distance-to-attractor
    feature_cols = ['score', 'fident', 'algn_fraction', 'tcov', 'qcov']
    available_features = [col for col in feature_cols if col in data_frame.columns]
    
    if len(available_features) > 0:
        # Create comprehensive feature dynamics plot
        n_features = len(available_features)
        n_cols = 3  # DC1, DC2, distance
        n_rows = n_features
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feat in enumerate(available_features):
            feat_values = data_frame[feat].values
            
            # Plot vs DC1
            ax = axes[i, 0]
            ax.scatter(diffusion_coords[:, 0], feat_values, 
                      alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
            if len(mimic_indices) > 0:
                ax.scatter(diffusion_coords[mimic_indices, 0], feat_values[mimic_indices],
                          c='lime', s=100, marker='o', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
            if len(candidate_indices) > 0:
                ax.scatter(diffusion_coords[candidate_indices, 0], feat_values[candidate_indices],
                          c='cyan', s=100, marker='s', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
            ax.set_xlabel('DC1')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs DC1')
            if i == 0:
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Plot vs DC2
            ax = axes[i, 1]
            ax.scatter(diffusion_coords[:, 1], feat_values, 
                      alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
            if len(mimic_indices) > 0:
                ax.scatter(diffusion_coords[mimic_indices, 1], feat_values[mimic_indices],
                          c='lime', s=100, marker='o', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
            if len(candidate_indices) > 0:
                ax.scatter(diffusion_coords[candidate_indices, 1], feat_values[candidate_indices],
                          c='cyan', s=100, marker='s', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
            ax.set_xlabel('DC2')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs DC2')
            ax.grid(True, alpha=0.3)
            
            # Plot vs distance from attractor (using diffusion distance if available)
            ax = axes[i, 2]
            ax.scatter(distances_for_plateau, feat_values, 
                      alpha=0.5, s=10, c=dc1_pseudotime, cmap='viridis')
            if len(mimic_indices) > 0:
                ax.scatter(distances_for_plateau[mimic_indices], feat_values[mimic_indices],
                          c='lime', s=100, marker='o', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Mimics', zorder=5)
            if len(candidate_indices) > 0:
                ax.scatter(distances_for_plateau[candidate_indices], feat_values[candidate_indices],
                          c='cyan', s=100, marker='s', edgecolors='black', 
                          linewidths=1.5, alpha=0.9, label='Candidates', zorder=5)
            ax.set_xlabel(f'Distance from Attractor ({distance_type})')
            ax.set_ylabel(feat)
            ax.set_title(f'{feat} vs Distance from Attractor ({distance_type})')
            ax.grid(True, alpha=0.3)
            
            # Analyze feature dynamics: check for plateau vs variation
            # Divide distance into bins and check variance within dense region
            n_bins = 10
            bin_edges = np.linspace(0, np.max(distances_for_plateau), n_bins+1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_vars = []
            bin_means = []
            bin_counts = []
            
            for j in range(n_bins):
                mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau < bin_edges[j+1])
                if j == n_bins - 1:  # Include last point
                    mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau <= bin_edges[j+1])
                if mask.sum() > 0:
                    bin_vars.append(np.var(feat_values[mask]))
                    bin_means.append(np.mean(feat_values[mask]))
                    bin_counts.append(mask.sum())
                else:
                    bin_vars.append(np.nan)
                    bin_means.append(np.nan)
                    bin_counts.append(0)
            
            # Check if feature plateaus in dense region (low distance)
            # Compare variance in closest 30% vs farthest 30%
            sorted_indices = np.argsort(distances_from_attractor)
            n_close = int(len(sorted_indices) * 0.3)
            n_far = int(len(sorted_indices) * 0.3)
            
            close_var = np.var(feat_values[sorted_indices[:n_close]])
            far_var = np.var(feat_values[sorted_indices[-n_far:]])
            close_mean = np.mean(feat_values[sorted_indices[:n_close]])
            far_mean = np.mean(feat_values[sorted_indices[-n_far:]])
            
            # Add trend line to plot
            if len(bin_centers) > 0 and not np.isnan(bin_means).all():
                valid_mask = ~np.isnan(bin_means)
                if valid_mask.sum() > 1:
                    ax.plot(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                           'r-', linewidth=2, alpha=0.7, label='Mean trend')
                    if i == 0:
                        ax.legend(loc='best')
        
        plt.tight_layout()
        plot_file_dynamics = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_feature_dynamics.png')
        plt.savefig(plot_file_dynamics, dpi=300, bbox_inches='tight')
        print(f"Saved feature dynamics plots to {plot_file_dynamics}")
        plt.close()
        
        # Print analysis for each feature and collect results
        print(f"\nFeature Dynamics Analysis (Plateau Detection):")
        print("="*60)
        
        plateau_results = []
        
        for i, feat in enumerate(available_features):
            feat_values = data_frame[feat].values
            sorted_indices = np.argsort(distances_for_plateau)
            n_close = int(len(sorted_indices) * 0.3)
            n_far = int(len(sorted_indices) * 0.3)
            
            close_var = np.var(feat_values[sorted_indices[:n_close]])
            far_var = np.var(feat_values[sorted_indices[-n_far:]])
            close_mean = np.mean(feat_values[sorted_indices[:n_close]])
            far_mean = np.mean(feat_values[sorted_indices[-n_far:]])
            total_var = np.var(feat_values)
            
            # Additional metrics for plateau detection
            # 1. Variance ratio (close/far)
            var_ratio = close_var / far_var if far_var > 0 else np.inf
            
            # 2. Coefficient of variation in close region
            close_std = np.std(feat_values[sorted_indices[:n_close]])
            close_cv = close_std / close_mean if close_mean != 0 else np.inf
            
            # 3. Slope estimate (linear regression on binned means)
            # Use bins computed earlier
            bin_edges = np.linspace(0, np.max(distances_for_plateau), 11)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_means = []
            for j in range(10):
                mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau < bin_edges[j+1])
                if j == 9:
                    mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau <= bin_edges[j+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(feat_values[mask]))
                else:
                    bin_means.append(np.nan)
            
            # Compute slope (rate of change)
            valid_bins = ~np.isnan(bin_means)
            if valid_bins.sum() > 1:
                slope, intercept = np.polyfit(bin_centers[valid_bins], np.array(bin_means)[valid_bins], 1)
            else:
                slope = np.nan
            
            # 4. Plateau score: low variance in close region relative to total
            plateau_score = 1.0 - (close_var / total_var) if total_var > 0 else 0.0
            
            # Classification
            if close_var < 0.01 * total_var and abs(slope) < 0.1 * np.std(feat_values):
                classification = "Strong Plateau (Terminal Regime)"
                plateau_detected = True
            elif close_var < 0.05 * total_var:
                classification = "Moderate Plateau"
                plateau_detected = True
            elif close_var > 0.5 * total_var:
                classification = "No Plateau (Unresolved/Noise)"
                plateau_detected = False
            else:
                classification = "Variable (Some Structure)"
                plateau_detected = False
            
            print(f"\n  {feat}:")
            print(f"    Close to attractor (30%): mean={close_mean:.4f}, std={close_std:.4f}, var={close_var:.4f}")
            print(f"    Far from attractor (30%): mean={far_mean:.4f}, var={far_var:.4f}")
            print(f"    Variance ratio (close/far): {var_ratio:.4f}")
            print(f"    Coefficient of variation (close): {close_cv:.4f}")
            print(f"    Slope (feature change per distance unit): {slope:.6f}")
            print(f"    Plateau score: {plateau_score:.4f} (1.0 = perfect plateau)")
            print(f"    → {classification}")
            
            plateau_results.append({
                'feature': feat,
                'close_mean': close_mean,
                'close_std': close_std,
                'close_var': close_var,
                'far_mean': far_mean,
                'far_var': far_var,
                'var_ratio': var_ratio,
                'close_cv': close_cv,
                'slope': slope,
                'plateau_score': plateau_score,
                'classification': classification,
                'plateau_detected': plateau_detected
            })
        
        # Save plateau detection results
        plateau_df = pd.DataFrame(plateau_results)
        plateau_file = os.path.join(pca_test_dir, f'{organism_name}_plateau_detection.csv')
        plateau_df.to_csv(plateau_file, index=False)
        print(f"\n✓ Plateau detection results saved to: {plateau_file}")
        
        # Summary
        n_plateaus = plateau_df['plateau_detected'].sum()
        print(f"\nSummary:")
        print(f"  Features with plateaus: {n_plateaus}/{len(available_features)}")
        if n_plateaus > 0:
            plateau_features = plateau_df[plateau_df['plateau_detected']]['feature'].tolist()
            print(f"  Plateau features: {', '.join(plateau_features)}")
        
        # Create compact plateau visualization
        # Layout: plots only, no summary table
        if len(available_features) > 0:
            from matplotlib.gridspec import GridSpec
            
            # Create figure optimized for plots only
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, 
                         left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # Left column: 3 plots stacked
            plot_axes_left = []
            if len(available_features) >= 1:
                plot_axes_left.append(fig.add_subplot(gs[0, 0]))
            if len(available_features) >= 2:
                plot_axes_left.append(fig.add_subplot(gs[1, 0]))
            if len(available_features) >= 3:
                plot_axes_left.append(fig.add_subplot(gs[2, 0]))
            
            # Right column: remaining plots stacked
            plot_axes_right = []
            if len(available_features) >= 4:
                plot_axes_right.append(fig.add_subplot(gs[0, 1]))
            if len(available_features) >= 5:
                plot_axes_right.append(fig.add_subplot(gs[1, 1]))
            if len(available_features) >= 6:
                plot_axes_right.append(fig.add_subplot(gs[2, 1]))
            
            # Helper function to create a single plateau plot
            def create_plateau_plot(ax, feat, result, feat_values, distances_for_plateau, distance_type):
                # Scatter plot
                ax.scatter(distances_for_plateau, feat_values, 
                          alpha=0.3, s=8, c='gray', label='All points')
                
                # Highlight close region
                sorted_indices = np.argsort(distances_for_plateau)
                n_close = int(len(sorted_indices) * 0.3)
                close_mask = np.zeros(len(distances_for_plateau), dtype=bool)
                close_mask[sorted_indices[:n_close]] = True
                
                ax.scatter(distances_for_plateau[close_mask], feat_values[close_mask],
                          alpha=0.7, s=20, c='red', label='Close (30%)', zorder=5)
                
                # Add binned mean trend
                bin_edges = np.linspace(0, np.max(distances_for_plateau), 11)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_means = []
                bin_stds = []
                for j in range(10):
                    mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau < bin_edges[j+1])
                    if j == 9:
                        mask = (distances_for_plateau >= bin_edges[j]) & (distances_for_plateau <= bin_edges[j+1])
                    if mask.sum() > 0:
                        bin_means.append(np.mean(feat_values[mask]))
                        bin_stds.append(np.std(feat_values[mask]))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                
                valid_mask = ~np.isnan(bin_means)
                if valid_mask.sum() > 1:
                    ax.plot(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                           'b-', linewidth=2, alpha=0.8, label='Mean', zorder=4)
                    ax.fill_between(bin_centers[valid_mask], 
                                   np.array(bin_means)[valid_mask] - np.array(bin_stds)[valid_mask],
                                   np.array(bin_means)[valid_mask] + np.array(bin_stds)[valid_mask],
                                   alpha=0.15, color='blue')
                
                # Add plateau region highlight
                close_threshold = np.percentile(distances_for_plateau, 30)
                ax.axvspan(0, close_threshold, alpha=0.1, color='red')
                
                ax.set_xlabel(f'Distance ({distance_type})', fontsize=9)
                ax.set_ylabel(feat, fontsize=9)
                ax.set_title(f'{feat}: {result["classification"]}\nScore: {result["plateau_score"]:.3f}, Slope: {result["slope"]:.4f}', 
                           fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
            
            # Plot left column (first 3 features)
            for i, (feat, result) in enumerate(zip(available_features[:3], plateau_results[:3])):
                if i < len(plot_axes_left):
                    feat_values = data_frame[feat].values
                    create_plateau_plot(plot_axes_left[i], feat, result, feat_values, distances_for_plateau, distance_type)
            
            # Plot right column (remaining features, up to 3)
            for i, (feat, result) in enumerate(zip(available_features[3:6], plateau_results[3:6])):
                if i < len(plot_axes_right):
                    feat_values = data_frame[feat].values
                    create_plateau_plot(plot_axes_right[i], feat, result, feat_values, distances_for_plateau, distance_type)
            
            # No overall title (removed per user request)
            
            plateau_plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_plateau_detection.png')
            plt.savefig(plateau_plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ Plateau detection visualization saved to: {plateau_plot_file}")
            plt.close()
    
    # ============================================================
    # 3D Distance Plateau Analysis (DC1, DC2, DC3)
    # ============================================================
    if n_diffusion_components >= 3:
        print("\n" + "="*60)
        print("3D Distance Plateau Analysis (DC1-DC2-DC3)")
        print("="*60)
        
        # Find attractor in 3D space (DC1, DC2, DC3)
        # Use same method as 2D attractor (mimic-based or natural)
        if use_natural_attractor:
            # Use natural attractor (highest density among all samples)
            n_neighbors_density_3d = min(50, len(diffusion_coords) // 10)
            nn_3d = NearestNeighbors(n_neighbors=n_neighbors_density_3d)
            nn_3d.fit(diffusion_coords[:, :3])  # Use DC1, DC2, DC3
            distances_to_neighbors_3d, _ = nn_3d.kneighbors(diffusion_coords[:, :3])
            local_density_3d = 1.0 / (distances_to_neighbors_3d.mean(axis=1) + 1e-10)
            attractor_idx_3d = np.argmax(local_density_3d)
            attractor_coords_3d = diffusion_coords[attractor_idx_3d, :3]
            print(f"3D Attractor point (natural/highest density among all samples): index {attractor_idx_3d}")
            print(f"  DC1: {attractor_coords_3d[0]:.4f}, DC2: {attractor_coords_3d[1]:.4f}, DC3: {attractor_coords_3d[2]:.4f}")
            print(f"  Local density: {local_density_3d[attractor_idx_3d]:.4f}")
        elif len(mimic_indices) > 0:
            # Use mimic-based attractor (highest density among known mimics) - DEFAULT
            # Get mimic coordinates in 3D diffusion space
            mimic_coords_3d = diffusion_coords[mimic_indices, :3]
            
            # Compute local density among mimics only
            n_neighbors_density_3d = min(len(mimic_indices) - 1, 10)
            if n_neighbors_density_3d < 1:
                n_neighbors_density_3d = 1
            
            if len(mimic_indices) > 1:
                nn_3d = NearestNeighbors(n_neighbors=n_neighbors_density_3d + 1)  # +1 to exclude self
                nn_3d.fit(mimic_coords_3d)
                distances_to_neighbors_3d, _ = nn_3d.kneighbors(mimic_coords_3d)
                # Local density = inverse of mean distance to neighbors (exclude self)
                local_density_3d_mimics = 1.0 / (distances_to_neighbors_3d[:, 1:].mean(axis=1) + 1e-10)
                
                # Find mimic with highest density in 3D space
                mimic_attractor_idx_3d = np.argmax(local_density_3d_mimics)
                attractor_idx_3d = mimic_indices[mimic_attractor_idx_3d]
                attractor_coords_3d = diffusion_coords[attractor_idx_3d, :3]
                print(f"3D Attractor point (highest density among known mimics): index {attractor_idx_3d}")
                print(f"  Mimic query: {data_frame.iloc[attractor_idx_3d]['query']}")
                print(f"  DC1: {attractor_coords_3d[0]:.4f}, DC2: {attractor_coords_3d[1]:.4f}, DC3: {attractor_coords_3d[2]:.4f}")
                print(f"  Local density (among mimics): {local_density_3d_mimics[mimic_attractor_idx_3d]:.4f}")
            else:
                # Only one mimic, use it as attractor
                attractor_idx_3d = mimic_indices[0]
                attractor_coords_3d = diffusion_coords[attractor_idx_3d, :3]
                print(f"3D Attractor point (only one known mimic): index {attractor_idx_3d}")
                print(f"  Mimic query: {data_frame.iloc[attractor_idx_3d]['query']}")
                print(f"  DC1: {attractor_coords_3d[0]:.4f}, DC2: {attractor_coords_3d[1]:.4f}, DC3: {attractor_coords_3d[2]:.4f}")
        else:
            # Fallback: no mimics found, use highest density among all samples
            print("Warning: No known mimics found. Using highest density among all samples as 3D attractor.")
            n_neighbors_density_3d = min(50, len(diffusion_coords) // 10)
            nn_3d = NearestNeighbors(n_neighbors=n_neighbors_density_3d)
            nn_3d.fit(diffusion_coords[:, :3])
            distances_to_neighbors_3d, _ = nn_3d.kneighbors(diffusion_coords[:, :3])
            local_density_3d = 1.0 / (distances_to_neighbors_3d.mean(axis=1) + 1e-10)
            attractor_idx_3d = np.argmax(local_density_3d)
            attractor_coords_3d = diffusion_coords[attractor_idx_3d, :3]
            print(f"3D Attractor point (highest density among all samples): index {attractor_idx_3d}")
            print(f"  DC1: {attractor_coords_3d[0]:.4f}, DC2: {attractor_coords_3d[1]:.4f}, DC3: {attractor_coords_3d[2]:.4f}")
            print(f"  Local density: {local_density_3d[attractor_idx_3d]:.4f}")
        
        # Compute Euclidean distance in (DC1, DC2, DC3) space from attractor
        distances_from_attractor_3d_euclidean = np.sqrt(
            (diffusion_coords[:, 0] - attractor_coords_3d[0])**2 + 
            (diffusion_coords[:, 1] - attractor_coords_3d[1])**2 +
            (diffusion_coords[:, 2] - attractor_coords_3d[2])**2
        )
        
        # Compute 3D diffusion distance if available
        # Note: eigenvalues array already starts from index 1 (DC1), so index 0 in eigenvalues = DC1
        diffusion_distances_3d = None
        if eigenvalues is not None and len(eigenvalues) >= 3:
            max_available_eigenvals = len(eigenvalues)
            if eigenvecs is not None:
                max_available_eigenvecs = eigenvecs.shape[1] - 1  # -1 because we skip eigenvector 0 (DC0)
                max_available = min(max_available_eigenvals, max_available_eigenvecs)
            else:
                max_available = max_available_eigenvals
            
            n_components_dist_3d = min(3, max_available)
            if n_components_dist_3d > 0 and n_samples < 5000:  # Only if we computed exact eigenvectors
                if len(eigenvalues) >= n_components_dist_3d and eigenvecs is not None and eigenvecs.shape[1] > n_components_dist_3d:
                    # eigenvecs[:, 1:n_components_dist_3d+1] gives DC1, DC2, DC3
                    # eigenvalues[0:n_components_dist_3d] gives λ1, λ2, λ3
                    attractor_eigenvec_3d = eigenvecs[attractor_idx_3d, 1:n_components_dist_3d+1]
                    all_eigenvecs_3d = eigenvecs[:, 1:n_components_dist_3d+1]
                    eigenvals_dist_3d = eigenvalues[0:n_components_dist_3d]
                    
                    # Ensure shapes match
                    if all_eigenvecs_3d.shape[1] == len(eigenvals_dist_3d):
                        # Diffusion distance: weighted by 1/λ (higher weight for larger eigenvalues)
                        diff_vec_3d = all_eigenvecs_3d - attractor_eigenvec_3d
                        diffusion_distances_3d = np.sqrt(np.sum((diff_vec_3d**2) / eigenvals_dist_3d[np.newaxis, :], axis=1))
                        print(f"Computed 3D diffusion distances using {n_components_dist_3d} components")
                    else:
                        print(f"  (Shape mismatch: eigenvecs has {all_eigenvecs_3d.shape[1]} components, eigenvalues has {len(eigenvals_dist_3d)}. Skipping 3D diffusion distance.)")
                else:
                    print(f"  (Not enough eigenvalues/eigenvectors: have {len(eigenvalues)} eigenvalues, {eigenvecs.shape[1]-1 if eigenvecs is not None else 0} eigenvectors. Skipping 3D diffusion distance.)")
            else:
                print("  (3D Diffusion distance not computed - using SpectralEmbedding approximation)")
        
        # Use diffusion distance for 3D plateau analysis if available, otherwise fall back to Euclidean
        if diffusion_distances_3d is not None:
            distances_for_plateau_3d = diffusion_distances_3d
            distance_type_3d = "Diffusion"
            print(f"Using 3D DIFFUSION DISTANCE for plateau analysis")
        else:
            distances_for_plateau_3d = distances_from_attractor_3d_euclidean
            distance_type_3d = "Euclidean"
            print(f"Using 3D EUCLIDEAN DISTANCE for plateau analysis (diffusion distance not available)")
        
        # Add to results DataFrame
        results_df['distance_from_attractor_3d_euclidean'] = distances_from_attractor_3d_euclidean
        if diffusion_distances_3d is not None:
            results_df['distance_from_attractor_3d_diffusion'] = diffusion_distances_3d
        
        print(f"\n3D Distance Statistics ({distance_type_3d} distance):")
        print(f"  Mean: {np.mean(distances_for_plateau_3d):.4f}")
        print(f"  Std: {np.std(distances_for_plateau_3d):.4f}")
        print(f"  Min: {np.min(distances_for_plateau_3d):.4f}")
        print(f"  Max: {np.max(distances_for_plateau_3d):.4f}")
        
        # Also report Euclidean for reference
        print(f"\n3D Distance Statistics (Euclidean - for reference):")
        print(f"  Mean: {np.mean(distances_from_attractor_3d_euclidean):.4f}")
        print(f"  Std: {np.std(distances_from_attractor_3d_euclidean):.4f}")
        print(f"  Min: {np.min(distances_from_attractor_3d_euclidean):.4f}")
        print(f"  Max: {np.max(distances_from_attractor_3d_euclidean):.4f}")
        
        # Feature dynamics plots vs 3D distance
        if len(available_features) > 0:
            n_features = len(available_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, feat in enumerate(available_features):
                ax = axes[i]
                feat_values = data_frame[feat].values
                
                ax.scatter(distances_for_plateau_3d, feat_values, 
                          alpha=0.3, s=10, c='gray', label='All points')
                
                if len(mimic_indices) > 0:
                    ax.scatter(distances_for_plateau_3d[mimic_indices], feat_values[mimic_indices],
                              c='lime', s=200, marker='o', label=f'Mimics (n={len(mimic_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
                
                if len(candidate_indices) > 0:
                    ax.scatter(distances_for_plateau_3d[candidate_indices], feat_values[candidate_indices],
                              c='cyan', s=200, marker='s', label=f'Candidates (n={len(candidate_indices)})', 
                              edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
                
                # Add binned mean trend
                n_bins = 20
                bin_edges = np.linspace(0, np.max(distances_for_plateau_3d), n_bins+1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_means = []
                bin_stds = []
                for j in range(n_bins):
                    mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d < bin_edges[j+1])
                    if j == n_bins - 1:
                        mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d <= bin_edges[j+1])
                    if mask.sum() > 0:
                        bin_means.append(np.mean(feat_values[mask]))
                        bin_stds.append(np.std(feat_values[mask]))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                
                valid_mask = ~np.isnan(bin_means)
                if valid_mask.sum() > 1:
                    ax.plot(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                           'b-', linewidth=3, alpha=0.8, label='Binned mean', zorder=4)
                    ax.fill_between(bin_centers[valid_mask], 
                                   np.array(bin_means)[valid_mask] - np.array(bin_stds)[valid_mask],
                                   np.array(bin_means)[valid_mask] + np.array(bin_stds)[valid_mask],
                                   alpha=0.2, color='blue', label='±1 std')
                
                ax.set_xlabel(f'Distance from Attractor (3D: DC1-DC2-DC3, {distance_type_3d})')
                ax.set_ylabel(feat)
                ax.set_title(f'{feat} vs 3D Distance ({distance_type_3d})')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plot_file_dynamics_3d = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_feature_dynamics_3d.png')
            plt.savefig(plot_file_dynamics_3d, dpi=300, bbox_inches='tight')
            print(f"Saved feature dynamics vs 3D distance plots to {plot_file_dynamics_3d}")
            plt.close()
        
        # Plateau detection using 3D distance
        print(f"\n3D Plateau Detection Analysis:")
        print("="*60)
        
        plateau_results_3d = []
        
        for i, feat in enumerate(available_features):
            feat_values = data_frame[feat].values
            sorted_indices = np.argsort(distances_for_plateau_3d)
            n_close = int(len(sorted_indices) * 0.3)
            n_far = int(len(sorted_indices) * 0.3)
            
            close_var = np.var(feat_values[sorted_indices[:n_close]])
            far_var = np.var(feat_values[sorted_indices[-n_far:]])
            close_mean = np.mean(feat_values[sorted_indices[:n_close]])
            far_mean = np.mean(feat_values[sorted_indices[-n_far:]])
            total_var = np.var(feat_values)
            
            # Additional metrics for plateau detection
            var_ratio = close_var / far_var if far_var > 0 else np.inf
            
            close_std = np.std(feat_values[sorted_indices[:n_close]])
            close_cv = close_std / close_mean if close_mean != 0 else np.inf
            
            # Slope estimate (linear regression on binned means)
            bin_edges = np.linspace(0, np.max(distances_for_plateau_3d), 11)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_means = []
            for j in range(10):
                mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d < bin_edges[j+1])
                if j == 9:
                    mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d <= bin_edges[j+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(feat_values[mask]))
                else:
                    bin_means.append(np.nan)
            
            valid_bins = ~np.isnan(bin_means)
            if valid_bins.sum() > 1:
                slope, intercept = np.polyfit(bin_centers[valid_bins], np.array(bin_means)[valid_bins], 1)
            else:
                slope = np.nan
            
            # Plateau score: low variance in close region relative to total
            plateau_score = 1.0 - (close_var / total_var) if total_var > 0 else 0.0
            
            # Classification
            if close_var < 0.01 * total_var and abs(slope) < 0.1 * np.std(feat_values):
                classification = "Strong Plateau (Terminal Regime)"
                plateau_detected = True
            elif close_var < 0.05 * total_var:
                classification = "Moderate Plateau"
                plateau_detected = True
            elif close_var > 0.5 * total_var:
                classification = "No Plateau (Unresolved/Noise)"
                plateau_detected = False
            else:
                classification = "Variable (Some Structure)"
                plateau_detected = False
            
            print(f"\n  {feat} (3D distance):")
            print(f"    Close to attractor (30%): mean={close_mean:.4f}, std={close_std:.4f}, var={close_var:.4f}")
            print(f"    Far from attractor (30%): mean={far_mean:.4f}, var={far_var:.4f}")
            print(f"    Variance ratio (close/far): {var_ratio:.4f}")
            print(f"    Coefficient of variation (close): {close_cv:.4f}")
            print(f"    Slope (feature change per distance unit): {slope:.6f}")
            print(f"    Plateau score: {plateau_score:.4f} (1.0 = perfect plateau)")
            print(f"    → {classification}")
            
            plateau_results_3d.append({
                'feature': feat,
                'close_mean': close_mean,
                'close_std': close_std,
                'close_var': close_var,
                'far_mean': far_mean,
                'far_var': far_var,
                'var_ratio': var_ratio,
                'close_cv': close_cv,
                'slope': slope,
                'plateau_score': plateau_score,
                'classification': classification,
                'plateau_detected': plateau_detected
            })
        
        # Save 3D plateau detection results
        plateau_df_3d = pd.DataFrame(plateau_results_3d)
        plateau_file_3d = os.path.join(pca_test_dir, f'{organism_name}_plateau_detection_3d.csv')
        plateau_df_3d.to_csv(plateau_file_3d, index=False)
        print(f"\n✓ 3D plateau detection results saved to: {plateau_file_3d}")
        
        # Summary
        n_plateaus_3d = plateau_df_3d['plateau_detected'].sum()
        print(f"\n3D Plateau Summary:")
        print(f"  Features with plateaus: {n_plateaus_3d}/{len(available_features)}")
        if n_plateaus_3d > 0:
            plateau_features_3d = plateau_df_3d[plateau_df_3d['plateau_detected']]['feature'].tolist()
            print(f"  Plateau features: {', '.join(plateau_features_3d)}")
        
        # Create compact 3D plateau visualization (plots only, no summary table)
        if len(available_features) > 0:
            from matplotlib.gridspec import GridSpec
            
            # Create figure optimized for plots only
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, 
                         left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # Left column: 3 plots stacked
            plot_axes_left = []
            if len(available_features) >= 1:
                plot_axes_left.append(fig.add_subplot(gs[0, 0]))
            if len(available_features) >= 2:
                plot_axes_left.append(fig.add_subplot(gs[1, 0]))
            if len(available_features) >= 3:
                plot_axes_left.append(fig.add_subplot(gs[2, 0]))
            
            # Right column: remaining plots stacked
            plot_axes_right = []
            if len(available_features) >= 4:
                plot_axes_right.append(fig.add_subplot(gs[0, 1]))
            if len(available_features) >= 5:
                plot_axes_right.append(fig.add_subplot(gs[1, 1]))
            if len(available_features) >= 6:
                plot_axes_right.append(fig.add_subplot(gs[2, 1]))
            
            # Helper function to create a single 3D plateau plot
            def create_plateau_plot_3d(ax, feat, result, feat_values, distances_for_plateau_3d, distance_type_3d):
                # Scatter plot
                ax.scatter(distances_for_plateau_3d, feat_values, 
                          alpha=0.3, s=8, c='gray', label='All points')
                
                # Highlight close region
                sorted_indices = np.argsort(distances_for_plateau_3d)
                n_close = int(len(sorted_indices) * 0.3)
                close_mask = np.zeros(len(distances_for_plateau_3d), dtype=bool)
                close_mask[sorted_indices[:n_close]] = True
                
                ax.scatter(distances_for_plateau_3d[close_mask], feat_values[close_mask],
                          alpha=0.7, s=20, c='red', label='Close (30%)', zorder=5)
                
                # Add binned mean trend
                bin_edges = np.linspace(0, np.max(distances_for_plateau_3d), 11)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_means = []
                bin_stds = []
                for j in range(10):
                    mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d < bin_edges[j+1])
                    if j == 9:
                        mask = (distances_for_plateau_3d >= bin_edges[j]) & (distances_for_plateau_3d <= bin_edges[j+1])
                    if mask.sum() > 0:
                        bin_means.append(np.mean(feat_values[mask]))
                        bin_stds.append(np.std(feat_values[mask]))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                
                valid_mask = ~np.isnan(bin_means)
                if valid_mask.sum() > 1:
                    ax.plot(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                           'b-', linewidth=2, alpha=0.8, label='Mean', zorder=4)
                    ax.fill_between(bin_centers[valid_mask], 
                                   np.array(bin_means)[valid_mask] - np.array(bin_stds)[valid_mask],
                                   np.array(bin_means)[valid_mask] + np.array(bin_stds)[valid_mask],
                                   alpha=0.15, color='blue')
                
                # Add plateau region highlight
                close_threshold = np.percentile(distances_for_plateau_3d, 30)
                ax.axvspan(0, close_threshold, alpha=0.1, color='red')
                
                ax.set_xlabel(f'Distance 3D ({distance_type_3d})', fontsize=9)
                ax.set_ylabel(feat, fontsize=9)
                ax.set_title(f'{feat}: {result["classification"]}\nScore: {result["plateau_score"]:.3f}, Slope: {result["slope"]:.4f}', 
                           fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
            
            # Plot left column (first 3 features)
            for i, (feat, result) in enumerate(zip(available_features[:3], plateau_results_3d[:3])):
                if i < len(plot_axes_left):
                    feat_values = data_frame[feat].values
                    create_plateau_plot_3d(plot_axes_left[i], feat, result, feat_values, distances_for_plateau_3d, distance_type_3d)
            
            # Plot right column (remaining features, up to 3)
            for i, (feat, result) in enumerate(zip(available_features[3:6], plateau_results_3d[3:6])):
                if i < len(plot_axes_right):
                    feat_values = data_frame[feat].values
                    create_plateau_plot_3d(plot_axes_right[i], feat, result, feat_values, distances_for_plateau_3d, distance_type_3d)
            
            # No overall title (removed per user request)
            
            plateau_plot_file_3d = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_plateau_detection_3d.png')
            plt.savefig(plateau_plot_file_3d, dpi=300, bbox_inches='tight')
            print(f"✓ 3D plateau detection visualization saved to: {plateau_plot_file_3d}")
            plt.close()
    
    # Update saved results with new distance columns
    results_df.to_csv(output_file, index=False)
    print(f"\nUpdated results saved to {output_file}")
    
    # Generate diffusion map summary: samples within 3D distance of top 10 closest known mimics
    print("\n" + "="*60)
    if is_multi_system:
        print("Generating Joint Diffusion Map Summary: Mimic Cloud in Unified Manifold")
        print("="*60)
        print("  Cloud defined in JOINT DIFFUSION SPACE (not feature space)")
        print("  This allows direct comparison of mimic regions across systems")
    else:
        print("Generating Diffusion Map Summary: Samples near Top 10 Closest Known Mimics")
        print("="*60)
    
    # Find known mimics in results
    if 'query' in results_df.columns:
        mimic_mask = results_df['query'].isin(validation_list).values
        mimic_indices = np.where(mimic_mask)[0]
        
        if len(mimic_indices) > 0:
            # Get 3D diffusion coordinates for mimics
            dc_cols = [col for col in results_df.columns if col.startswith('Diffusion')]
            if len(dc_cols) >= 3:
                mimic_coords_3d = results_df.iloc[mimic_indices][dc_cols[:3]].values
                all_coords_3d = results_df[dc_cols[:3]].values
                
                # Find mimics that form the main cluster in diffusion space
                # Goal: Exclude outlier mimics that are far from the main cluster
                # Strategy: Use a fixed threshold based on percentile distances to identify core cluster
                # Compute pairwise distances between all mimics in 3D space
                mimic_coords_all = all_coords_3d[mimic_indices]
                from scipy.spatial.distance import cdist
                mimic_pairwise_distances_all = cdist(mimic_coords_all, mimic_coords_all, metric='euclidean')
                
                if len(mimic_indices) > 1:
                    upper_triangle = np.triu(mimic_pairwise_distances_all, k=1)
                    mimic_pairwise_flat = upper_triangle[upper_triangle > 0]
                    
                    if len(mimic_pairwise_flat) > 0:
                        # Calculate distance statistics
                        min_pairwise_dist = np.min(mimic_pairwise_flat)
                        q25_pairwise_dist = np.percentile(mimic_pairwise_flat, 25)
                        q50_pairwise_dist = np.median(mimic_pairwise_flat)
                        q75_pairwise_dist = np.percentile(mimic_pairwise_flat, 75)
                        max_pairwise_dist = np.max(mimic_pairwise_flat)
                        
                        print(f"  Mimic pairwise distance statistics:")
                        print(f"    Min: {min_pairwise_dist:.6f}")
                        print(f"    25th percentile: {q25_pairwise_dist:.6f}")
                        print(f"    Median: {q50_pairwise_dist:.6f}")
                        print(f"    75th percentile: {q75_pairwise_dist:.6f}")
                        print(f"    Max: {max_pairwise_dist:.6f}")
                        
                        # Use a tighter threshold to identify the main cluster
                        # Try progressively tighter thresholds, preferring ones that exclude outliers
                        # Goal: Find the tightest threshold that still gives a substantial cluster
                        thresholds_to_try = [
                            q25_pairwise_dist * 0.75,  # Tightest: 75% of 25th percentile
                            q25_pairwise_dist,  # 25th percentile
                            q50_pairwise_dist,  # Median (looser)
                        ]
                        
                        best_cluster = None
                        best_threshold = None
                        min_cluster_size = max(2, len(mimic_indices) // 3)  # Want at least 1/3 of mimics, or 2 minimum
                        
                        for cluster_threshold in thresholds_to_try:
                            # Build adjacency graph: mimics are connected if within threshold
                            # Find connected components (clusters)
                            from scipy.sparse import csgraph
                            adjacency = (mimic_pairwise_distances_all <= cluster_threshold).astype(int)
                            # Remove diagonal (self-connections)
                            np.fill_diagonal(adjacency, 0)
                            
                            # Find connected components
                            n_components, labels = csgraph.connected_components(
                                csgraph=adjacency, directed=False, return_labels=True
                            )
                            
                            # Find the largest component
                            if n_components > 0:
                                component_sizes = [np.sum(labels == i) for i in range(n_components)]
                                component_sizes_sorted = sorted(component_sizes, reverse=True)
                                largest_component_size = component_sizes_sorted[0]
                                largest_component_idx = np.argmax(component_sizes)
                                largest_component = np.where(labels == largest_component_idx)[0]
                                
                                # Prefer this threshold if:
                                # 1. It gives a cluster of reasonable size (at least min_cluster_size)
                                # 2. It excludes at least one mimic (not all mimics)
                                # 3. The largest cluster is clearly dominant (at least 2x the second largest, if exists)
                                is_good_cluster = (
                                    largest_component_size >= min_cluster_size and
                                    largest_component_size < len(mimic_indices)  # Excludes at least one
                                )
                                
                                if len(component_sizes_sorted) > 1:
                                    second_largest = component_sizes_sorted[1]
                                    is_dominant = largest_component_size >= second_largest * 2
                                    is_good_cluster = is_good_cluster and is_dominant
                                
                                if is_good_cluster:
                                    best_cluster = largest_component
                                    best_threshold = cluster_threshold
                                    print(f"  Using threshold: {cluster_threshold:.6f}")
                                    print(f"  Found main cluster with {len(best_cluster)} mimics (excludes {len(mimic_indices) - len(best_cluster)} outlier(s))")
                                    break
                        
                        # Use the best cluster found, or fall back if no good cluster found
                        if best_cluster is not None:
                            core_mimic_indices = best_cluster.tolist()
                            cluster_threshold = best_threshold
                        else:
                            # Fallback: use a very tight threshold (25th percentile * 0.5) to force exclusion
                            cluster_threshold = q25_pairwise_dist * 0.5
                            print(f"  No ideal cluster found, using tight threshold ({cluster_threshold:.6f})")
                            
                            # Build adjacency and find largest component
                            from scipy.sparse import csgraph
                            adjacency = (mimic_pairwise_distances_all <= cluster_threshold).astype(int)
                            np.fill_diagonal(adjacency, 0)
                            n_components, labels = csgraph.connected_components(
                                csgraph=adjacency, directed=False, return_labels=True
                            )
                            
                            if n_components > 0:
                                component_sizes = [np.sum(labels == i) for i in range(n_components)]
                                largest_component_idx = np.argmax(component_sizes)
                                core_mimic_indices = np.where(labels == largest_component_idx)[0].tolist()
                                print(f"  Found cluster with {len(core_mimic_indices)} mimics using tight threshold")
                            else:
                                # Last resort: use median with core-based approach
                                cluster_threshold = q50_pairwise_dist
                                print(f"  Using median threshold ({cluster_threshold:.6f}) as last resort")
                                
                                # Find the mimic with the most neighbors
                                neighbor_counts = []
                                for i in range(len(mimic_indices)):
                                    neighbors = np.sum((mimic_pairwise_distances_all[i, :] <= cluster_threshold) & 
                                                      (mimic_pairwise_distances_all[i, :] > 0))
                                    neighbor_counts.append(neighbors)
                                
                                core_mimic_idx = np.argmax(neighbor_counts)
                                core_mimic_indices = [core_mimic_idx]
                                
                                # Add all mimics within threshold of core
                                for i in range(len(mimic_indices)):
                                    if i != core_mimic_idx:
                                        if mimic_pairwise_distances_all[core_mimic_idx, i] <= cluster_threshold:
                                            core_mimic_indices.append(i)
                                print(f"  Found cluster with {len(core_mimic_indices)} mimics")
                        
                        # Report which mimics were included/excluded
                        excluded_indices = [i for i in range(len(mimic_indices)) if i not in core_mimic_indices]
                        print(f"  Core cluster: {len(core_mimic_indices)} mimics included")
                        if len(excluded_indices) > 0:
                            excluded_queries = results_df.iloc[mimic_indices[excluded_indices]]['query'].values
                            print(f"  Excluded {len(excluded_indices)} outlier mimic(s): {', '.join(excluded_queries)}")
                            # Show distances of excluded mimics to nearest core mimic
                            core_coords = mimic_coords_all[core_mimic_indices]
                            for excl_idx in excluded_indices:
                                excl_coord = mimic_coords_all[excl_idx:excl_idx+1]
                                dists_to_core = cdist(excl_coord, core_coords, metric='euclidean')[0]
                                min_dist_to_core = np.min(dists_to_core)
                                print(f"    - {results_df.iloc[mimic_indices[excl_idx]]['query']}: min distance to core = {min_dist_to_core:.6f} (threshold = {cluster_threshold:.6f})")
                        else:
                            print(f"  Warning: All mimics included - consider using a tighter threshold")
                        
                        top_10_mimic_indices = mimic_indices[core_mimic_indices]
                    else:
                        # Only one unique mimic, use all
                        top_10_mimic_indices = mimic_indices
                        print(f"  Using all {len(mimic_indices)} mimics (only one unique position)")
                    
                    top_10_mimic_queries = results_df.iloc[top_10_mimic_indices]['query'].values
                else:
                    top_10_mimic_indices = mimic_indices
                    top_10_mimic_queries = results_df.iloc[mimic_indices]['query'].values
                
                # Create a "cloud" region around all known mimics
                # Get all mimic coordinates in 3D space
                mimic_coords_3d = all_coords_3d[top_10_mimic_indices]
                
                # Compute distances from each sample to the nearest mimic
                from scipy.spatial.distance import cdist
                distances_to_mimics = cdist(all_coords_3d, mimic_coords_3d, metric='euclidean')
                min_dist_to_any_mimic = np.min(distances_to_mimics, axis=1)
                
                # Define the "cloud" region
                # Option 1: Use convex hull approach (more complex, but captures shape)
                # Option 2: Use distance-based: samples within X distance of any mimic
                # Option 3: Use bounding box with padding
                
                # Calculate the spread of mimics to determine appropriate buffer
                mimic_spread = np.max(mimic_coords_3d, axis=0) - np.min(mimic_coords_3d, axis=0)
                avg_spread = np.mean(mimic_spread)
                
                # Calculate pairwise distances between mimics
                mimic_pairwise_distances = cdist(mimic_coords_3d, mimic_coords_3d, metric='euclidean')
                # Get upper triangle (exclude diagonal and lower triangle)
                upper_triangle = np.triu(mimic_pairwise_distances, k=1)
                mimic_pairwise_distances_flat = upper_triangle[upper_triangle > 0]
                
                if len(mimic_pairwise_distances_flat) > 0:
                    median_mimic_distance = np.median(mimic_pairwise_distances_flat)
                    max_mimic_distance = np.max(mimic_pairwise_distances_flat)
                    # Use robust percentiles to exclude outliers
                    q90_mimic_distance = np.percentile(mimic_pairwise_distances_flat, 90)
                    q95_mimic_distance = np.percentile(mimic_pairwise_distances_flat, 95)
                else:
                    median_mimic_distance = avg_spread
                    max_mimic_distance = avg_spread
                    q90_mimic_distance = avg_spread
                    q95_mimic_distance = avg_spread
                
                # Define thresholds (more conservative, robust to outliers):
                # 1. "Within cloud": distance <= 75th percentile of mimic pairwise distances (tighter than median)
                # 2. "Just around cloud": distance <= 90th percentile (excludes extreme outliers) + buffer
                #    This prevents outlier mimics from inflating the cloud size
                if len(mimic_pairwise_distances_flat) > 0:
                    q75_mimic_distance = np.percentile(mimic_pairwise_distances_flat, 75)
                    # Use 75th percentile instead of median for tighter "within cloud"
                    threshold_within = q75_mimic_distance
                else:
                    threshold_within = median_mimic_distance
                
                # Use 90th percentile instead of max to exclude outlier mimics from cloud definition
                # This ensures the cloud is defined by the main cluster, not distant outliers
                # Add a small buffer (25%) for "around cloud" threshold
                buffer_factor = 0.25  # Add 25% buffer beyond 90th percentile distance
                threshold_around = q90_mimic_distance * (1 + buffer_factor)
                
                # Log the difference between max and 90th percentile to show if outliers were excluded
                if len(mimic_pairwise_distances_flat) > 0 and max_mimic_distance > q90_mimic_distance * 1.1:
                    print(f"  Note: Using 90th percentile ({q90_mimic_distance:.6f}) instead of max ({max_mimic_distance:.6f})")
                    print(f"    This excludes outlier mimics that are far from the main cluster")
                
                # Count samples
                samples_within_cloud = np.sum(min_dist_to_any_mimic <= threshold_within)
                samples_around_cloud = np.sum(min_dist_to_any_mimic <= threshold_around)
                samples_outside_cloud = len(results_df) - samples_around_cloud
                
                # Also compute statistics
                mimic_center = np.mean(mimic_coords_3d, axis=0)
                distances_to_center = np.sqrt(np.sum((all_coords_3d - mimic_center)**2, axis=1))
                
                summary_data = {
                    'mimic_queries': results_df.iloc[top_10_mimic_indices]['query'].tolist(),
                    'num_mimics': len(top_10_mimic_indices),
                    'mimic_coords': mimic_coords_3d,
                    'mimic_center': mimic_center,
                    'mimic_spread': mimic_spread,
                    'median_mimic_distance': median_mimic_distance,
                    'max_mimic_distance': max_mimic_distance,
                    'q90_mimic_distance': q90_mimic_distance if len(mimic_pairwise_distances_flat) > 0 else avg_spread,
                    'threshold_within': threshold_within,
                    'threshold_around': threshold_around,
                    'samples_within_cloud': samples_within_cloud,
                    'samples_around_cloud': samples_around_cloud,
                    'samples_outside_cloud': samples_outside_cloud,
                    'total_samples': len(results_df),
                    'min_dist_to_any_mimic': min_dist_to_any_mimic,
                    'distances_to_center': distances_to_center
                }
                
                # Write summary to file
                if is_multi_system:
                    summary_file = os.path.join(pca_test_dir, 'joint_diffusion_map_mimic_proximity_summary.txt')
                else:
                    summary_file = os.path.join(pca_test_dir, f'{organism_name}_diffusion_map_mimic_proximity_summary.txt')
                with open(summary_file, 'w') as f:
                    if is_multi_system:
                        f.write("Joint Diffusion Map Summary: Mimic Cloud in Unified Manifold\n")
                        f.write("="*80 + "\n\n")
                        f.write("MULTI-SYSTEM MODE: Cloud defined in JOINT DIFFUSION SPACE\n")
                        f.write("  This allows direct comparison of mimic regions across systems\n")
                        f.write("  All systems share the same diffusion manifold coordinates\n\n")
                    else:
                        f.write("Diffusion Map Summary: Samples within Mimic Cloud Region\n")
                        f.write("="*80 + "\n\n")
                    f.write(f"Total samples in analysis: {len(results_df)}\n")
                    if is_multi_system:
                        systems = results_df['system'].unique()
                        f.write(f"Systems: {', '.join(sorted(systems))}\n")
                        for system in sorted(systems):
                            system_mask = results_df['system'] == system
                            f.write(f"  {system}: {system_mask.sum()} samples\n")
                    f.write(f"Total known mimics found: {len(mimic_indices)}\n")
                    if is_multi_system:
                        # Show mimic breakdown by system
                        for system in sorted(systems):
                            system_mask = results_df['system'] == system
                            system_mimic_mask = system_mask & mimic_mask
                            if system_mimic_mask.sum() > 0:
                                f.write(f"  {system}: {system_mimic_mask.sum()} mimics\n")
                    f.write(f"Mimics used for cloud definition: {len(top_10_mimic_indices)} (core cluster only)\n")
                    if len(top_10_mimic_indices) < len(mimic_indices):
                        f.write(f"  Note: {len(mimic_indices) - len(top_10_mimic_indices)} outlier mimic(s) excluded from cloud definition\n\n")
                    else:
                        f.write("\n")
                    
                    f.write("Mimic Cloud Definition:\n")
                    if is_multi_system:
                        f.write("  - Cloud defined in JOINT DIFFUSION SPACE (not feature space)\n")
                        f.write("  - Uses diffusion coordinates from unified manifold across all systems\n")
                    f.write("  - We define a 'cloud' region around the CORE CLUSTER of known mimics\n")
                    f.write("  - Outlier mimics (far from main cluster) are excluded from cloud definition\n")
                    f.write("  - 'Within cloud': samples within 75th percentile of core mimic pairwise distances\n")
                    f.write("  - 'Around cloud': samples within 90th percentile + 25% buffer\n\n")
                    
                    f.write("-"*80 + "\n")
                    f.write("Mimic Cloud Statistics\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Number of mimics in cloud: {summary_data['num_mimics']}\n")
                    f.write(f"Mimic queries: {', '.join(summary_data['mimic_queries'])}\n\n")
                    
                    f.write(f"Cloud dimensions (spread of mimics):\n")
                    f.write(f"  DC1 spread: {summary_data['mimic_spread'][0]:.6f}\n")
                    f.write(f"  DC2 spread: {summary_data['mimic_spread'][1]:.6f}\n")
                    f.write(f"  DC3 spread: {summary_data['mimic_spread'][2]:.6f}\n")
                    f.write(f"  Average spread: {np.mean(summary_data['mimic_spread']):.6f}\n\n")
                    
                    f.write(f"Distances between mimics:\n")
                    f.write(f"  Median pairwise distance: {summary_data['median_mimic_distance']:.6f}\n")
                    f.write(f"  Maximum pairwise distance: {summary_data['max_mimic_distance']:.6f}\n")
                    f.write(f"  90th percentile distance: {summary_data['q90_mimic_distance']:.6f}\n")
                    if summary_data['max_mimic_distance'] > summary_data['q90_mimic_distance'] * 1.1:
                        ratio = summary_data['max_mimic_distance'] / summary_data['q90_mimic_distance']
                        f.write(f"  Note: Max distance is {ratio:.2f}x larger than 90th percentile\n")
                        f.write(f"    → Outlier mimics detected, using 90th percentile for cloud definition\n")
                    f.write("\n")
                    
                    f.write(f"Cloud boundaries:\n")
                    f.write(f"  'Within cloud' threshold: {summary_data['threshold_within']:.6f} (75th percentile of mimic distances)\n")
                    f.write(f"  'Around cloud' threshold: {summary_data['threshold_around']:.6f} (90th percentile + 25% buffer)\n\n")
                    
                    f.write("-"*80 + "\n")
                    f.write("Sample Counts\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Samples within cloud: {summary_data['samples_within_cloud']} / {summary_data['total_samples']} ({100.0*summary_data['samples_within_cloud']/summary_data['total_samples']:.2f}%)\n")
                    f.write(f"Samples around cloud (including within): {summary_data['samples_around_cloud']} / {summary_data['total_samples']} ({100.0*summary_data['samples_around_cloud']/summary_data['total_samples']:.2f}%)\n")
                    f.write(f"Samples outside cloud: {summary_data['samples_outside_cloud']} / {summary_data['total_samples']} ({100.0*summary_data['samples_outside_cloud']/summary_data['total_samples']:.2f}%)\n\n")
                    
                    if is_multi_system:
                        f.write("-"*80 + "\n")
                        f.write("Per-System Breakdown\n")
                        f.write("-"*80 + "\n")
                        systems = results_df['system'].unique()
                        within_mask = summary_data['min_dist_to_any_mimic'] <= summary_data['threshold_within']
                        around_mask = summary_data['min_dist_to_any_mimic'] <= summary_data['threshold_around']
                        outside_mask = ~around_mask
                        for system in sorted(systems):
                            system_mask = results_df['system'] == system
                            sys_within = (system_mask & within_mask).sum()
                            sys_around = (system_mask & around_mask).sum()
                            sys_outside = (system_mask & outside_mask).sum()
                            sys_total = system_mask.sum()
                            f.write(f"{system}:\n")
                            f.write(f"  Within cloud: {sys_within} / {sys_total} ({100.0*sys_within/sys_total:.2f}%)\n")
                            f.write(f"  Around cloud: {sys_around} / {sys_total} ({100.0*sys_around/sys_total:.2f}%)\n")
                            f.write(f"  Outside cloud: {sys_outside} / {sys_total} ({100.0*sys_outside/sys_total:.2f}%)\n\n")
                    
                    # Distance statistics
                    f.write("-"*80 + "\n")
                    f.write("Distance Statistics\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Distance from samples to nearest mimic:\n")
                    f.write(f"  Min: {np.min(summary_data['min_dist_to_any_mimic']):.6f}\n")
                    f.write(f"  25th percentile: {np.percentile(summary_data['min_dist_to_any_mimic'], 25):.6f}\n")
                    f.write(f"  Median: {np.median(summary_data['min_dist_to_any_mimic']):.6f}\n")
                    f.write(f"  75th percentile: {np.percentile(summary_data['min_dist_to_any_mimic'], 75):.6f}\n")
                    f.write(f"  Max: {np.max(summary_data['min_dist_to_any_mimic']):.6f}\n\n")
                    
                    f.write(f"Distance from samples to cloud center:\n")
                    f.write(f"  Min: {np.min(summary_data['distances_to_center']):.6f}\n")
                    f.write(f"  Median: {np.median(summary_data['distances_to_center']):.6f}\n")
                    f.write(f"  Max: {np.max(summary_data['distances_to_center']):.6f}\n")
                
                print(f"✓ Diffusion map proximity summary saved to: {summary_file}")
                
                # Create 3D visualization of the mimic cloud
                print("\nCreating 3D visualization of mimic cloud...")
                from mpl_toolkits.mplot3d import Axes3D
                
                # Classify samples based on distance to mimics
                within_mask = summary_data['min_dist_to_any_mimic'] <= summary_data['threshold_within']
                around_mask = summary_data['min_dist_to_any_mimic'] <= summary_data['threshold_around']
                outside_mask = ~around_mask
                
                # Create figure with multiple viewing angles
                viewing_angles = [
                    {'elev': 20, 'azim': 45, 'suffix': 'standard'},
                    {'elev': 20, 'azim': 135, 'suffix': 'rotated'},
                    {'elev': 90, 'azim': 0, 'suffix': 'top_down'},
                    {'elev': 0, 'azim': 0, 'suffix': 'side_view'}
                ]
                
                for view in viewing_angles:
                    fig = plt.figure(figsize=(16, 12) if is_multi_system else (14, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # In multi-system mode, color by system; otherwise use cloud membership
                    if is_multi_system:
                        # Color all points by system first
                        systems = results_df['system'].unique()
                        system_colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
                        system_color_map = dict(zip(systems, system_colors))
                        
                        for system in systems:
                            system_mask = results_df['system'] == system
                            system_coords = all_coords_3d[system_mask]
                            ax.scatter(system_coords[:, 0], 
                                     system_coords[:, 1], 
                                     system_coords[:, 2],
                                     c=[system_color_map[system]], s=8, alpha=0.4, 
                                     label=f'{system} (n={system_mask.sum()})', zorder=1)
                        
                        # Then overlay cloud regions with higher zorder
                        # Plot samples around cloud but not within (yellow/orange, higher alpha)
                        around_not_within = around_mask & ~within_mask
                        if around_not_within.sum() > 0:
                            ax.scatter(all_coords_3d[around_not_within, 0], 
                                     all_coords_3d[around_not_within, 1], 
                                     all_coords_3d[around_not_within, 2],
                                     c='orange', s=12, alpha=0.6, edgecolors='black', linewidths=0.5,
                                     label=f'Around cloud ({around_not_within.sum()})', zorder=3)
                        
                        # Plot samples within cloud (blue, higher alpha)
                        if within_mask.sum() > 0:
                            ax.scatter(all_coords_3d[within_mask, 0], 
                                     all_coords_3d[within_mask, 1], 
                                     all_coords_3d[within_mask, 2],
                                     c='blue', s=15, alpha=0.8, edgecolors='black', linewidths=1,
                                     label=f'Within cloud ({within_mask.sum()})', zorder=4)
                    else:
                        # Single-system mode: color by cloud membership
                        # Plot samples outside cloud (gray, low alpha)
                        if outside_mask.sum() > 0:
                            ax.scatter(all_coords_3d[outside_mask, 0], 
                                     all_coords_3d[outside_mask, 1], 
                                     all_coords_3d[outside_mask, 2],
                                     c='lightgray', s=5, alpha=0.2, label=f'Outside cloud ({outside_mask.sum()})', zorder=1)
                    
                        # Plot samples around cloud but not within (yellow/orange)
                        around_not_within = around_mask & ~within_mask
                        if around_not_within.sum() > 0:
                            ax.scatter(all_coords_3d[around_not_within, 0], 
                                     all_coords_3d[around_not_within, 1], 
                                     all_coords_3d[around_not_within, 2],
                                     c='orange', s=8, alpha=0.4, label=f'Around cloud ({around_not_within.sum()})', zorder=2)
                        
                        # Plot samples within cloud (blue)
                        if within_mask.sum() > 0:
                            ax.scatter(all_coords_3d[within_mask, 0], 
                                     all_coords_3d[within_mask, 1], 
                                     all_coords_3d[within_mask, 2],
                                     c='blue', s=10, alpha=0.6, label=f'Within cloud ({within_mask.sum()})', zorder=3)
                    
                    # Plot mimics (red, large, high zorder)
                    ax.scatter(summary_data['mimic_coords'][:, 0], 
                             summary_data['mimic_coords'][:, 1], 
                             summary_data['mimic_coords'][:, 2],
                             c='red', s=100, alpha=1.0, edgecolors='black', linewidths=2, 
                             label=f'Known mimics ({len(summary_data["mimic_queries"])})', zorder=10)
                    
                    # Plot cloud center (green star)
                    ax.scatter([summary_data['mimic_center'][0]], 
                             [summary_data['mimic_center'][1]], 
                             [summary_data['mimic_center'][2]],
                             c='green', marker='*', s=300, alpha=1.0, edgecolors='black', linewidths=1,
                             label='Cloud center', zorder=11)
                    
                    # Draw spheres representing cloud boundaries (optional, can be slow)
                    # Instead, draw bounding box or convex hull outline
                    try:
                        from scipy.spatial import ConvexHull
                        if len(summary_data['mimic_coords']) >= 4:
                            hull = ConvexHull(summary_data['mimic_coords'])
                            # Plot edges of convex hull
                            for simplex in hull.simplices:
                                ax.plot3D(summary_data['mimic_coords'][simplex, 0],
                                        summary_data['mimic_coords'][simplex, 1],
                                        summary_data['mimic_coords'][simplex, 2],
                                        'r--', alpha=0.3, linewidth=1, zorder=5)
                    except:
                        pass  # Skip if convex hull fails
                    
                    # Set labels and title
                    ax.set_xlabel('DC1', fontsize=12)
                    ax.set_ylabel('DC2', fontsize=12)
                    ax.set_zlabel('DC3', fontsize=12)
                    
                    if is_multi_system:
                        # Add system breakdown to title
                        title = f'Joint Diffusion Map: Mimic Cloud Across Systems\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n'
                        title += f'Total - Within: {within_mask.sum()} | Around: {around_not_within.sum()} | Outside: {outside_mask.sum()}\n'
                        # Add per-system breakdown
                        for system in systems:
                            system_mask = results_df['system'] == system
                            sys_within = (system_mask & within_mask).sum()
                            sys_around = (system_mask & around_not_within).sum()
                            sys_outside = (system_mask & outside_mask).sum()
                            title += f'{system}: W={sys_within} A={sys_around} O={sys_outside} | '
                        ax.set_title(title, fontsize=10)
                    else:
                        ax.set_title(f'Mimic Cloud in 3D Diffusion Space\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n' +
                                   f'Within: {within_mask.sum()} | Around: {around_not_within.sum()} | Outside: {outside_mask.sum()}', 
                                   fontsize=11)
                    
                    # Set viewing angle
                    ax.view_init(elev=view['elev'], azim=view['azim'])
                    
                    # Set robust axis limits
                    def compute_robust_axis_limits_3d(data, percentile_low=1.0, percentile_high=99.0, padding_factor=0.05):
                        if len(data) == 0:
                            return -1, 1
                        low_bound = np.percentile(data, percentile_low)
                        high_bound = np.percentile(data, percentile_high)
                        data_range = high_bound - low_bound
                        if data_range == 0:
                            padding = abs(low_bound * 0.1) if low_bound != 0 else 1.0
                            return low_bound - padding, high_bound + padding
                        padding = data_range * padding_factor
                        return low_bound - padding, high_bound + padding
                    
                    xlim = compute_robust_axis_limits_3d(all_coords_3d[:, 0])
                    ylim = compute_robust_axis_limits_3d(all_coords_3d[:, 1])
                    zlim = compute_robust_axis_limits_3d(all_coords_3d[:, 2])
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)
                    
                    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot
                    cloud_plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_mimic_cloud_3d_{view["suffix"]}.png')
                    plt.savefig(cloud_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Saved: {cloud_plot_file}")
                
                print("✓ Mimic cloud 3D visualizations complete")
            else:
                print("  (Skipping: Need at least 3 diffusion components for 3D analysis)")
        else:
            print("  (Skipping: No known mimics found in results)")
    else:
        print("  (Skipping: 'query' column not found in results)")
    
    # Visualize spectral clustering cloud if available
    if spectral_cloud_results is not None and 'query' in results_df.columns:
        print("\n" + "="*60)
        print("Creating visualization of spectral clustering mimic cloud")
        print("="*60)
        
        mimic_mask = results_df['query'].isin(validation_list).values
        mimic_indices = np.where(mimic_mask)[0]
        
        if len(mimic_indices) > 0:
            dc_cols = [col for col in results_df.columns if col.startswith('Diffusion')]
            if len(dc_cols) >= 3:
                all_coords_3d = results_df[dc_cols[:3]].values
                spectral_within = spectral_cloud_results['within_cloud']
                spectral_around = spectral_cloud_results['around_cloud']
                spectral_outside = ~spectral_around
                cluster_labels = spectral_cloud_results['cluster_labels']
                mimic_cluster_id = spectral_cloud_results['mimic_cluster_id']
                
                # Get mimic coordinates
                mimic_coords_3d = all_coords_3d[mimic_indices]
                mimic_center = np.mean(mimic_coords_3d, axis=0)
                
                # Calculate distances of mimics from cluster centroid to identify core vs outliers
                from scipy.spatial.distance import cdist
                mimic_distances_to_center = np.linalg.norm(mimic_coords_3d - mimic_center, axis=1)
                median_mimic_distance = np.median(mimic_distances_to_center)
                q75_mimic_distance = np.percentile(mimic_distances_to_center, 75)
                
                # Classify mimics as core vs outliers
                core_mimic_mask = mimic_distances_to_center <= median_mimic_distance
                outlier_mimic_mask = mimic_distances_to_center > q75_mimic_distance
                
                print(f"  Mimic distribution analysis:")
                print(f"    Total mimics: {len(mimic_indices)}")
                print(f"    Core mimics (≤ median distance): {core_mimic_mask.sum()}")
                print(f"    Outlier mimics (> 75th percentile): {outlier_mimic_mask.sum()}")
                print(f"    Median distance from center: {median_mimic_distance:.4f}")
                print(f"    75th percentile distance: {q75_mimic_distance:.4f}")
                
                # Check if multi-system
                is_multi_system = 'system' in results_df.columns
                
                viewing_angles = [
                    {'elev': 20, 'azim': 45, 'suffix': 'standard'},
                    {'elev': 20, 'azim': 135, 'suffix': 'rotated'},
                    {'elev': 90, 'azim': 0, 'suffix': 'top_down'},
                    {'elev': 0, 'azim': 0, 'suffix': 'side_view'},
                    {'elev': 30, 'azim': 60, 'suffix': 'elevated'},
                    {'elev': 20, 'azim': 225, 'suffix': 'diagonal1'},
                    {'elev': 20, 'azim': 315, 'suffix': 'diagonal2'},
                    {'elev': 0, 'azim': 90, 'suffix': 'yz_plane'}
                ]
                
                for view in viewing_angles:
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    if is_multi_system:
                        # Multi-system mode: color by system first
                        systems = results_df['system'].unique()
                        system_colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
                        system_color_map = dict(zip(systems, system_colors))
                        
                        for system in systems:
                            system_mask = results_df['system'] == system
                            system_coords = all_coords_3d[system_mask]
                            ax.scatter(system_coords[:, 0], 
                                     system_coords[:, 1], 
                                     system_coords[:, 2],
                                     c=[system_color_map[system]], s=8, alpha=0.4, 
                                     label=f'{system} (n={system_mask.sum()})', zorder=1)
                        
                        # Overlay spectral cloud regions
                        around_not_within = spectral_around & ~spectral_within
                        if around_not_within.sum() > 0:
                            ax.scatter(all_coords_3d[around_not_within, 0], 
                                     all_coords_3d[around_not_within, 1], 
                                     all_coords_3d[around_not_within, 2],
                                     c='orange', s=12, alpha=0.6, edgecolors='black', linewidths=0.5,
                                     label=f'Spectral around cloud ({around_not_within.sum()})', zorder=3)
                        
                        if spectral_within.sum() > 0:
                            ax.scatter(all_coords_3d[spectral_within, 0], 
                                     all_coords_3d[spectral_within, 1], 
                                     all_coords_3d[spectral_within, 2],
                                     c='blue', s=15, alpha=0.8, edgecolors='black', linewidths=1,
                                     label=f'Spectral within cloud ({spectral_within.sum()})', zorder=4)
                    else:
                        # Single-system mode
                        if spectral_outside.sum() > 0:
                            ax.scatter(all_coords_3d[spectral_outside, 0], 
                                     all_coords_3d[spectral_outside, 1], 
                                     all_coords_3d[spectral_outside, 2],
                                     c='lightgray', s=5, alpha=0.2, label=f'Outside spectral cloud ({spectral_outside.sum()})', zorder=1)
                        
                        around_not_within = spectral_around & ~spectral_within
                        if around_not_within.sum() > 0:
                            ax.scatter(all_coords_3d[around_not_within, 0], 
                                     all_coords_3d[around_not_within, 1], 
                                     all_coords_3d[around_not_within, 2],
                                     c='orange', s=8, alpha=0.4, label=f'Spectral around cloud ({around_not_within.sum()})', zorder=2)
                        
                        if spectral_within.sum() > 0:
                            ax.scatter(all_coords_3d[spectral_within, 0], 
                                     all_coords_3d[spectral_within, 1], 
                                     all_coords_3d[spectral_within, 2],
                                     c='blue', s=10, alpha=0.6, label=f'Spectral within cloud ({spectral_within.sum()})', zorder=3)
                    
                    # Plot mimics with color coding: core (red) vs outliers (orange)
                    if core_mimic_mask.sum() > 0:
                        core_coords = mimic_coords_3d[core_mimic_mask]
                        ax.scatter(core_coords[:, 0], 
                                 core_coords[:, 1], 
                                 core_coords[:, 2],
                                 c='red', s=100, alpha=1.0, edgecolors='black', linewidths=2, 
                                 label=f'Core mimics ({core_mimic_mask.sum()})', zorder=10)
                    
                    if outlier_mimic_mask.sum() > 0:
                        outlier_coords = mimic_coords_3d[outlier_mimic_mask]
                        ax.scatter(outlier_coords[:, 0], 
                                 outlier_coords[:, 1], 
                                 outlier_coords[:, 2],
                                 c='orange', s=120, alpha=1.0, edgecolors='black', linewidths=2, 
                                 marker='^', label=f'Outlier mimics ({outlier_mimic_mask.sum()})', zorder=11)
                    
                    # Plot intermediate mimics (between core and outlier)
                    intermediate_mask = ~core_mimic_mask & ~outlier_mimic_mask
                    if intermediate_mask.sum() > 0:
                        intermediate_coords = mimic_coords_3d[intermediate_mask]
                        ax.scatter(intermediate_coords[:, 0], 
                                 intermediate_coords[:, 1], 
                                 intermediate_coords[:, 2],
                                 c='darkred', s=100, alpha=0.8, edgecolors='black', linewidths=1.5, 
                                 label=f'Intermediate mimics ({intermediate_mask.sum()})', zorder=10)
                    
                    # Plot cluster center (green star)
                    ax.scatter([mimic_center[0]], 
                             [mimic_center[1]], 
                             [mimic_center[2]],
                             c='green', marker='*', s=300, alpha=1.0, edgecolors='black', linewidths=1,
                             label='Mimic center', zorder=11)
                    
                    # Draw convex hull of mimics
                    try:
                        from scipy.spatial import ConvexHull
                        if len(mimic_coords_3d) >= 4:
                            hull = ConvexHull(mimic_coords_3d)
                            for simplex in hull.simplices:
                                ax.plot3D(mimic_coords_3d[simplex, 0],
                                        mimic_coords_3d[simplex, 1],
                                        mimic_coords_3d[simplex, 2],
                                        'r--', alpha=0.3, linewidth=1, zorder=5)
                    except:
                        pass
                    
                    # Set labels and title
                    ax.set_xlabel('DC1', fontsize=12)
                    ax.set_ylabel('DC2', fontsize=12)
                    ax.set_zlabel('DC3', fontsize=12)
                    
                    if is_multi_system:
                        title = f'Spectral Clustering: Mimic Cloud Across Systems\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n'
                        title += f'Total - Within: {spectral_within.sum()} | Outside: {spectral_outside.sum()}\n'
                        title += f'Mimic Cluster ID: {mimic_cluster_id} | Total Clusters: {spectral_cloud_results["n_clusters"]}\n'
                        title += f'Mimics: Core={core_mimic_mask.sum()} | Outlier={outlier_mimic_mask.sum()} | Intermediate={intermediate_mask.sum()}\n'
                        for system in systems:
                            system_mask = results_df['system'] == system
                            sys_within = (system_mask & spectral_within).sum()
                            sys_outside = (system_mask & spectral_outside).sum()
                            title += f'{system}: W={sys_within} O={sys_outside} | '
                        ax.set_title(title, fontsize=10)
                    else:
                        ax.set_title(f'Spectral Clustering: Mimic Cloud in 3D Diffusion Space\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n' +
                                   f'Within: {spectral_within.sum()} | Outside: {spectral_outside.sum()}\n' +
                                   f'Mimic Cluster ID: {mimic_cluster_id} | Total Clusters: {spectral_cloud_results["n_clusters"]}\n' +
                                   f'Mimics: Core={core_mimic_mask.sum()} | Outlier={outlier_mimic_mask.sum()} | Intermediate={intermediate_mask.sum()}', 
                                   fontsize=11)
                    
                    ax.view_init(elev=view['elev'], azim=view['azim'])
                    
                    # Set robust axis limits
                    def compute_robust_axis_limits_3d(data, percentile_low=1.0, percentile_high=99.0, padding_factor=0.05):
                        if len(data) == 0:
                            return -1, 1
                        low_bound = np.percentile(data, percentile_low)
                        high_bound = np.percentile(data, percentile_high)
                        data_range = high_bound - low_bound
                        if data_range == 0:
                            padding = abs(low_bound * 0.1) if low_bound != 0 else 1.0
                            return low_bound - padding, high_bound + padding
                        padding = data_range * padding_factor
                        return low_bound - padding, high_bound + padding
                    
                    xlim = compute_robust_axis_limits_3d(all_coords_3d[:, 0])
                    ylim = compute_robust_axis_limits_3d(all_coords_3d[:, 1])
                    zlim = compute_robust_axis_limits_3d(all_coords_3d[:, 2])
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)
                    
                    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot
                    spectral_plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_spectral_cloud_3d_{view["suffix"]}.png')
                    plt.savefig(spectral_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Saved: {spectral_plot_file}")
                
                print("✓ Spectral clustering cloud 3D visualizations complete")
                
                # Also create 2D plot
                if len(dc_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    if is_multi_system:
                        systems = results_df['system'].unique()
                        system_colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
                        system_color_map = dict(zip(systems, system_colors))
                        
                        for system in systems:
                            system_mask = results_df['system'] == system
                            system_coords = results_df.loc[system_mask, dc_cols[:2]].values
                            ax.scatter(system_coords[:, 0], system_coords[:, 1],
                                     c=[system_color_map[system]], s=8, alpha=0.4, 
                                     label=f'{system} (n={system_mask.sum()})')
                        
                        if spectral_within.sum() > 0:
                            ax.scatter(results_df.loc[spectral_within, dc_cols[0]], 
                                     results_df.loc[spectral_within, dc_cols[1]],
                                     c='blue', s=15, alpha=0.8, edgecolors='black', linewidths=1,
                                     label=f'Spectral within cloud ({spectral_within.sum()})', zorder=4)
                    else:
                        if spectral_outside.sum() > 0:
                            ax.scatter(results_df.loc[spectral_outside, dc_cols[0]], 
                                     results_df.loc[spectral_outside, dc_cols[1]],
                                     c='lightgray', s=5, alpha=0.2, label=f'Outside ({spectral_outside.sum()})')
                        
                        if spectral_within.sum() > 0:
                            ax.scatter(results_df.loc[spectral_within, dc_cols[0]], 
                                     results_df.loc[spectral_within, dc_cols[1]],
                                     c='blue', s=10, alpha=0.6, label=f'Within cloud ({spectral_within.sum()})')
                    
                    # Plot mimics with color coding: core (red) vs outliers (orange)
                    mimic_coords_2d = results_df.iloc[mimic_indices][dc_cols[:2]].values
                    mimic_center_2d = np.mean(mimic_coords_2d, axis=0)
                    
                    # Recalculate 2D distances for consistency
                    mimic_distances_2d = np.linalg.norm(mimic_coords_2d - mimic_center_2d, axis=1)
                    median_mimic_distance_2d = np.median(mimic_distances_2d)
                    q75_mimic_distance_2d = np.percentile(mimic_distances_2d, 75)
                    core_mimic_mask_2d = mimic_distances_2d <= median_mimic_distance_2d
                    outlier_mimic_mask_2d = mimic_distances_2d > q75_mimic_distance_2d
                    intermediate_mask_2d = ~core_mimic_mask_2d & ~outlier_mimic_mask_2d
                    
                    if core_mimic_mask_2d.sum() > 0:
                        core_coords_2d = mimic_coords_2d[core_mimic_mask_2d]
                        ax.scatter(core_coords_2d[:, 0], core_coords_2d[:, 1],
                                 c='red', s=100, alpha=1.0, edgecolors='black', linewidths=2,
                                 label=f'Core mimics ({core_mimic_mask_2d.sum()})', zorder=10)
                    
                    if outlier_mimic_mask_2d.sum() > 0:
                        outlier_coords_2d = mimic_coords_2d[outlier_mimic_mask_2d]
                        ax.scatter(outlier_coords_2d[:, 0], outlier_coords_2d[:, 1],
                                 c='orange', s=120, alpha=1.0, edgecolors='black', linewidths=2,
                                 marker='^', label=f'Outlier mimics ({outlier_mimic_mask_2d.sum()})', zorder=11)
                    
                    if intermediate_mask_2d.sum() > 0:
                        intermediate_coords_2d = mimic_coords_2d[intermediate_mask_2d]
                        ax.scatter(intermediate_coords_2d[:, 0], intermediate_coords_2d[:, 1],
                                 c='darkred', s=100, alpha=0.8, edgecolors='black', linewidths=1.5,
                                 label=f'Intermediate mimics ({intermediate_mask_2d.sum()})', zorder=10)
                    
                    ax.scatter([mimic_center_2d[0]], [mimic_center_2d[1]],
                             c='green', marker='*', s=300, alpha=1.0, edgecolors='black', linewidths=1,
                             label='Mimic center', zorder=11)
                    
                    ax.set_xlabel('DC1', fontsize=12)
                    ax.set_ylabel('DC2', fontsize=12)
                    ax.set_title(f'Spectral Clustering: Mimic Cloud in 2D Diffusion Space\n' +
                               f'Within: {spectral_within.sum()} | Outside: {spectral_outside.sum()}\n' +
                               f'Mimic Cluster ID: {mimic_cluster_id} | Total Clusters: {spectral_cloud_results["n_clusters"]}\n' +
                               f'Mimics: Core={core_mimic_mask_2d.sum()} | Outlier={outlier_mimic_mask_2d.sum()} | Intermediate={intermediate_mask_2d.sum()}',
                               fontsize=11)
                    ax.legend(loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    spectral_2d_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_spectral_cloud_2d.png')
                    plt.savefig(spectral_2d_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Saved: {spectral_2d_file}")
            else:
                print("  (Skipping: Need at least 3 diffusion components for 3D visualization)")
        else:
            print("  (Skipping: No known mimics found in results)")
    elif spectral_cloud_results is not None:
        print("  (Skipping spectral cloud visualization: 'query' column not found in results)")
    
    # Visualize transferred cloud (works even without local mimics)
    if transferred_cloud_results is not None:
        print("\n" + "="*60)
        print("Creating visualization of transferred mimic cloud")
        print("="*60)
        
        # Get diffusion coordinates
        dc_cols = [col for col in results_df.columns if col.startswith('Diffusion')]
        if len(dc_cols) >= 3:
            all_coords_3d = results_df[dc_cols[:3]].values
            transferred_within = transferred_cloud_results['within_cloud']
            transferred_around = transferred_cloud_results['around_cloud']
            transferred_outside = ~transferred_around
            
            # Check if we have local mimics for comparison
            has_local_mimics = 'query' in results_df.columns and len(results_df[results_df['query'].isin(validation_list)]) > 0
            if has_local_mimics and 'feature_space_within_cloud' in results_df.columns:
                local_within = results_df['feature_space_within_cloud'].values
            else:
                local_within = None
            
            viewing_angles = [
                {'elev': 20, 'azim': 45, 'suffix': 'standard'},
                {'elev': 20, 'azim': 135, 'suffix': 'rotated'},
                {'elev': 90, 'azim': 0, 'suffix': 'top_down'},
                {'elev': 0, 'azim': 0, 'suffix': 'side_view'}
            ]
            
            for view in viewing_angles:
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot samples outside transferred cloud (gray)
                if transferred_outside.sum() > 0:
                    ax.scatter(all_coords_3d[transferred_outside, 0], 
                             all_coords_3d[transferred_outside, 1], 
                             all_coords_3d[transferred_outside, 2],
                             c='lightgray', s=3, alpha=0.15, label=f'Outside transferred cloud ({transferred_outside.sum()})', zorder=1)
                
                # Plot samples around transferred cloud but not within (orange)
                transferred_around_not_within = transferred_around & ~transferred_within
                if transferred_around_not_within.sum() > 0:
                    ax.scatter(all_coords_3d[transferred_around_not_within, 0], 
                             all_coords_3d[transferred_around_not_within, 1], 
                             all_coords_3d[transferred_around_not_within, 2],
                             c='orange', s=6, alpha=0.4, label=f'Around transferred cloud ({transferred_around_not_within.sum()})', zorder=2)
                
                # Plot samples within transferred cloud (purple/blue)
                if transferred_within.sum() > 0:
                    color = 'purple' if local_within is None else 'purple'
                    label = f'Within transferred cloud ({transferred_within.sum()})'
                    if local_within is not None:
                        # Show comparison
                        local_only = local_within & ~transferred_within
                        transferred_only = transferred_within & ~local_within
                        in_both = local_within & transferred_within
                        
                        if local_only.sum() > 0:
                            ax.scatter(all_coords_3d[local_only, 0], 
                                     all_coords_3d[local_only, 1], 
                                     all_coords_3d[local_only, 2],
                                     c='blue', s=8, alpha=0.5, label=f'Local cloud only ({local_only.sum()})', zorder=3)
                        
                        if transferred_only.sum() > 0:
                            ax.scatter(all_coords_3d[transferred_only, 0], 
                                     all_coords_3d[transferred_only, 1], 
                                     all_coords_3d[transferred_only, 2],
                                     c='purple', s=8, alpha=0.5, label=f'Transferred cloud only ({transferred_only.sum()})', zorder=4)
                        
                        if in_both.sum() > 0:
                            ax.scatter(all_coords_3d[in_both, 0], 
                                     all_coords_3d[in_both, 1], 
                                     all_coords_3d[in_both, 2],
                                     c='cyan', s=12, alpha=0.7, label=f'In both clouds ({in_both.sum()})', zorder=5)
                    else:
                        # No local cloud, just show transferred
                        ax.scatter(all_coords_3d[transferred_within, 0], 
                                 all_coords_3d[transferred_within, 1], 
                                 all_coords_3d[transferred_within, 2],
                                 c='purple', s=10, alpha=0.6, label=label, zorder=3)
                
                # Label candidate samples by query ID, colored by cloud membership (within/around/outside)
                if 'query' in results_df.columns:
                    candidate_mask = results_df['query'].isin(candidate_list).values
                    if candidate_mask.sum() > 0:
                        candidates_within = candidate_mask & transferred_within
                        candidates_around_only = candidate_mask & transferred_around & ~transferred_within
                        candidates_outside = candidate_mask & transferred_outside

                        def _plot_and_annotate_candidates(mask, color, label_prefix):
                            idxs = np.where(mask)[0]
                            if len(idxs) == 0:
                                return
                            # Scatter all candidate points in this category
                            ax.scatter(all_coords_3d[idxs, 0],
                                       all_coords_3d[idxs, 1],
                                       all_coords_3d[idxs, 2],
                                       c=color, s=50, alpha=0.9,
                                       edgecolors='black', linewidths=1.0,
                                       label=f'{label_prefix} candidates ({len(idxs)})',
                                       zorder=8)
                            # Add text labels with query IDs
                            for idx in idxs:
                                x, y, z = all_coords_3d[idx]
                                qid = str(results_df.iloc[idx]['query'])
                                ax.text(x, y, z, qid,
                                        fontsize=7,
                                        color='black',
                                        ha='center', va='center',
                                        bbox=dict(boxstyle='round,pad=0.2',
                                                  facecolor=color,
                                                  alpha=0.85,
                                                  edgecolor='black'),
                                        zorder=9)

                        _plot_and_annotate_candidates(candidates_within, 'deepskyblue', 'Within transferred cloud')
                        _plot_and_annotate_candidates(candidates_around_only, 'gold', 'Around transferred cloud')
                        _plot_and_annotate_candidates(candidates_outside, 'darkgray', 'Outside transferred cloud')
                
                # Plot known mimics if they exist
                if 'query' in results_df.columns:
                    mimic_mask = results_df['query'].isin(validation_list).values
                    if mimic_mask.sum() > 0:
                        mimic_coords_3d = all_coords_3d[mimic_mask]
                        ax.scatter(mimic_coords_3d[:, 0], 
                                 mimic_coords_3d[:, 1], 
                                 mimic_coords_3d[:, 2],
                                 c='red', s=150, alpha=1.0, edgecolors='black', linewidths=2, 
                                 label=f'Known mimics ({mimic_mask.sum()})', zorder=10)
                
                ax.set_xlabel('DC1', fontsize=12)
                ax.set_ylabel('DC2', fontsize=12)
                ax.set_zlabel('DC3', fontsize=12)
                
                if local_within is not None:
                    title = f'Transferred vs Local Mimic Cloud Comparison\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n'
                    if 'in_both' in locals():
                        title += f'Transferred: {transferred_within.sum()} | Local: {local_within.sum()} | Both: {in_both.sum()}'
                    else:
                        title += f'Transferred: {transferred_within.sum()} | Local: {local_within.sum()}'
                else:
                    title = f'Transferred Mimic Cloud\n(View: elev={view["elev"]}°, azim={view["azim"]}°)\n'
                    title += f'Within: {transferred_within.sum()} | Around: {transferred_around_not_within.sum()} | Outside: {transferred_outside.sum()}'
                
                ax.set_title(title, fontsize=11)
                ax.view_init(elev=view['elev'], azim=view['azim'])
                
                # Set robust axis limits
                xlim_low, xlim_high = compute_robust_axis_limits(all_coords_3d[:, 0], percentile_low=1.0, percentile_high=99.0)
                ylim_low, ylim_high = compute_robust_axis_limits(all_coords_3d[:, 1], percentile_low=1.0, percentile_high=99.0)
                zlim_low, zlim_high = compute_robust_axis_limits(all_coords_3d[:, 2], percentile_low=1.0, percentile_high=99.0)
                ax.set_xlim(xlim_low, xlim_high)
                ax.set_ylim(ylim_low, ylim_high)
                ax.set_zlim(zlim_low, zlim_high)
                
                ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                ax.grid(True, alpha=0.3)
                
                # Save plot
                if local_within is not None:
                    comparison_plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_transferred_cloud_comparison_3d_{view["suffix"]}.png')
                else:
                    comparison_plot_file = os.path.join(pca_test_dir, f'diffusion_map_{organism_name}_transferred_cloud_3d_{view["suffix"]}.png')
                plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: {comparison_plot_file}")
            
            print("✓ Transferred cloud visualizations complete")
        else:
            print("  (Skipping: Need at least 3 diffusion components for 3D visualization)")
    else:
        print("  (Skipping: Need at least 3 diffusion components for 3D analysis)")
    
    print("\n" + "="*60)
    
    return results_df


def estimate_optimal_clusters(reduced_data, clustering_method='kmeans', k_range=None, method='silhouette'):
    """
    Estimate optimal number of clusters using silhouette score or elbow method.
    
    Args:
        reduced_data: Data to cluster (already reduced dimensionality)
        clustering_method: 'kmeans' or 'agglomerative'
        k_range: Range of k values to test (default: 2 to min(20, n_samples//10))
        method: 'silhouette' (maximize) or 'elbow' (minimize WCSS)
    
    Returns:
        optimal_k: Best number of clusters
        scores: Dictionary with scores for each k
    """
    n_samples = len(reduced_data)
    
    if k_range is None:
        # Reasonable range: 2 to min(20, n_samples//10, or 50% of samples)
        max_k = min(20, max(2, n_samples // 10))
        k_range = range(2, max_k + 1)
    else:
        k_range = range(k_range[0], k_range[1] + 1)
    
    print(f"\nEstimating optimal number of clusters (testing k={min(k_range)} to {max(k_range)})...")
    print(f"Method: {method}")
    
    scores = {}
    wcss_scores = {}  # For elbow method
    
    for k in k_range:
        try:
            if clustering_method == 'kmeans':
                clusterer = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            elif clustering_method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=k)
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method}")
            
            cluster_labels = clusterer.fit_predict(reduced_data)
            
            # Compute silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                sil_score = silhouette_score(reduced_data, cluster_labels)
                scores[k] = sil_score
                
                # For elbow method, compute WCSS (within-cluster sum of squares)
                if method == 'elbow' and clustering_method == 'kmeans':
                    wcss = clusterer.inertia_
                    wcss_scores[k] = wcss
                
                print(f"  k={k:2d}: Silhouette={sil_score:.4f}", end="")
                if method == 'elbow' and clustering_method == 'kmeans':
                    print(f", WCSS={wcss:.2f}")
                else:
                    print()
            else:
                print(f"  k={k:2d}: Skipped (only 1 cluster)")
        except Exception as e:
            print(f"  k={k:2d}: Error - {e}")
            continue
    
    if not scores:
        print("Warning: Could not compute scores for any k. Using default k=15.")
        return 15, {}
    
    if method == 'silhouette':
        # Maximize silhouette score
        optimal_k = max(scores, key=scores.get)
        optimal_score = scores[optimal_k]
        print(f"\nOptimal k={optimal_k} (silhouette score: {optimal_score:.4f})")
    elif method == 'elbow':
        if not wcss_scores:
            # Fallback to silhouette if elbow not available
            optimal_k = max(scores, key=scores.get)
            optimal_score = scores[optimal_k]
            print(f"\nOptimal k={optimal_k} (silhouette score: {optimal_score:.4f}, elbow method not available)")
        else:
            # Find elbow: look for largest decrease in WCSS
            k_list = sorted(wcss_scores.keys())
            if len(k_list) >= 2:
                # Compute rate of change (second derivative approximation)
                decreases = []
                for i in range(1, len(k_list)):
                    k_prev = k_list[i-1]
                    k_curr = k_list[i]
                    wcss_prev = wcss_scores[k_prev]
                    wcss_curr = wcss_scores[k_curr]
                    decrease = wcss_prev - wcss_curr
                    decreases.append((k_curr, decrease))
                
                # Find point with largest decrease (elbow)
                optimal_k, max_decrease = max(decreases, key=lambda x: x[1])
                print(f"\nOptimal k={optimal_k} (elbow method, WCSS decrease: {max_decrease:.2f})")
            else:
                optimal_k = k_list[0]
                print(f"\nOptimal k={optimal_k} (only one k tested)")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return optimal_k, scores


def value_driven_pca_clustering(pca_test_dir, data_frame, organism_name, n_clusters=10, clustering_method='kmeans', include_lengths=False, plot_3d=False, use_umap=False, umap_n_neighbors=15, umap_min_dist=0.1, include_evolutionary=False, include_targeting=False, include_plddt=False, include_go_terms=False, scaler_type='robust', use_diffusion=False, diffusion_alpha=1.0, diffusion_n_neighbors=15, exclude_derived_features=False, auto_clusters=False, cluster_estimation_method='silhouette', use_natural_attractor=False, save_cloud_definition=False, transfer_cloud_from=None, adjust_tm_score=False, use_adaptive_epsilon=False, adaptive_k_neighbors=7, auto_n_neighbors=False, use_spectral_cloud=False, spectral_n_clusters=None, spectral_gamma=None, epsilon_scale=1.0):
    """
    Perform purely value-driven clustering based on alignment features only.
    No GO term labels are used - clustering is based solely on alignment feature values.
    This is for discovering new mimic candidates, not validating known ones.
    
    Args:
        include_lengths: If False, exclude qlen and tlen (absolute lengths) and use only ratio features.
                         Default False to avoid mixing absolute values with ratios (0-1 scale).
        plot_3d: If True, create 3D visualization (PC1, PC2, PC3) in addition to 2D.
    """
    # Select specific columns for analysis
    selected_columns = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction']
    
    if include_lengths:
        selected_columns.extend(['qlen', 'tlen'])
    
    # Add evolutionary features if requested
    if include_evolutionary:
        evo_cols = []
        if 'symbiont_branch_dnds_avg' in data_frame.columns:
            evo_cols.append('symbiont_branch_dnds_avg')
        if 'non_symbiont_branch_dnds_avg' in data_frame.columns:
            evo_cols.append('non_symbiont_branch_dnds_avg')
        if 'test_fraction' in data_frame.columns:
            evo_cols.append('test_fraction')
        
        if len(evo_cols) > 0:
            # Check for missing values
            print(f"\nEvolutionary features analysis:")
            for col in evo_cols:
                missing = data_frame[col].isna().sum() + (data_frame[col] == '').sum()
                total = len(data_frame)
                pct_missing = missing / total * 100
                print(f"  {col}: {missing}/{total} missing ({pct_missing:.1f}%)")
            
            # Check how many rows have ALL evorates vs some missing
            evo_subset = data_frame[evo_cols]
            all_missing = evo_subset.isna().all(axis=1).sum()
            some_missing = evo_subset.isna().any(axis=1).sum() - all_missing
            complete = (~evo_subset.isna().any(axis=1)).sum()
            
            print(f"  Rows with all evorates: {complete} ({complete/total*100:.1f}%)")
            print(f"  Rows with some missing: {some_missing} ({some_missing/total*100:.1f}%)")
            print(f"  Rows with all missing: {all_missing} ({all_missing/total*100:.1f}%)")
            
            if all_missing / total > 0.5:
                print(f"  WARNING: >50% of rows missing all evorates. Consider not using evolutionary features.")
            elif complete / total < 0.3:
                print(f"  WARNING: <30% of rows have complete evorate data. May need imputation or filtering.")
            
            selected_columns.extend(evo_cols)
            print(f"  Added evolutionary features: {evo_cols}")
    
    # Add targeting features if requested
    if include_targeting:
        target_cols = []
        # Only include SP probability (not mTP)
        if 'query_SP_probability' in data_frame.columns:
            target_cols.append('query_SP_probability')
        selected_columns.extend(target_cols)
        if len(target_cols) > 0:
            print(f"Added targeting features: {target_cols}")
        elif 'query_SP_probability' not in data_frame.columns:
            print("Warning: query_SP_probability column not found")
    
    # Add pLDDT features if requested
    if include_plddt:
        plddt_cols = []
        if 'plddt_query_region' in data_frame.columns:
            plddt_cols.append('plddt_query_region')
        if 'plddt_target_region' in data_frame.columns:
            plddt_cols.append('plddt_target_region')
        
        if len(plddt_cols) > 0:
            # Check for missing values
            print(f"\npLDDT features analysis:")
            for col in plddt_cols:
                missing = data_frame[col].isna().sum() + (data_frame[col] == '').sum()
                total = len(data_frame)
                pct_missing = missing / total * 100
                print(f"  {col}: {missing}/{total} missing ({pct_missing:.1f}%)")
            
            selected_columns.extend(plddt_cols)
            print(f"  Added pLDDT features: {plddt_cols}")
        else:
            print("Warning: pLDDT columns (plddt_query_region, plddt_target_region) not found")
    
    # Adjust TM-score by pLDDT if requested
    if adjust_tm_score:
        if 'plddt_query_region' in data_frame.columns and 'plddt_target_region' in data_frame.columns and 'score' in data_frame.columns:
            print("\nAdjusting TM-score by pLDDT confidence...")
            # Convert pLDDT to numeric, handling 'NA' strings and missing values
            plddt_query = pd.to_numeric(data_frame['plddt_query_region'], errors='coerce')
            plddt_target = pd.to_numeric(data_frame['plddt_target_region'], errors='coerce')
            
            # Use minimum confidence (most conservative - penalizes if either structure is uncertain)
            min_plddt = pd.concat([plddt_query, plddt_target], axis=1).min(axis=1)
            
            # Normalize pLDDT to 0-1 scale (pLDDT is typically 0-100)
            plddt_normalized = min_plddt / 100.0
            
            # Weight TM-score by confidence: higher confidence = higher weight
            # Store in dataframe for later use
            data_frame['score_weighted_by_plddt'] = data_frame['score'] * plddt_normalized
            
            # Replace 'score' with 'score_weighted_by_plddt' in selected columns
            if 'score' in selected_columns:
                selected_columns.remove('score')
                selected_columns.append('score_weighted_by_plddt')
                print("  Replaced 'score' with 'score_weighted_by_plddt' in feature set")
                print("  This downweights alignments with low structure confidence in diffusion mapping")
        else:
            print("Warning: --adjust_tm_score requires 'score', 'plddt_query_region', and 'plddt_target_region' columns")
            print("  Falling back to standard 'score' feature")
    
    # Add derived features
    test_data_frame = data_frame[selected_columns].copy()
    
    # Check for and handle any remaining missing values (shouldn't happen if filtering/imputation worked)
    missing_before = test_data_frame.isna().sum().sum()
    if missing_before > 0:
        print(f"\nWarning: {missing_before} missing values found in feature columns")
        print("  Missing values per column:")
        for col in test_data_frame.columns:
            missing = test_data_frame[col].isna().sum()
            if missing > 0:
                print(f"    {col}: {missing} ({missing/len(test_data_frame)*100:.1f}%)")
        
        # If we have missing values, we need to handle them
        # Option 1: Drop rows with any missing (conservative)
        initial_rows = len(test_data_frame)
        test_data_frame = test_data_frame.dropna()
        rows_after = len(test_data_frame)
        if rows_after < initial_rows:
            print(f"  Dropped {initial_rows - rows_after} rows with missing values")
            print(f"  Remaining: {rows_after} rows ({rows_after/initial_rows*100:.1f}%)")
            # Also need to update data_frame to match - reset index to ensure alignment
            data_frame = data_frame.loc[test_data_frame.index].copy().reset_index(drop=True)
            test_data_frame = test_data_frame.reset_index(drop=True)
        
        # Update selected_columns to only include columns that exist
        selected_columns = [col for col in selected_columns if col in test_data_frame.columns]
    
    # Create derived features if not excluded and we have the base columns
    derived_features_added = []
    if not exclude_derived_features:
        # score_per_length: normalized score by protein sizes
        # Use the active score column (either 'score' or 'score_weighted_by_plddt' if adjusted)
        score_col = 'score_weighted_by_plddt' if adjust_tm_score and 'score_weighted_by_plddt' in data_frame.columns else 'score'
        if 'qlen' in data_frame.columns and 'tlen' in data_frame.columns and score_col in data_frame.columns:
            test_data_frame['score_per_length'] = data_frame[score_col] / (data_frame['qlen'] * data_frame['tlen'] + 1e-10)
            derived_features_added.append('score_per_length')
        
        # coverage_balance: asymmetry between query and target coverage
        if 'tcov' in data_frame.columns and 'qcov' in data_frame.columns:
            test_data_frame['coverage_balance'] = abs(data_frame['tcov'] - data_frame['qcov'])
            derived_features_added.append('coverage_balance')
        
        # identity_coverage: combined quality metric
        if 'fident' in data_frame.columns and 'algn_fraction' in data_frame.columns:
            test_data_frame['identity_coverage'] = data_frame['fident'] * data_frame['algn_fraction']
            derived_features_added.append('identity_coverage')
        
        if derived_features_added:
            selected_columns.extend(derived_features_added)
            print(f"Added {len(derived_features_added)} derived features: {', '.join(derived_features_added)}")
    else:
        print("Excluding derived features (--exclude_derived_features flag set)")
    
    # Handle GO terms if requested (need to encode them)
    if include_go_terms:
        go_term_columns = ['target_cellular_components', 'target_molecular_functions', 'target_biological_processes']
        go_features_added = []
        
        for go_col in go_term_columns:
            if go_col in data_frame.columns:
                # Parse GO terms if they're strings
                def parse_go_list(val):
                    if pd.isna(val) or val == '' or val == '[]':
                        return []
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str):
                        try:
                            import ast
                            parsed = ast.literal_eval(val)
                            if isinstance(parsed, list):
                                return parsed
                        except:
                            return []
                    return []
                
                go_terms_list = data_frame[go_col].apply(parse_go_list)
                
                # Get most common GO terms (top 20 per category)
                all_terms = [term for sublist in go_terms_list for term in sublist if term]
                if all_terms:
                    from collections import Counter
                    term_counts = Counter(all_terms)
                    top_terms = [term for term, count in term_counts.most_common(20)]
                    
                    # Create binary indicators for top terms
                    for term in top_terms:
                        col_name = f'{go_col}_{term[:30].replace(" ", "_")}'  # Truncate and sanitize
                        test_data_frame[col_name] = go_terms_list.apply(lambda x: 1 if term in x else 0)
                        go_features_added.append(col_name)
        
        if go_features_added:
            selected_columns.extend(go_features_added)
            print(f"Added {len(go_features_added)} GO term binary features")
        else:
            print("Warning: No GO terms found to encode")
    
    # Filter out any columns that don't exist
    available_columns = [col for col in selected_columns if col in test_data_frame.columns]
    test_data_frame = test_data_frame[available_columns].copy()
    selected_columns = available_columns
    
    print(f"Using {len(selected_columns)} features for clustering: {selected_columns}")
    
    # Scale the data
    if scaler_type == 'zscore' or scaler_type == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler (z-score normalization: mean=0, std=1)")
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        print("Using RobustScaler (median/IQR scaling, robust to outliers)")
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("Using MinMaxScaler (scales to 0-1 range)")
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Choose 'robust', 'zscore'/'standard', or 'minmax'")
    
    scaled_data = scaler.fit_transform(test_data_frame)
    
    # Run dimensionality reduction (unsupervised - no labels)
    # This is purely data-driven for discovering patterns, not validating known examples
    if use_diffusion:
        # Diffusion maps are for trajectory/pseudotime analysis, not clustering
        # Handle this separately
        print("Note: Diffusion maps are used for trajectory analysis, not clustering.")
        print("Computing diffusion map and pseudotime...")
        
        diffusion_results = compute_diffusion_map_analysis(
            scaled_data, 
            data_frame, 
            pca_test_dir, 
            organism_name,
            diffusion_alpha=diffusion_alpha,
            diffusion_n_neighbors=diffusion_n_neighbors,
            plot_3d=plot_3d,
            use_natural_attractor=use_natural_attractor,
            save_cloud_definition=save_cloud_definition,
            transfer_cloud_from=transfer_cloud_from,
            use_adaptive_epsilon=use_adaptive_epsilon,
            adaptive_k_neighbors=adaptive_k_neighbors,
            auto_n_neighbors=auto_n_neighbors,
            use_spectral_cloud=use_spectral_cloud if 'use_spectral_cloud' in locals() else False,
            spectral_n_clusters=spectral_n_clusters if 'spectral_n_clusters' in locals() else None,
            spectral_gamma=spectral_gamma if 'spectral_gamma' in locals() else None,
            scaler=scaler,
            epsilon_scale=epsilon_scale
        )
        return  # Exit early - no clustering for diffusion maps
    elif use_umap:
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install with: pip install umap-learn or conda install -c conda-forge umap-learn")
        print(f"Using UMAP for dimensionality reduction (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})...")
        # UMAP typically works best with 2-3 dimensions for visualization
        # For clustering, we'll use more dimensions
        n_umap_components = min(10, len(selected_columns) - 1)  # Use up to 10 components for clustering
        
        # Ensure reproducibility: set numpy random seed before UMAP
        np.random.seed(42)
        
        # Print diagnostic info for reproducibility
        print(f"  Data shape: {scaled_data.shape}")
        print(f"  Features used: {len(selected_columns)}")
        print(f"  Random seed: 42 (for reproducibility)")
        print(f"  UMAP parameters: n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, n_components={n_umap_components}")
        
        reducer = umap.UMAP(n_components=n_umap_components, 
                           n_neighbors=umap_n_neighbors, 
                           min_dist=umap_min_dist,
                           random_state=42,
                           metric='euclidean',
                           verbose=False)
        reduced_data = reducer.fit_transform(scaled_data)
        method_name = 'umap'
        # For visualization, we'll use the first 3 components
        pca_data = reduced_data[:, :min(3, n_umap_components)]
        n_components = n_umap_components
        print(f"UMAP reduction complete. Using {n_components} components for clustering.")
    else:
        # Run PCA (default)
        pca = PCA()
        pca.fit(scaled_data)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        
        print(f"Number of components to retain 95% of the variance: {n_components}")
        print(f"First 3 components explain {sum(explained_variance_ratio[:3]):.2%} of variance")
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(scaled_data)
        pca_data = reduced_data  # For clustering
        method_name = 'pca'
    
    # Estimate optimal number of clusters if requested
    if auto_clusters:
        optimal_k, cluster_scores = estimate_optimal_clusters(
            reduced_data, 
            clustering_method=clustering_method,
            method=cluster_estimation_method
        )
        n_clusters = optimal_k
        print(f"\nUsing automatically estimated number of clusters: {n_clusters}")
    else:
        cluster_scores = None
    
    # Perform clustering on PCA-reduced data
    if clustering_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, n_init=20, random_state=42, max_iter=600)
    elif clustering_method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    cluster_labels = clusterer.fit_predict(reduced_data)
    
    # Diagnostic: Check if data has natural cluster structure
    print(f"\nData Structure Diagnostics:")
    print(f"  Number of data points: {len(reduced_data)}")
    print(f"  Dimensionality: {reduced_data.shape[1]}")
    
    # Check variance in reduced space
    if use_umap:
        variance_per_component = np.var(reduced_data, axis=0)
        print(f"  Variance per UMAP component: {variance_per_component[:5]}")
    else:
        variance_per_component = np.var(reduced_data, axis=0)
        print(f"  Variance per PC: {variance_per_component[:5]}")
    
    # Check if data is too uniform (low variance suggests poor separation)
    total_variance = np.sum(variance_per_component)
    if total_variance < 1.0:
        print(f"  Warning: Low total variance ({total_variance:.4f}). Data may be too uniform for clustering.")
    print()
    
    # Evaluate UMAP quality if using UMAP
    if use_umap:
        print("\n" + "="*60)
        print("UMAP Quality Metrics")
        print("="*60)
        
        # Trustworthiness: measures how well local structure is preserved (0-1, higher is better)
        # Sample subset if data is too large (trustworthiness is O(n^2))
        max_samples = min(5000, len(scaled_data))
        if len(scaled_data) > max_samples:
            # Use fixed random seed for reproducibility
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(len(scaled_data), max_samples, replace=False)
            sample_original = scaled_data[sample_indices]
            sample_reduced = reduced_data[sample_indices, :min(10, n_components)]
            n_neighbors_trust = min(umap_n_neighbors, max_samples - 1)
        else:
            sample_original = scaled_data
            sample_reduced = reduced_data[:, :min(10, n_components)]
            n_neighbors_trust = umap_n_neighbors
        
        try:
            trust = trustworthiness(sample_original, sample_reduced, n_neighbors=n_neighbors_trust, metric='euclidean')
            print(f"Trustworthiness (local structure preservation): {trust:.4f} (higher is better, max=1.0)")
            if trust < 0.7:
                print("  Warning: Low trustworthiness. Consider increasing n_neighbors or decreasing min_dist.")
            elif trust > 0.95:
                print("  Note: Very high trustworthiness. May be overfitting to local structure.")
        except Exception as e:
            print(f"  Could not compute trustworthiness: {e}")
        
        # Cluster quality metrics
        try:
            silhouette = silhouette_score(reduced_data, cluster_labels)
            print(f"Silhouette Score (cluster separation): {silhouette:.4f} (higher is better, range: -1 to 1)")
            if silhouette < 0.2:
                print("  Warning: Poor cluster separation. Consider adjusting n_neighbors or min_dist.")
        except Exception as e:
            print(f"  Could not compute silhouette score: {e}")
        
        try:
            db_score = davies_bouldin_score(reduced_data, cluster_labels)
            print(f"Davies-Bouldin Index (cluster quality): {db_score:.4f} (lower is better)")
            if db_score > 2.0:
                print("  Warning: High DB index suggests poor cluster separation.")
        except Exception as e:
            print(f"  Could not compute Davies-Bouldin score: {e}")
        
        # Parameter recommendations
        print("\nParameter Tuning Guidelines:")
        print(f"  Current: n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}")
        print("  - If clusters are too tight/overlapping: increase min_dist (try 0.2-0.5)")
        print("  - If clusters are too spread out: decrease min_dist (try 0.01-0.05)")
        print("  - If local structure is lost: decrease n_neighbors (try 5-10)")
        print("  - If global structure is lost: increase n_neighbors (try 30-50)")
        
        # Check if number of clusters might be the issue
        unique_clusters = len(np.unique(cluster_labels))
        print(f"\nCluster Count Analysis:")
        print(f"  Requested clusters: {n_clusters}")
        print(f"  Actual unique clusters: {unique_clusters}")
        if unique_clusters < n_clusters:
            print(f"  Warning: Fewer clusters than requested. Data may not support {n_clusters} clusters.")
            print(f"  Consider trying fewer clusters (e.g., {unique_clusters} or {max(2, unique_clusters-2)})")
        
        # Check cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            pct = size / len(cluster_labels) * 100
            print(f"  Cluster {cluster_id}: {size} points ({pct:.1f}%)")
        
        # Check for very small clusters
        small_clusters = cluster_sizes[cluster_sizes < len(cluster_labels) * 0.01]  # < 1% of data
        if len(small_clusters) > 0:
            print(f"\n  Warning: {len(small_clusters)} very small cluster(s) detected.")
            print(f"  This may indicate over-clustering. Consider reducing n_clusters.")
        
        print("="*60 + "\n")
    
    # Create DataFrame with results
    if use_diffusion:
        # Create full reduced data for DataFrame (all components for clustering)
        pca_columns = [f'Diffusion{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(reduced_data, columns=pca_columns)
        # For plotting, use first 3 components
        pca_df_plot = pca_df.iloc[:, :min(3, n_components)].copy()
    elif use_umap:
        # Create full reduced data for DataFrame (all components for clustering)
        pca_columns = [f'UMAP{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(reduced_data, columns=pca_columns)
        # For plotting, use first 3 components
        pca_df_plot = pca_df.iloc[:, :min(3, n_components)].copy()
    else:
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(reduced_data, columns=pca_columns)
        # For plotting, use first 3 components
        pca_df_plot = pca_df.iloc[:, :min(3, n_components)].copy()
    
    pca_df['query'] = data_frame['query'].values
    pca_df['cluster'] = cluster_labels
    
    # Add candidate/mimic labels if they exist
    # Only label the BEST alignment from each known mimic/candidate (not all alignments)
    pca_df['label'] = 'cluster_' + pca_df['cluster'].astype(str)
    
    # Calculate alignment quality score (rows are in same order as data_frame)
    alignment_quality = (data_frame['score'].values * 
                        data_frame['fident'].values * 
                        data_frame['algn_fraction'].values)
    
    # For mimics: only label the best alignment per query
    mimic_mask = data_frame['query'].isin(validation_list).values
    if mimic_mask.sum() > 0:
        # Create temp dataframe to find best per query
        mimic_df = pd.DataFrame({
            'query': data_frame['query'].values,
            'quality': alignment_quality,
            'row_idx': range(len(data_frame))
        })
        mimic_df = mimic_df[mimic_mask]
        best_mimic_indices = mimic_df.groupby('query')['quality'].idxmax().values
        # Map back to original DataFrame indices using row_idx, then use those for pca_df
        # pca_df has same row order as data_frame (both filtered the same way)
        original_mimic_indices = mimic_df.loc[best_mimic_indices, 'row_idx'].values
        pca_df.loc[original_mimic_indices, 'label'] = 'mimic'
        print(f"Labeled {len(original_mimic_indices)} best mimic alignments (one per known mimic)")
    
    # For candidates: only label the best alignment per query
    candidate_mask = data_frame['query'].isin(candidate_list).values
    if candidate_mask.sum() > 0:
        candidate_df = pd.DataFrame({
            'query': data_frame['query'].values,
            'quality': alignment_quality,
            'row_idx': range(len(data_frame))
        })
        candidate_df = candidate_df[candidate_mask]
        best_candidate_indices = candidate_df.groupby('query')['quality'].idxmax().values
        # Map back to original DataFrame indices using row_idx, then use those for pca_df
        original_candidate_indices = candidate_df.loc[best_candidate_indices, 'row_idx'].values
        pca_df.loc[original_candidate_indices, 'label'] = 'candidate'
        print(f"Labeled {len(original_candidate_indices)} best candidate alignments (one per known candidate)")
    
    # Save results
    suffix = f'{method_name}_{clustering_method}_{n_clusters}'
    pca_df.to_csv(pca_test_dir + f'/{organism_name}_value_driven_{suffix}.csv', index=False)
    
    # Output mimic information with clusters and GO terms as CSV
    mimic_points = pca_df[pca_df['label'] == 'mimic'].copy()
    if len(mimic_points) > 0:
        # Get corresponding rows from original data_frame (same order)
        mimic_indices = mimic_points.index
        mimic_data = []
        
        for idx in mimic_indices:
            query_id = pca_df.loc[idx, 'query']
            cluster_id = pca_df.loc[idx, 'cluster']
            
            # Get original alignment data
            orig_row = data_frame.iloc[idx]
            
            # Format GO terms (handle if they're strings or lists)
            def format_go_terms(go_val):
                if pd.isna(go_val) or go_val == '' or go_val == '[]':
                    return ''
                if isinstance(go_val, str):
                    return go_val
                if isinstance(go_val, list):
                    return '; '.join(str(x) for x in go_val)
                return str(go_val)
            
            mimic_data.append({
                'query': query_id,
                'target': orig_row.get('target', ''),
                'cluster': cluster_id,
                'score': orig_row.get('score', ''),
                'fident': orig_row.get('fident', ''),
                'algn_fraction': orig_row.get('algn_fraction', ''),
                'target_cellular_components': format_go_terms(orig_row.get('target_cellular_components', '')),
                'target_molecular_functions': format_go_terms(orig_row.get('target_molecular_functions', '')),
                'target_biological_processes': format_go_terms(orig_row.get('target_biological_processes', ''))
            })
        
        # Write to CSV
        mimic_df = pd.DataFrame(mimic_data)
        csv_file = pca_test_dir + f'/{organism_name}_mimics_clusters_GO_{suffix}.csv'
        mimic_df.to_csv(csv_file, index=False)
        print(f"Mimic cluster and GO term information (CSV) saved to: {csv_file}")
        
        # Also write text format
        txt_file = pca_test_dir + f'/{organism_name}_mimics_clusters_GO_{suffix}.txt'
        with open(txt_file, 'w') as f:
            f.write(f"Mimic Alignments: Cluster Assignments and GO Terms\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total mimic points: {len(mimic_data)}\n")
            f.write(f"Method: {method_name.upper()}, Clustering: {clustering_method}\n\n")
            
            for i, info in enumerate(mimic_data, 1):
                f.write(f"{'='*80}\n")
                f.write(f"Mimic {i}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Query ID: {info['query']}\n")
                f.write(f"Target ID: {info['target']}\n")
                f.write(f"Cluster: {info['cluster']}\n")
                f.write(f"Alignment Score: {info['score']}\n")
                f.write(f"Fractional Identity: {info['fident']}\n")
                f.write(f"Alignment Fraction: {info['algn_fraction']}\n")
                f.write(f"\nGO Terms:\n")
                f.write(f"  Cellular Components: {info['target_cellular_components'] or 'None'}\n")
                f.write(f"  Molecular Functions: {info['target_molecular_functions'] or 'None'}\n")
                f.write(f"  Biological Processes: {info['target_biological_processes'] or 'None'}\n")
                f.write(f"\n")
            
            # Summary by cluster
            f.write(f"\n{'='*80}\n")
            f.write(f"Summary by Cluster\n")
            f.write(f"{'='*80}\n")
            # Count mimics per cluster
            mimic_cluster_counts = {}
            for info in mimic_data:
                cluster = info['cluster']
                mimic_cluster_counts[cluster] = mimic_cluster_counts.get(cluster, 0) + 1
            
            # Count total samples per cluster (from full pca_df)
            total_cluster_counts = pca_df['cluster'].value_counts().to_dict()
            
            for cluster in sorted(mimic_cluster_counts.keys()):
                total_samples = total_cluster_counts.get(cluster, 0)
                mimic_count = mimic_cluster_counts[cluster]
                f.write(f"Cluster {cluster}: {total_samples} total sample(s), {mimic_count} mimic(s)\n")
                cluster_queries = [info['query'] for info in mimic_data if info['cluster'] == cluster]
                f.write(f"  Mimic Query IDs: {', '.join(cluster_queries)}\n")
            
            # Also list clusters with no mimics
            clusters_with_mimics = set(mimic_cluster_counts.keys())
            clusters_without_mimics = set(total_cluster_counts.keys()) - clusters_with_mimics
            if clusters_without_mimics:
                f.write(f"\nClusters without known mimics:\n")
                for cluster in sorted(clusters_without_mimics):
                    total_samples = total_cluster_counts.get(cluster, 0)
                    f.write(f"  Cluster {cluster}: {total_samples} sample(s)\n")
        
        print(f"Mimic cluster and GO term information (text) saved to: {txt_file}")
    
    # Create 2D visualization
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    # Generate colors for clusters
    unique_clusters = sorted(pca_df['cluster'].unique())
    cluster_colors = plt.cm.get_cmap('tab20', len(unique_clusters))
    
    # Get column names for plotting (use first 2 components)
    x_col = pca_df_plot.columns[0]
    y_col = pca_df_plot.columns[1] if len(pca_df_plot.columns) > 1 else pca_df_plot.columns[0]
    
    for cluster_id in unique_clusters:
        cluster_mask = pca_df['cluster'] == cluster_id
        cluster_data = pca_df_plot[cluster_mask]
        if cluster_id == -1:  # Noise points (if any)
            plt.scatter(cluster_data[x_col], cluster_data[y_col], 
                       c='gray', s=4, alpha=0.5, label=f'Noise')
        else:
            plt.scatter(cluster_data[x_col], cluster_data[y_col],
                       c=[cluster_colors(cluster_id)], s=10, 
                       label=f'Cluster {cluster_id}', alpha=0.6)
    
    # Highlight candidates and mimics
    candidates = pca_df_plot[pca_df['label'] == 'candidate']
    mimics = pca_df_plot[pca_df['label'] == 'mimic']
    
    if len(candidates) > 0:
        plt.scatter(candidates[x_col], candidates[y_col], 
                   c='cyan', edgecolor='k', s=35, zorder=10, label='candidate')
    if len(mimics) > 0:
        plt.scatter(mimics[x_col], mimics[y_col],
                   c='mediumspringgreen', edgecolor='k', s=20, zorder=10, label='mimic')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if use_diffusion:
        method_label = 'Diffusion Map'
    elif use_umap:
        method_label = 'UMAP'
    else:
        method_label = 'PCA'
    title = f'Value-Driven {method_label} Clustering ({clustering_method}) - {organism_name.capitalize()}'
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(pca_test_dir + f'/value_driven_{suffix}_2d.png', dpi=300)
    plt.close()
    
    # Create 3D visualization if requested
    if plot_3d and len(pca_df_plot.columns) >= 3:
        plt.clf()
        z_col = pca_df_plot.columns[2]
        
        # Define multiple viewing angles: (elevation, azimuth, title_suffix)
        view_angles = [
            (30, 45, 'standard'),
            (0, 0, 'xy_plane'),      # Top-down view (looking down Z-axis)
            (90, 0, 'xz_plane'),     # Side view (looking along Y-axis)
            (0, 90, 'yz_plane'),     # Side view (looking along X-axis)
            (30, 135, 'diagonal1'),
            (30, 225, 'diagonal2'),
            (60, 45, 'elevated'),
        ]
        
        for elev, azim, suffix in view_angles:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Generate colors for clusters
            for cluster_id in unique_clusters:
                cluster_mask = pca_df['cluster'] == cluster_id
                cluster_data = pca_df_plot[cluster_mask]
                if cluster_id == -1:  # Noise points (if any)
                    ax.scatter(cluster_data[x_col], cluster_data[y_col], cluster_data[z_col],
                             c='gray', s=4, alpha=0.5, label=f'Noise')
                else:
                    ax.scatter(cluster_data[x_col], cluster_data[y_col], cluster_data[z_col],
                             c=[cluster_colors(cluster_id)], s=10, 
                             label=f'Cluster {cluster_id}', alpha=0.6)
            
            # Highlight candidates and mimics
            if len(candidates) > 0:
                ax.scatter(candidates[x_col], candidates[y_col], candidates[z_col],
                         c='cyan', edgecolor='k', s=35, label='candidate', marker='^')
            if len(mimics) > 0:
                ax.scatter(mimics[x_col], mimics[y_col], mimics[z_col],
                         c='mediumspringgreen', edgecolor='k', s=20, label='mimic', marker='s')
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            if use_diffusion:
                method_label = 'Diffusion Map'
            elif use_umap:
                method_label = 'UMAP'
            else:
                method_label = 'PCA'
            title_3d = f'Value-Driven {method_label} Clustering 3D ({clustering_method}) - {organism_name.capitalize()}\n(View: elev={elev}°, azim={azim}°)'
            ax.set_title(title_3d)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plot_file_3d = os.path.join(pca_test_dir, f'value_driven_{method_label.lower().replace(" ", "_")}_{clustering_method}_3d_{suffix}.png')
            plt.savefig(plot_file_3d, dpi=300)
            plt.close()
            print(f"Saved 3D visualization ({suffix}) to {plot_file_3d}")
    elif plot_3d and n_components < 3:
        print("Warning: Cannot create 3D plot - only {n_components} components available. Need at least 3.")
    
    print(f"Value-driven clustering complete. Found {len(unique_clusters)} clusters.")

def pca_analysis(pca_test_dir, data_frame):
     # Store the categorical value in a separate variable
    data_frame['simplified_labels'] = data_frame['target_cellular_components'].apply(classify_label)
    labels = data_frame['simplified_labels']
    # Drop the categorical column from the data frame
    data_frame = data_frame.drop(columns=['target_cellular_components', 'simplified_labels'])
    
    # Select specific columns for PCA
    selected_columns = ['score', 'tcov', 'qcov', 'fident', 'algn_fraction', 'qlen', 'tlen' ]  # Replace with your column names
    data_frame = data_frame[selected_columns]
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data_frame)
    
    pca = PCA()
    pca.fit(scaled_data)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    '''
    # Plot the cumulative explained variance ratio  
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)  

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Ratio')
    plt.savefig(pca_test_dir  + '/cumulative_explained_variance_ratio.png')
    '''
    
    # Choose the number of components based on the cumulative explained variance plot
      # For example, to reduce to 2 dimensions
    # Get PCA loadings
    loadings = pca.components_

    # Create a DataFrame for the loadings
    loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=selected_columns)

    print(loadings_df)
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"Number of components to retain 95% of the variance: {n_components}")
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    
    plt.clf()
    
     # Add labels to each point
    for i, label in enumerate(labels):
        if 'mitochon' in ''.join(label):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], c='red', edgecolor='k', s=50)
        elif 'ER' in ''.join(label):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], c='green', edgecolor='k', s=50)
        elif 'GA' in ''.join(label):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], c='yellow', edgecolor='k', s=50)
        else:
            plt.scatter(pca_data[i, 0], pca_data[i, 1], c='blue', edgecolor='k', s=50)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='mitochondrial'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='ER'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Golgi'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='other')]
    plt.legend(handles=handles, title='Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.savefig(pca_test_dir + '/pca_visualization_rbs2.png')
    plt.close()
    
def analyze_system_manifold_overlap(diffusion_results_file, output_dir, n_neighbors=15, use_dc123=True):
    """
    Analyze and visualize manifold overlap between different systems.
    
    Parameters:
    -----------
    diffusion_results_file : str
        Path to CSV file with diffusion map results (must have 'system' column)
    output_dir : str
        Directory to save output visualizations and statistics
    n_neighbors : int
        Number of neighbors for density estimation
    use_dc123 : bool
        If True, use DC1, DC2, DC3 for analysis; else use DC1, DC2
    """
    from scipy.spatial.distance import cdist
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import NearestNeighbors
    import seaborn as sns
    
    def simplify_system_name(name):
        """Simplify system names for display in plots."""
        name_lower = name.lower()
        if 'amoeba' in name_lower or 'ameoba' in name_lower:
            return 'amoeba'
        elif 'human' in name_lower:
            return 'human'
        elif 'wmel' in name_lower or 'wolbachia' in name_lower:
            return 'wmel'
        elif 'dros' in name_lower or 'drosophila' in name_lower:
            return 'dros'
        elif 'legionella' in name_lower or 'leg' in name_lower:
            return 'legionella'
        elif 'helicobacter' in name_lower or 'hp' in name_lower:
            return 'helicobacter'
        else:
            # Return first word or keep original if short
            return name.split('_')[0] if '_' in name else name
    
    print("\n" + "="*60)
    print("System Manifold Overlap Analysis")
    print("="*60)
    
    # Load diffusion map results
    results_df = pd.read_csv(diffusion_results_file)
    
    if 'system' not in results_df.columns:
        print("Error: 'system' column not found in results file")
        return
    
    systems = sorted(results_df['system'].unique())
    n_systems = len(systems)
    
    # Create simplified names for display
    system_display_names = {sys: simplify_system_name(sys) for sys in systems}
    
    print(f"Found {n_systems} systems: {', '.join(systems)}")
    print(f"Display names: {', '.join([f'{k} -> {v}' for k, v in system_display_names.items()])}")
    
    # Get diffusion coordinates
    if use_dc123:
        dc_cols = ['Diffusion1', 'Diffusion2', 'Diffusion3']
        n_dims = 3
    else:
        dc_cols = ['Diffusion1', 'Diffusion2']
        n_dims = 2
    
    missing_cols = [col for col in dc_cols if col not in results_df.columns]
    if missing_cols:
        print(f"Error: Missing diffusion coordinate columns: {missing_cols}")
        return
    
    coords = results_df[dc_cols].values
    
    os.makedirs(output_dir, exist_ok=True)
    
    # VALIDATION: Check if systems form distinct sub-structures in the merged manifold
    print("\n" + "="*60)
    print("VALIDATION: Checking if systems form distinct sub-structures")
    print("="*60)
    
    from sklearn.metrics import silhouette_score
    
    # 1. System size balance
    system_sizes = results_df['system'].value_counts()
    print(f"\nSystem sizes:")
    for sys, size in system_sizes.items():
        pct = size / len(results_df) * 100
        print(f"  {sys}: {size} alignments ({pct:.1f}%)")
    
    size_ratio = system_sizes.max() / system_sizes.min()
    if size_ratio > 5:
        print(f"\n  WARNING: Large size imbalance (ratio: {size_ratio:.1f}x)")
        print(f"    The larger system may dominate the global diffusion map structure.")
        print(f"    Consider: (1) subsampling larger system, (2) separate analysis, or")
        print(f"    (3) interpreting results with this imbalance in mind.")
    else:
        print(f"\n  System sizes are reasonably balanced (ratio: {size_ratio:.1f}x)")
    
    # 2. Spatial separation (silhouette score)
    system_labels = results_df['system'].astype('category').cat.codes.values
    if len(np.unique(system_labels)) > 1:
        silhouette = silhouette_score(coords, system_labels)
        print(f"\n  Silhouette score (system separation): {silhouette:.3f}")
        print(f"    Range: -1 (poor separation) to +1 (perfect separation)")
        if silhouette > 0.3:
            print(f"    ✓ Systems form reasonably distinct clusters")
        elif silhouette > 0.1:
            print(f"    ⚠ Systems show some separation but significant overlap")
        else:
            print(f"    ⚠ Systems are highly overlapping - may not form distinct sub-structures")
    
    # 3. Centroid separation
    centroids = {}
    for sys in systems:
        mask = results_df['system'] == sys
        centroids[sys] = coords[mask].mean(axis=0)
    
    print(f"\n  Centroid distances (between system centers):")
    min_centroid_dist = float('inf')
    for i, sys1 in enumerate(systems):
        for j, sys2 in enumerate(systems):
            if i < j:
                dist = np.linalg.norm(centroids[sys1] - centroids[sys2])
                print(f"    {sys1} <-> {sys2}: {dist:.4f}")
                min_centroid_dist = min(min_centroid_dist, dist)
    
    # 4. Within-system spread vs between-system distance
    print(f"\n  System spread (std within each system):")
    for sys in systems:
        mask = results_df['system'] == sys
        spread = coords[mask].std(axis=0).mean()  # Average std across dimensions
        print(f"    {sys}: {spread:.4f}")
    
    avg_spread = np.mean([coords[results_df['system'] == sys].std(axis=0).mean() 
                          for sys in systems])
    separation_ratio = min_centroid_dist / avg_spread if avg_spread > 0 else 0
    print(f"\n  Separation ratio (centroid distance / avg spread): {separation_ratio:.2f}")
    if separation_ratio > 2.0:
        print(f"    ✓ Systems are well-separated relative to their spread")
    elif separation_ratio > 1.0:
        print(f"    ⚠ Systems show moderate separation")
    else:
        print(f"    ⚠ Systems overlap significantly - may not be distinct sub-structures")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("  The merged diffusion map approach is VALID if:")
    print("    1. Systems form distinct regions (silhouette > 0.1, separation ratio > 1.0)")
    print("    2. You want to understand how systems relate in a unified feature space")
    print("    3. The global structure captures meaningful biological relationships")
    print("\n  This approach is INAPPROPRIATE if:")
    print("    1. Systems are completely overlapping (silhouette < 0, separation ratio < 0.5)")
    print("    2. One system dominates (size ratio > 10x) and you want equal representation")
    print("    3. You need system-specific manifolds that are not influenced by others")
    print("="*60 + "\n")
    
    # 1. Compute pairwise system overlap statistics
    print("\nComputing pairwise system overlap statistics...")
    overlap_stats = []
    
    for i, sys1 in enumerate(systems):
        for j, sys2 in enumerate(systems):
            if i >= j:
                continue
            
            mask1 = results_df['system'] == sys1
            mask2 = results_df['system'] == sys2
            
            coords1 = coords[mask1]
            coords2 = coords[mask2]
            
            # Compute convex hull or density-based overlap
            # Method 1: Nearest neighbor overlap (samples from sys1 near sys2 and vice versa)
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(coords2)
            distances1_to_2, _ = nn.kneighbors(coords1)
            # Samples from sys1 that are close to sys2
            close1_to_2 = (distances1_to_2[:, 0] < np.percentile(distances1_to_2, 25)).sum()
            
            nn.fit(coords1)
            distances2_to_1, _ = nn.kneighbors(coords2)
            close2_to_1 = (distances2_to_1[:, 0] < np.percentile(distances2_to_1, 25)).sum()
            
            # Overlap percentage
            overlap_pct_1 = close1_to_2 / len(coords1) * 100
            overlap_pct_2 = close2_to_1 / len(coords2) * 100
            avg_overlap = (overlap_pct_1 + overlap_pct_2) / 2
            
            # Jaccard-like similarity: intersection / union in k-nearest neighbor space
            # For each sample, find its k nearest neighbors across all systems
            nn_all = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn_all.fit(coords)
            
            # Count how many neighbors from each system
            neighbors1_in_1 = 0
            neighbors1_in_2 = 0
            neighbors2_in_1 = 0
            neighbors2_in_2 = 0
            
            for idx in np.where(mask1)[0]:
                _, neighbor_indices = nn_all.kneighbors([coords[idx]])
                neighbor_systems = results_df.iloc[neighbor_indices[0][1:]]['system']  # Exclude self
                neighbors1_in_1 += (neighbor_systems == sys1).sum()
                neighbors1_in_2 += (neighbor_systems == sys2).sum()
            
            for idx in np.where(mask2)[0]:
                _, neighbor_indices = nn_all.kneighbors([coords[idx]])
                neighbor_systems = results_df.iloc[neighbor_indices[0][1:]]['system']
                neighbors2_in_1 += (neighbor_systems == sys1).sum()
                neighbors2_in_2 += (neighbor_systems == sys2).sum()
            
            # Jaccard similarity: shared neighbors / total unique neighbors
            shared_neighbors = neighbors1_in_2 + neighbors2_in_1
            total_neighbors = neighbors1_in_1 + neighbors1_in_2 + neighbors2_in_1 + neighbors2_in_2
            jaccard_similarity = shared_neighbors / total_neighbors if total_neighbors > 0 else 0
            
            # ===== GLOBAL GEOMETRIC ANALYSIS =====
            # 1. Centroid distance (center of mass separation)
            centroid1 = np.mean(coords1, axis=0)
            centroid2 = np.mean(coords2, axis=0)
            centroid_distance = np.linalg.norm(centroid1 - centroid2)
            
            # 2. Covariance matrix comparison (shape/orientation)
            cov1 = np.cov(coords1.T)
            cov2 = np.cov(coords2.T)
            
            # Eigenvalues (principal axes lengths)
            eigenvals1, eigenvecs1 = np.linalg.eigh(cov1)
            eigenvals2, eigenvecs2 = np.linalg.eigh(cov2)
            # Sort by eigenvalue (largest first)
            idx1 = eigenvals1.argsort()[::-1]
            idx2 = eigenvals2.argsort()[::-1]
            eigenvals1 = eigenvals1[idx1]
            eigenvals2 = eigenvals2[idx2]
            eigenvecs1 = eigenvecs1[:, idx1]
            eigenvecs2 = eigenvecs2[:, idx2]
            
            # Principal direction alignment (dot product of first principal components)
            # NOTE: PC1 may not be informative if DC1 has minimal variation
            pc1_alignment = abs(np.dot(eigenvecs1[:, 0], eigenvecs2[:, 0]))
            
            # Pseudotime gradient alignment (more informative - captures actual flow direction)
            # Compute gradient direction from low to high pseudotime
            pseudotime_gradient_alignment = np.nan
            if 'diffusion_pseudotime' in results_df.columns:
                try:
                    # Get pseudotime values for each system
                    pseudotime1 = results_df.loc[mask1, 'diffusion_pseudotime'].values
                    pseudotime2 = results_df.loc[mask2, 'diffusion_pseudotime'].values
                    
                    # Compute gradient direction: direction from low to high pseudotime
                    # Use least squares to find direction of increasing pseudotime
                    # gradient = (X^T X)^(-1) X^T y, where X is coords and y is pseudotime
                    
                    # System 1: find direction of increasing pseudotime
                    if len(pseudotime1) > 2 and pseudotime1.std() > 1e-6:
                        # Center the coordinates and pseudotime
                        coords1_centered = coords1 - coords1.mean(axis=0)
                        pseudotime1_centered = pseudotime1 - pseudotime1.mean()
                        # Least squares: gradient = (X^T X)^(-1) X^T y
                        try:
                            gradient1 = np.linalg.lstsq(coords1_centered, pseudotime1_centered, rcond=None)[0]
                            gradient1 = gradient1 / (np.linalg.norm(gradient1) + 1e-10)  # Normalize
                        except:
                            gradient1 = None
                    else:
                        gradient1 = None
                    
                    # System 2: find direction of increasing pseudotime
                    if len(pseudotime2) > 2 and pseudotime2.std() > 1e-6:
                        # Center the coordinates and pseudotime
                        coords2_centered = coords2 - coords2.mean(axis=0)
                        pseudotime2_centered = pseudotime2 - pseudotime2.mean()
                        # Least squares: gradient = (X^T X)^(-1) X^T y
                        try:
                            gradient2 = np.linalg.lstsq(coords2_centered, pseudotime2_centered, rcond=None)[0]
                            gradient2 = gradient2 / (np.linalg.norm(gradient2) + 1e-10)  # Normalize
                        except:
                            gradient2 = None
                    else:
                        gradient2 = None
                    
                    # Compute alignment of gradient directions
                    if gradient1 is not None and gradient2 is not None:
                        pseudotime_gradient_alignment = abs(np.dot(gradient1, gradient2))
                except:
                    pseudotime_gradient_alignment = np.nan
            
            # Alternative: Direction of maximum variation (if pseudotime not available)
            # Use the principal component with largest eigenvalue (most variation)
            max_var_alignment = np.nan
            if len(eigenvals1) > 0 and len(eigenvals2) > 0:
                # Find which PC has maximum variance (not necessarily PC1)
                max_var_idx1 = np.argmax(eigenvals1)
                max_var_idx2 = np.argmax(eigenvals2)
                max_var_alignment = abs(np.dot(eigenvecs1[:, max_var_idx1], eigenvecs2[:, max_var_idx2]))
            
            # Volume comparison (determinant of covariance = approximate volume)
            vol1 = np.sqrt(np.linalg.det(cov1)) if np.linalg.det(cov1) > 0 else 0
            vol2 = np.sqrt(np.linalg.det(cov2)) if np.linalg.det(cov2) > 0 else 0
            volume_ratio = vol1 / vol2 if vol2 > 0 else np.inf
            
            # Span comparison (range in each dimension)
            span1 = np.max(coords1, axis=0) - np.min(coords1, axis=0)
            span2 = np.max(coords2, axis=0) - np.min(coords2, axis=0)
            span_ratio = np.mean(span1 / span2) if np.all(span2 > 0) else np.nan
            
            # Mahalanobis distance between centroids (accounts for shape)
            try:
                # Use pooled covariance for Mahalanobis distance
                pooled_cov = (cov1 * (len(coords1) - 1) + cov2 * (len(coords2) - 1)) / (len(coords1) + len(coords2) - 2)
                inv_pooled_cov = np.linalg.pinv(pooled_cov)
                diff = centroid1 - centroid2
                mahalanobis_distance = np.sqrt(diff @ inv_pooled_cov @ diff)
            except:
                mahalanobis_distance = np.nan
            
            # 3. Convex hull overlap (global geometric overlap)
            try:
                from scipy.spatial import ConvexHull
                hull1 = ConvexHull(coords1)
                hull2 = ConvexHull(coords2)
                
                # Count points from sys1 inside hull2 and vice versa
                from scipy.spatial.distance import cdist
                # Check if points from sys1 are inside hull2
                # (simplified: check if distance to hull is small)
                hull2_distances = cdist(coords1, coords2[hull2.vertices])
                points1_in_hull2 = (hull2_distances.min(axis=1) < np.percentile(cdist(coords2, coords2[hull2.vertices]).min(axis=1), 10)).sum()
                
                hull1_distances = cdist(coords2, coords1[hull1.vertices])
                points2_in_hull1 = (hull1_distances.min(axis=1) < np.percentile(cdist(coords1, coords1[hull1.vertices]).min(axis=1), 10)).sum()
                
                convex_hull_overlap_pct = ((points1_in_hull2 / len(coords1)) + (points2_in_hull1 / len(coords2))) / 2 * 100
            except:
                convex_hull_overlap_pct = np.nan
            
            # 4. Spread comparison (standard deviation in each dimension)
            std1 = np.std(coords1, axis=0)
            std2 = np.std(coords2, axis=0)
            std_ratio = np.mean(std1 / std2) if np.all(std2 > 0) else np.nan
            
            overlap_stats.append({
                'system1': sys1,
                'system2': sys2,
                'overlap_pct_sys1': overlap_pct_1,
                'overlap_pct_sys2': overlap_pct_2,
                'avg_overlap_pct': avg_overlap,
                'jaccard_similarity': jaccard_similarity,
                'n_samples_sys1': len(coords1),
                'n_samples_sys2': len(coords2),
                # Global geometric metrics
                'centroid_distance': centroid_distance,
                'mahalanobis_distance': mahalanobis_distance,
                'pc1_alignment': pc1_alignment,  # May be uninformative if DC1 has minimal variation
                'pseudotime_gradient_alignment': pseudotime_gradient_alignment,  # Flow direction alignment
                'max_variation_alignment': max_var_alignment,  # Direction of maximum variation
                'volume_ratio': volume_ratio,
                'span_ratio': span_ratio,
                'std_ratio': std_ratio,
                'convex_hull_overlap_pct': convex_hull_overlap_pct
            })
    
    overlap_df = pd.DataFrame(overlap_stats)
    
    if len(overlap_df) == 0:
        print("Warning: No system pairs found for overlap analysis (need at least 2 systems)")
        return
    
    overlap_df.to_csv(os.path.join(output_dir, 'system_manifold_overlap_stats.csv'), index=False)
    print(f"Saved overlap statistics to: {os.path.join(output_dir, 'system_manifold_overlap_stats.csv')}")
    
    # 2. Create overlap heatmap
    print("Creating overlap heatmap...")
    overlap_matrix = np.zeros((n_systems, n_systems))
    jaccard_matrix = np.zeros((n_systems, n_systems))
    
    for _, row in overlap_df.iterrows():
        i = systems.index(row['system1'])
        j = systems.index(row['system2'])
        overlap_matrix[i, j] = row['avg_overlap_pct']
        overlap_matrix[j, i] = row['avg_overlap_pct']
        jaccard_matrix[i, j] = row['jaccard_similarity']
        jaccard_matrix[j, i] = row['jaccard_similarity']
    
    # Set diagonal to 100% (self-overlap)
    np.fill_diagonal(overlap_matrix, 100.0)
    np.fill_diagonal(jaccard_matrix, 1.0)
    
    # Use simplified names for display
    display_labels = [system_display_names[sys] for sys in systems]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overlap percentage heatmap
    sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=display_labels, yticklabels=display_labels, ax=ax1, cbar_kws={'label': 'Overlap %'})
    ax1.set_title('System Manifold Overlap Percentage')
    ax1.set_xlabel('System')
    ax1.set_ylabel('System')
    
    # Jaccard similarity heatmap
    sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels, ax=ax2, cbar_kws={'label': 'Jaccard Similarity'})
    ax2.set_title('System Manifold Jaccard Similarity')
    ax2.set_xlabel('System')
    ax2.set_ylabel('System')
    
    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, 'system_manifold_overlap_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {heatmap_file}")
    
    # 3. Create pairwise comparison plots
    print("Creating pairwise system comparison plots...")
    n_pairs = len(overlap_df)
    
    if n_dims == 2:
        fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 6))
        if n_pairs == 1:
            axes = [axes]
    else:
        # For 3D, create separate figures
        axes = None
    
    for idx, (_, row) in enumerate(overlap_df.iterrows()):
        sys1 = row['system1']
        sys2 = row['system2']
        
        mask1 = results_df['system'] == sys1
        mask2 = results_df['system'] == sys2
        
        coords1 = coords[mask1]
        coords2 = coords[mask2]
        
        if n_dims == 2:
            ax = axes[idx]
            ax.scatter(coords1[:, 0], coords1[:, 1], alpha=0.5, s=10, label=system_display_names[sys1], c='blue')
            ax.scatter(coords2[:, 0], coords2[:, 1], alpha=0.5, s=10, label=system_display_names[sys2], c='red')
            
            # Add global geometric features
            # Centroids
            centroid1 = np.mean(coords1, axis=0)
            centroid2 = np.mean(coords2, axis=0)
            ax.scatter(centroid1[0], centroid1[1], s=200, marker='*', c='blue', 
                      edgecolors='black', linewidths=1.5, label=f'{system_display_names[sys1]} centroid', zorder=5)
            ax.scatter(centroid2[0], centroid2[1], s=200, marker='*', c='red', 
                      edgecolors='black', linewidths=1.5, label=f'{system_display_names[sys2]} centroid', zorder=5)
            
            # Principal components (first PC as arrows)
            # The arrow shows the main direction of variation (first principal component)
            # It points in the direction where the system's data spreads the most
            cov1 = np.cov(coords1.T)
            cov2 = np.cov(coords2.T)
            eigenvals1, eigenvecs1 = np.linalg.eigh(cov1)
            eigenvals2, eigenvecs2 = np.linalg.eigh(cov2)
            idx1 = eigenvals1.argsort()[::-1]
            idx2 = eigenvals2.argsort()[::-1]
            pc1_1 = eigenvecs1[:, idx1[0]] * np.sqrt(eigenvals1[idx1[0]]) * 2
            pc1_2 = eigenvecs2[:, idx2[0]] * np.sqrt(eigenvals2[idx2[0]]) * 2
            arrow1 = ax.arrow(centroid1[0], centroid1[1], pc1_1[0], pc1_1[1], 
                   head_width=0.05, head_length=0.05, fc='blue', ec='blue', linewidth=2, alpha=0.7,
                   label=f'{system_display_names[sys1]} PC1 (main variation)')
            arrow2 = ax.arrow(centroid2[0], centroid2[1], pc1_2[0], pc1_2[1], 
                   head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2, alpha=0.7,
                   label=f'{system_display_names[sys2]} PC1 (main variation)')
            
            # Covariance ellipses (2 standard deviations)
            from matplotlib.patches import Ellipse
            angle1 = np.degrees(np.arctan2(eigenvecs1[1, idx1[0]], eigenvecs1[0, idx1[0]]))
            angle2 = np.degrees(np.arctan2(eigenvecs2[1, idx2[0]], eigenvecs2[0, idx2[0]]))
            width1 = 2 * np.sqrt(eigenvals1[idx1[0]]) * 2
            height1 = 2 * np.sqrt(eigenvals1[idx1[1]]) * 2
            width2 = 2 * np.sqrt(eigenvals2[idx2[0]]) * 2
            height2 = 2 * np.sqrt(eigenvals2[idx2[1]]) * 2
            ellipse1 = Ellipse(centroid1, width1, height1, angle=angle1, 
                              alpha=0.2, facecolor='blue', edgecolor='blue', linewidth=2)
            ellipse2 = Ellipse(centroid2, width2, height2, angle=angle2, 
                              alpha=0.2, facecolor='red', edgecolor='red', linewidth=2)
            ax.add_patch(ellipse1)
            ax.add_patch(ellipse2)
            
            # Centroid distance line
            ax.plot([centroid1[0], centroid2[0]], [centroid1[1], centroid2[1]], 
                   'k--', alpha=0.5, linewidth=1, label=f'Centroid dist: {row["centroid_distance"]:.3f}')
            
            ax.set_xlabel('DC1')
            ax.set_ylabel('DC2')
            title = f'{system_display_names[sys1]} vs {system_display_names[sys2]}\nOverlap: {row["avg_overlap_pct"]:.1f}% | '
            if 'pseudotime_gradient_alignment' in row and not np.isnan(row['pseudotime_gradient_alignment']):
                title += f'Pseudotime grad align: {row["pseudotime_gradient_alignment"]:.3f} | '
            elif 'max_variation_alignment' in row and not np.isnan(row['max_variation_alignment']):
                title += f'Max var align: {row["max_variation_alignment"]:.3f} | '
            else:
                title += f'PC1 align: {row["pc1_alignment"]:.3f} | '
            title += f'Centroid dist: {row["centroid_distance"]:.3f}'
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        else:
            # 3D plot with global geometric features
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], 
                      alpha=0.5, s=10, label=system_display_names[sys1], c='blue')
            ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2],
                      alpha=0.5, s=10, label=system_display_names[sys2], c='red')
            
            # Add centroids
            centroid1 = np.mean(coords1, axis=0)
            centroid2 = np.mean(coords2, axis=0)
            ax.scatter(centroid1[0], centroid1[1], centroid1[2], s=200, marker='*', 
                      c='blue', edgecolors='black', linewidths=1.5, label=f'{system_display_names[sys1]} centroid', zorder=5)
            ax.scatter(centroid2[0], centroid2[1], centroid2[2], s=200, marker='*', 
                      c='red', edgecolors='black', linewidths=1.5, label=f'{system_display_names[sys2]} centroid', zorder=5)
            
            # Add first principal component as arrow
            # The arrow shows the main direction of variation (first principal component)
            # It points in the direction where the system's data spreads the most
            cov1 = np.cov(coords1.T)
            cov2 = np.cov(coords2.T)
            eigenvals1, eigenvecs1 = np.linalg.eigh(cov1)
            eigenvals2, eigenvecs2 = np.linalg.eigh(cov2)
            idx1 = eigenvals1.argsort()[::-1]
            idx2 = eigenvals2.argsort()[::-1]
            pc1_1 = eigenvecs1[:, idx1[0]] * np.sqrt(eigenvals1[idx1[0]]) * 2
            pc1_2 = eigenvecs2[:, idx2[0]] * np.sqrt(eigenvals2[idx2[0]]) * 2
            ax.quiver(centroid1[0], centroid1[1], centroid1[2], 
                     pc1_1[0], pc1_1[1], pc1_1[2], color='blue', arrow_length_ratio=0.2, linewidth=2,
                     label=f'{system_display_names[sys1]} PC1 (main variation)')
            ax.quiver(centroid2[0], centroid2[1], centroid2[2], 
                     pc1_2[0], pc1_2[1], pc1_2[2], color='red', arrow_length_ratio=0.2, linewidth=2,
                     label=f'{system_display_names[sys2]} PC1 (main variation)')
            
            # Centroid distance line
            ax.plot([centroid1[0], centroid2[0]], [centroid1[1], centroid2[1]], 
                   [centroid1[2], centroid2[2]], 'k--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('DC1')
            ax.set_ylabel('DC2')
            ax.set_zlabel('DC3')
            title = f'{system_display_names[sys1]} vs {system_display_names[sys2]}\nOverlap: {row["avg_overlap_pct"]:.1f}% | '
            if 'pseudotime_gradient_alignment' in row and not np.isnan(row['pseudotime_gradient_alignment']):
                title += f'Pseudotime grad align: {row["pseudotime_gradient_alignment"]:.3f} | '
            elif 'max_variation_alignment' in row and not np.isnan(row['max_variation_alignment']):
                title += f'Max var align: {row["max_variation_alignment"]:.3f} | '
            else:
                title += f'PC1 align: {row["pc1_alignment"]:.3f} | '
            title += f'Centroid dist: {row["centroid_distance"]:.3f}'
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7, loc='best')
            
            pair_file = os.path.join(output_dir, f'system_pairwise_{sys1}_vs_{sys2}.png')
            plt.savefig(pair_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    if n_dims == 2:
        plt.tight_layout()
        pairwise_file = os.path.join(output_dir, 'system_pairwise_comparisons.png')
        plt.savefig(pairwise_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pairwise comparisons to: {pairwise_file}")
    
    # 4. Create density plots for each system
    print("Creating system density plots...")
    if n_dims == 2:
        fig, axes = plt.subplots(1, n_systems, figsize=(6*n_systems, 6))
        if n_systems == 1:
            axes = [axes]
        
        for idx, system in enumerate(systems):
            mask = results_df['system'] == system
            system_coords = coords[mask]
            
            # Create 2D density plot
            ax = axes[idx]
            ax.hexbin(system_coords[:, 0], system_coords[:, 1], gridsize=30, cmap='viridis', mincnt=1)
            ax.set_xlabel('DC1')
            ax.set_ylabel('DC2')
            ax.set_title(f'{system_display_names[system]}\n(n={len(system_coords)})')
        
        plt.tight_layout()
        density_file = os.path.join(output_dir, 'system_manifold_densities.png')
        plt.savefig(density_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved density plots to: {density_file}")
    
    # 5. Create global geometric metrics visualization
    print("Creating global geometric metrics visualization...")
    if len(overlap_df) > 0:
        # Create matrices for global metrics
        # Note: pc1_alignment may be uninformative if DC1 has minimal variation
        metric_names = ['centroid_distance', 'mahalanobis_distance', 
                       'pseudotime_gradient_alignment', 'max_variation_alignment', 'pc1_alignment',
                       'volume_ratio', 'span_ratio', 'std_ratio', 'convex_hull_overlap_pct']
        
        # Build matrices for each metric
        # Filter to only metrics that exist in the dataframe
        available_metrics = [m for m in metric_names if m in overlap_df.columns]
        n_metrics = len(available_metrics)
        
        # Create appropriate grid size
        if n_metrics <= 9:
            n_rows, n_cols = 3, 3
            figsize = (18, 15)
        elif n_metrics <= 12:
            n_rows, n_cols = 3, 4
            figsize = (24, 15)
        else:
            n_rows, n_cols = 4, 4
            figsize = (24, 20)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Turn off unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        for idx, metric in enumerate(available_metrics):
                
            matrix = np.zeros((n_systems, n_systems))
            matrix[:] = np.nan
            
            for _, row in overlap_df.iterrows():
                i = systems.index(row['system1'])
                j = systems.index(row['system2'])
                val = row[metric]
                if not np.isnan(val) and not np.isinf(val):
                    matrix[i, j] = val
                    matrix[j, i] = val
            
            # Set diagonal (self-comparison)
            if metric in ['pc1_alignment', 'pseudotime_gradient_alignment', 'max_variation_alignment', 'convex_hull_overlap_pct']:
                if metric == 'convex_hull_overlap_pct':
                    np.fill_diagonal(matrix, 100.0)
                else:
                    np.fill_diagonal(matrix, 1.0)  # Perfect alignment with self
            elif metric in ['volume_ratio', 'span_ratio', 'std_ratio']:
                np.fill_diagonal(matrix, 1.0)
            else:
                np.fill_diagonal(matrix, 0.0)
            
            # Create heatmap
            ax = axes[idx]
            mask = np.isnan(matrix)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis', 
                       xticklabels=display_labels, yticklabels=display_labels, ax=ax,
                       mask=mask, cbar_kws={'label': metric.replace('_', ' ').title()})
            ax.set_title(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_xlabel('System')
            ax.set_ylabel('System')
        
        plt.tight_layout()
        geom_file = os.path.join(output_dir, 'system_global_geometric_metrics.png')
        plt.savefig(geom_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved global geometric metrics to: {geom_file}")
    
    # 6. Summary statistics
    print("\n" + "="*60)
    print("Manifold Overlap Summary")
    print("="*60)
    
    if len(overlap_df) > 0 and 'jaccard_similarity' in overlap_df.columns:
        print(f"\nMost similar systems (by Jaccard similarity):")
        n_show = min(3, len(overlap_df))
        top_similar = overlap_df.nlargest(n_show, 'jaccard_similarity')
        for _, row in top_similar.iterrows():
            print(f"  {row['system1']} <-> {row['system2']}: {row['jaccard_similarity']:.3f}")
    else:
        print("\nWarning: Could not compute Jaccard similarity statistics")
    
    if len(overlap_df) > 0 and 'avg_overlap_pct' in overlap_df.columns:
        print(f"\nMost overlapping systems (by average overlap %):")
        n_show = min(3, len(overlap_df))
        top_overlap = overlap_df.nlargest(n_show, 'avg_overlap_pct')
        for _, row in top_overlap.iterrows():
            print(f"  {row['system1']} <-> {row['system2']}: {row['avg_overlap_pct']:.1f}%")
    else:
        print("\nWarning: Could not compute overlap percentage statistics")
    
    # Global geometric summary
    if len(overlap_df) > 0:
        print(f"\nGlobal Geometric Analysis:")
        print(f"  Closest systems (by centroid distance):")
        if 'centroid_distance' in overlap_df.columns:
            closest = overlap_df.nsmallest(min(3, len(overlap_df)), 'centroid_distance')
            for _, row in closest.iterrows():
                print(f"    {row['system1']} <-> {row['system2']}: {row['centroid_distance']:.4f}")
        
        print(f"  Most aligned systems (by pseudotime gradient - flow direction):")
        if 'pseudotime_gradient_alignment' in overlap_df.columns:
            most_aligned = overlap_df.nlargest(min(3, len(overlap_df)), 'pseudotime_gradient_alignment')
            for _, row in most_aligned.iterrows():
                if not np.isnan(row['pseudotime_gradient_alignment']):
                    print(f"    {row['system1']} <-> {row['system2']}: {row['pseudotime_gradient_alignment']:.3f}")
        
        print(f"  Most aligned systems (by maximum variation direction):")
        if 'max_variation_alignment' in overlap_df.columns:
            most_aligned = overlap_df.nlargest(min(3, len(overlap_df)), 'max_variation_alignment')
            for _, row in most_aligned.iterrows():
                if not np.isnan(row['max_variation_alignment']):
                    print(f"    {row['system1']} <-> {row['system2']}: {row['max_variation_alignment']:.3f}")
        
        print(f"  Most aligned systems (by PC1 - may be uninformative if DC1 has minimal variation):")
        if 'pc1_alignment' in overlap_df.columns:
            most_aligned = overlap_df.nlargest(min(3, len(overlap_df)), 'pc1_alignment')
            for _, row in most_aligned.iterrows():
                print(f"    {row['system1']} <-> {row['system2']}: {row['pc1_alignment']:.3f}")
        
        print(f"  Most similar volume (by volume ratio, closest to 1.0):")
        if 'volume_ratio' in overlap_df.columns:
            vol_ratios = overlap_df.copy()
            vol_ratios['vol_diff_from_1'] = abs(vol_ratios['volume_ratio'] - 1.0)
            most_similar_vol = vol_ratios.nsmallest(min(3, len(vol_ratios)), 'vol_diff_from_1')
            for _, row in most_similar_vol.iterrows():
                print(f"    {row['system1']} <-> {row['system2']}: {row['volume_ratio']:.3f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*60 + "\n")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='path to input data file')
    args = parser.parse_args()
    
    data_frame = pd.read_csv(args.input)
    pca_analysis(data_frame)

if __name__ == '__main__': 
    main()

