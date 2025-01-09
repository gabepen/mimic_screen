import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from gensim.models import Word2Vec
import argparse
import re
import pdb

candidate_list = ['Q73HU7', 'Q73IF8', 'Q73GY6', 'Q73GG5', 'Q73HX8', 'P61189']
validation_list = [
    'Q5ZVD8', 'Q5ZTI6', 'Q5ZUA2', 'Q5ZSI9', 'Q5ZUX1', 'Q5ZWA1', 
    'Q5ZU58', 'Q5ZU32', 'Q5ZUS4', 'Q5ZRQ0', 'Q5ZSZ6', 'Q5ZVF7', 
    'Q5ZYU9', 'Q5ZTM4', 'Q5ZSB6'
]

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
    
def lda_pca_analysis(pca_test_dir, data_frame, organism_name, go_term_type, n_clusters=20):
    
    #go_term_type = 'target_cellular_components'
    #n_clusters = 20
    test_data_frame, labels, selected_columns = semantic_clustering(data_frame, go_term_type, n_clusters)
   #test_data_frame, labels, selected_columns = apply_labels(data_frame)
    
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='path to input data file')
    args = parser.parse_args()
    
    data_frame = pd.read_csv(args.input)
    pca_analysis(data_frame)

if __name__ == '__main__': 
    main()

