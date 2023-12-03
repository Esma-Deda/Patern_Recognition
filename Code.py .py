import os
import zipfile
import string
import torch
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import BertTokenizer, BertModel
from collections import Counter
from nltk.corpus import stopwords
import csv
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import warnings


# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 0: Unzip the dataset
def unzip_dataset(zip_file_path, extracted_folder_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        print("Dataset successfully unzipped.")
    except Exception as e:
        print(f"Error unzipping the dataset: {str(e)}")

# Step 1: Preprocess Text
def preprocess_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token not in string.punctuation]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

# Step 2: Embed Text
def embed_text(text, tokenizer, model):
    cleaned_text = preprocess_text(text, tokenizer)
    input_ids = tokenizer.encode(cleaned_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(input_ids)['last_hidden_state']
    return embeddings.mean(dim=1).numpy()

# Step 3: Perform Dimensionality Reduction
def dimensionality_reduction(embeddings, n_components=5, max_samples=None):
    if max_samples is not None:
        embeddings = embeddings[:max_samples]
    flattened_embeddings = [emb.flatten() for emb in embeddings]
    return flattened_embeddings

# Step 4: Function to find the optimal number of clusters using silhouette score
def find_optimal_clusters_silhouette(embeddings, max_clusters=10):
    flattened_embeddings = np.vstack(embeddings)
    silhouette_scores = []

    num_samples = len(flattened_embeddings)
    max_valid_clusters = min(num_samples - 1, max_clusters)

    for num_clusters in range(2, max_valid_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(flattened_embeddings)
        silhouette_avg = silhouette_score(flattened_embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    best_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"The optimal number of clusters is: {best_num_clusters}")
    return best_num_clusters


# Step 5: Perform Clustering
def perform_clustering(reduced_embeddings, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embeddings) + 1
    return cluster_labels



# Step 6: Save Cluster Information to CSV
def save_cluster_info_to_csv(texts, cluster_labels, output_dir, clusters):
    cluster_info = []

    for cluster_id in set(cluster_labels):
        cluster_texts = [text for i, text in enumerate(texts) if i < len(cluster_labels) and cluster_labels[i] == cluster_id]
        num_documents = len(cluster_texts)
        cluster_text = " ".join(cluster_texts)
        words = cluster_text.split()
        word_counts = Counter(words)
        top_words = [word for word, count in word_counts.most_common(10)]

        G = clusters.get(cluster_id, nx.Graph())

        if G is not None and len(G.nodes) > 0:
            try:
                cluster_coefficient = nx.average_clustering(G)
                diameter = nx.diameter(G)
                density = nx.density(G)
            except nx.NetworkXError as e:
                print(f"Error calculating network characteristics for Cluster {cluster_id}: {e}")
                cluster_coefficient = diameter = density = None
        else:
            cluster_coefficient = diameter = density = None

        top_words_network = get_top_words_per_cluster(cluster_texts, top_words_limit=50)

        cluster_info.append({
            'Cluster': cluster_id,
            'Number of Documents': num_documents,
            'Top Words': ', '.join(top_words),
            'Num Nodes': len(G.nodes) if G is not None else 0,
            'Dimension': len(G.edges) if G is not None else 0,
            'Diameter': diameter,
            'Density': density,
            'Cluster Coefficient': cluster_coefficient,
            'Top Words (Network)': ', '.join(top_words_network)
        })

    csv_file_path = os.path.join(output_dir, 'info_all_clusters.csv')

    df = pd.DataFrame(cluster_info)
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.exists(csv_file_path))

    print(f"Information saved to: {csv_file_path}")


# Step 7: Visualize Clusters using UMAP
def visualize_clusters_umap(embeddings, cluster_labels, output_dir):
    df_umap = pd.DataFrame(embeddings)
    df_umap['Cluster'] = cluster_labels

    reducer = umap.UMAP(n_components=2)
    umap_embeddings = reducer.fit_transform(df_umap.iloc[:, :-1])

    df_umap_final = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df_umap_final['Cluster'] = cluster_labels

    fig_umap = px.scatter(df_umap_final, x='UMAP1', y='UMAP2', color='Cluster', title='UMAP Visualization')
    
    # Save UMAP visualization as PNG
    fig_umap.write_image(os.path.join(output_dir, 'umap_visualization.png'))

    fig_umap.show()

# Step 8: Build and Visualize Word Networks for Each Cluster
def build_and_visualize_word_networks(texts, output_dir, cluster_labels, tokenizer, max_clusters=None, threshold_multiplier=4, node_size=50, edge_width=1):
    # Create graph for each cluster
    clusters = {cluster_id: nx.Graph() for cluster_id in set(cluster_labels)}

    # Extract words from each document in each cluster
    for i, text in enumerate(texts):
            cluster_id = cluster_labels[i]
            if max_clusters is not None and cluster_id >= max_clusters:
                continue
            tokens = preprocess_text(text, tokenizer).split()

            # Add edges between words with inverse average distance as edge weight
            for j in range(len(tokens) - 1):
                word1, word2 = tokens[j], tokens[j + 1]

                # Calculate average distance between occurrences
                avg_distance = j + 1  # Average distance 

                # Add edge with inverse average distance as weight
                clusters[cluster_id].add_edge(word1, word2, weight=1 / (avg_distance + 1))

    for cluster_id, G in clusters.items():
        if G is None:
            print(f"Warning: No graph (G) provided for Cluster {cluster_id}. Skipping visualization.")
            continue

        pos = nx.random_layout(G)

        # Calculate the average weight
        avg_weight = sum(edge[2]['weight'] for edge in G.edges(data=True)) / G.number_of_edges()

        # Set the threshold as a multiple of the average weight
        threshold = avg_weight * threshold_multiplier

        # Create a filtered graph by adding edges that meet the threshold
        filtered_G = nx.Graph()
        for edge in G.edges(data=True):
            if edge[2]['weight'] >= threshold:
                filtered_G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # Filter edges based on the threshold 
        edge_weights = [(u, v, w['weight']) for u, v, w in filtered_G.edges(data=True)]

        print(f"Avg Weight: {avg_weight}, Threshold: {threshold}")

        # Plot the figure
        plt.figure(figsize=(12, 10))
        nx.draw(filtered_G, pos, with_labels=True, font_size=8, node_color='skyblue', edgelist=edge_weights, edge_color='black', edge_cmap=plt.cm.Blues, width=edge_width, node_size=node_size)
        plt.title(f'Word Network - Cluster {cluster_id}')
        plt.annotate(f'Output Directory:\n{output_dir}', xy=(0.5, -0.1), xycoords="axes fraction", ha="center", color='black')
        # Save the plot as PNG
        plt.savefig(os.path.join(output_dir, f'word_network_cluster_{cluster_id}.png'))

    return clusters



# Step 9: Get Top Words per Cluster
def get_top_words_per_cluster(texts, top_words_limit=50):
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set(['at', 'and', 'in', 'th', '...'])
    all_stop_words = stop_words.union(custom_stop_words)

    words = ' '.join(texts).split()
    filtered_words = [word for word in words if word.lower() not in all_stop_words]

    word_counts = Counter(filtered_words)
    top_words = [word for word, count in word_counts.most_common(top_words_limit) if word.lower() != 'weight']
    return top_words

# Step 10: Analyze Network Characteristics
def analyze_clusters(texts, cluster_labels, clusters, output_dir):
    cluster_info = {}
    for cluster_id, G in clusters.items():
        try:
            cluster_texts = [text for i, text in enumerate(texts) if i < len(cluster_labels) and cluster_labels[i] == cluster_id]
            cluster_text = " ".join(cluster_texts)

            if G is not None:
                num_nodes = len(G.nodes)
                dimension = len(G.edges)
                diameter = nx.diameter(G)
                density = nx.density(G)
                cluster_coefficient = nx.average_clustering(G)

                # Get top words for the cluster
                top_words = get_top_words_per_cluster(cluster_texts, top_words_limit=50)

                # Write network information to CSV file
                with open(os.path.join(output_dir, f'network_info_cluster_{cluster_id}.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Cluster ID', 'Num Nodes', 'Dimension', 'Diameter', 'Density', 'Cluster Coefficient', 'Top Words'])
                    writer.writerow([cluster_id, num_nodes, dimension, diameter, density, cluster_coefficient, ', '.join(top_words)])

                # Other information
                cluster_info[cluster_id] = {
                    'G': G,
                    'Num Nodes': num_nodes,
                    'Dimension': dimension,
                    'Diameter': diameter,
                    'Density': density,
                    'Cluster Coefficient': cluster_coefficient,
                    'Top Words': top_words
                }

                # Calculate centrality measures 
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)

                # Write data to CSV files
                centrality_headers = ['Node', 'Centrality']
                with open(os.path.join(output_dir, f'betweenness_centrality_cluster_{cluster_id}.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(centrality_headers)
                    for node, betweenness in betweenness_centrality.items():
                        writer.writerow([node, betweenness])

                with open(os.path.join(output_dir, f'closeness_centrality_cluster_{cluster_id}.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(centrality_headers)
                    for node, closeness in closeness_centrality.items():
                        writer.writerow([node, closeness])

                with open(os.path.join(output_dir, f'degree_centrality_cluster_{cluster_id}.csv'), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(centrality_headers)
                    for node, degree in degree_centrality.items():
                        writer.writerow([node, degree])

                # Visualize the word network for each cluster
                if G is not None:
                    build_and_visualize_word_networks(G, cluster_id, output_dir, threshold_multiplier=4)

                # Histograms for centrality measures
                plot_histogram(G, cluster_id, output_dir)

        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")

    return cluster_info



# Step 12: Get Top Words per Cluster
def get_top_words_per_cluster(texts, top_words_limit=50):
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set(['at', 'and', 'in', 'th', '...'])
    all_stop_words = stop_words.union(custom_stop_words)

    words = ' '.join(texts).split()
    filtered_words = [word for word in words if word.lower() not in all_stop_words]

    word_counts = Counter(filtered_words)
    top_words = [word for word, count in word_counts.most_common(top_words_limit) if word.lower() != 'weight']
    return top_words

# Step 13: Plot Histograms for Centrality Measures
def plot_histogram(clusters, output_dir):
    try:
        # Create a single figure with subplots for each centrality measure
        plt.figure(figsize=(15, 5))

        for i, centrality_measure in enumerate(['degree', 'betweenness', 'closeness'], start=1):
            # Load data for both clusters
            cluster_1_data = pd.read_csv(os.path.join(output_dir, f'{centrality_measure}_centrality_cluster_1.csv'))
            cluster_2_data = pd.read_csv(os.path.join(output_dir, f'{centrality_measure}_centrality_cluster_2.csv'))

            # Create subplots for each centrality measure
            plt.subplot(1, 3, i)

            # Customize x-axis range for degree and betweenness centrality
            if centrality_measure in ['degree', 'betweenness']:
                x_axis_range = (0, 0.05) if centrality_measure == 'degree' else (0, 0.02)
                sns.histplot(data=cluster_1_data, x='Centrality', color='red', bins=20, kde=True, label='Cluster 1')
                sns.histplot(data=cluster_2_data, x='Centrality', color='blue', bins=20, kde=True, label='Cluster 2')
                plt.xlim(x_axis_range)
            else:
                sns.histplot(data=cluster_1_data, x='Centrality', color='red', bins=20, kde=True, label='Cluster 1')
                sns.histplot(data=cluster_2_data, x='Centrality', color='blue', bins=20, kde=True, label='Cluster 2')

            plt.title(f'{centrality_measure.capitalize()} Centrality Histogram')
            plt.xlabel(f'{centrality_measure.capitalize()} Centrality')
            plt.ylabel('Frequency')
            plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the entire figure as a PNG
        plt.savefig(os.path.join(output_dir, 'centrality_histogram_clusters.png'))

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error plotting histograms for centrality measures: {e}")


   
# Step 14: Main Function
def main():
    # Set the paths for the zip file and the extracted folder
    zip_file_path = '/Users/esmaisufi/Desktop/trainingdata_v3.zip'
    extracted_folder_path = '/Users/esmaisufi/Desktop/trainingdata_v3/train'

    # Set the output directory to the current directory
    output_dir = os.path.abspath(os.path.dirname(__file__))

    # Initialize BERT tokenizer and model (using BioBERT)
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")

    # Set the maximum number of samples for clustering
    max_samples_clustering = None  # None means all documents will be considered for clustering

    # Unzip the dataset
    unzip_dataset(zip_file_path, extracted_folder_path)

    # Initialize an empty dictionary for clusters
    clusters = {}

    # Iterate over the text files and process the content
    text_files = [os.path.join(extracted_folder_path, file) for file in os.listdir(extracted_folder_path) if file.endswith('.txt')]
    embeddings = []
    texts = []

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                # Preprocess and Embed Text
                embeddings.append(embed_text(text, tokenizer, model))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Number of text files: {len(text_files)}")
    print(f"Number of embeddings: {len(embeddings)}")

    if len(embeddings) > 0:
        # Perform Dimensionality Reduction
        flattened_embeddings = dimensionality_reduction(embeddings, max_samples=max_samples_clustering)

        # Automatically determine the number of clusters
        optimal_num_clusters = find_optimal_clusters_silhouette(flattened_embeddings)

    # Perform Clustering 
    max_samples_clustering = min(max_samples_clustering, len(flattened_embeddings)) if max_samples_clustering is not None else len(flattened_embeddings)
    if optimal_num_clusters <= len(texts):  # Ensure the number of clusters is valid
        cluster_labels = perform_clustering(flattened_embeddings[:max_samples_clustering], num_clusters=optimal_num_clusters)
    else:
        print("Error: Number of clusters is greater than the number of documents.")
        return
    
    # Calculate and print the silhouette score for the optimal number of clusters
    silhouette_avg = silhouette_score(flattened_embeddings, cluster_labels)
    print(f"Silhouette score for {optimal_num_clusters} clusters: {silhouette_avg}")

    # Visualize Word Networks for Each Cluster with the Threshold
    increased_threshold = 6  

    # Visualize Clusters using UMAP
    visualize_clusters_umap(flattened_embeddings, cluster_labels, output_dir)

    # Build and Visualize Word Networks for Each Cluster
    clusters = build_and_visualize_word_networks(texts, output_dir, cluster_labels, tokenizer, threshold_multiplier=increased_threshold)

    # Analyze Network Characteristics
    analyze_clusters(texts, cluster_labels, clusters, output_dir)

    # Save Cluster Information to CSV
    save_cluster_info_to_csv(texts, cluster_labels, output_dir, clusters)

    # Plot Histograms for Centrality Measures
    plot_histogram(clusters, output_dir)
    

    
if __name__ == "__main__":
    main()
