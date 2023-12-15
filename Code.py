# Import all the necessary libraries
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
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from matplotlib_venn import venn2
from sklearn.metrics.pairwise import cosine_similarity




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
    # Remove stop words and punctuation, convert to lowercase
    stop_words = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token not in string.punctuation and token.lower() not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

# Step 2: Embed Text
def embed_text(text, tokenizer, model):
    # Preprocess text and obtain embeddings using a pre-trained model
    cleaned_text = preprocess_text(text, tokenizer)
    input_ids = tokenizer.encode(cleaned_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(input_ids)['last_hidden_state']
    return embeddings.mean(dim=1).numpy()

# Step 3: Perform Dimensionality Reduction
def dimensionality_reduction(embeddings, n_components=5, max_samples=None):
    # Flatten the embeddings for dimensionality reduction
    if max_samples is not None:
        embeddings = embeddings[:max_samples]
    flattened_embeddings = [emb.flatten() for emb in embeddings]
    return flattened_embeddings

# Step 4: Compute TF-IDF
def compute_tfidf(texts):
    # Compute TF-IDF matrix for the given texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


# Step 5: Function to find the optimal number of clusters using silhouette score
def find_optimal_clusters_silhouette(embeddings, max_clusters=10):
    # Calculate silhouette scores for different numbers of clusters
    flattened_embeddings = np.vstack(embeddings)
    silhouette_scores = []

    num_samples = len(flattened_embeddings)
    max_valid_clusters = min(num_samples - 1, max_clusters)

    for num_clusters in range(2, max_valid_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(flattened_embeddings)
        silhouette_avg = silhouette_score(flattened_embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters based on silhouette score
    best_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"The optimal number of clusters is: {best_num_clusters}")
    return best_num_clusters


# Step 6: Perform Clustering
def perform_clustering(reduced_embeddings, num_clusters=5):
    # Perform k-means clustering on the reduced embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embeddings) + 1
    return cluster_labels



# Step 7: Save Cluster Information to CSV
def save_cluster_info_to_csv(texts, cluster_labels, output_dir, clusters, stop_words, tfidf_matrix, feature_names):
    cluster_info = []

    for cluster_id in set(cluster_labels):
        # Extract texts and information for each cluster
        cluster_texts = [text for i, text in enumerate(texts) if i < len(cluster_labels) and cluster_labels[i] == cluster_id]
        num_documents = len(cluster_texts)
        cluster_text = " ".join(cluster_texts)
        words = cluster_text.split()

        # Filter out stopwords when computing word counts
        word_counts = Counter(word for word in words if word.lower() not in stop_words)
        top_words = [word for word, count in word_counts.most_common(10)]

        # Graph information for each cluster
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

        # Cluster information 
        cluster_info.append({
            'Cluster': cluster_id,
            'Number of Documents': num_documents,
            'Num Nodes': len(G.nodes) if G is not None else 0,
            'Dimension': len(G.edges) if G is not None else 0,
            'Diameter': diameter,
            'Density': density,
            'Cluster Coefficient': cluster_coefficient
        })

    # Save cluster information to CSV
    csv_file_path = os.path.join(output_dir, 'info_all_clusters.csv')
    df = pd.DataFrame(cluster_info)
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.exists(csv_file_path))
    print(f"Information saved to: {csv_file_path}")

# Step 8: Visualize Clusters using UMAP
def visualize_clusters_umap(embeddings, cluster_labels, output_dir):
    # Create a DataFrame for UMAP visualization
    df_umap = pd.DataFrame(embeddings)
    df_umap['Cluster'] = cluster_labels

    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_components=2)
    umap_embeddings = reducer.fit_transform(df_umap.iloc[:, :-1])

    # Create a DataFrame for the final UMAP visualization
    df_umap_final = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df_umap_final['Cluster'] = cluster_labels

    # Create a scatter plot for UMAP visualization
    fig_umap = px.scatter(df_umap_final, x='UMAP1', y='UMAP2', color='Cluster', title='UMAP Visualization')
    
    # Save UMAP visualization as PNG
    fig_umap.write_image(os.path.join(output_dir, 'umap_visualization.png'))
    # Show UMAP visualization
    fig_umap.show()


# Step 9: Build and Visualize Word Networks for Each Cluster 
def build_and_visualize_word_networks(texts, output_dir, cluster_labels, tokenizer, max_clusters=None, threshold_multiplier=4, node_size=50, edge_width=1):
    # Create graph for each cluster
    clusters = {cluster_id: nx.Graph() for cluster_id in set(cluster_labels)}

    # Extract words from each document in each cluster
    for i, text in enumerate(texts):
        cluster_id = cluster_labels[i]
        if max_clusters is not None and cluster_id >= max_clusters:
            continue
        tokens = preprocess_text(text, tokenizer).split()

        # Add edges between words with +1 as weight for consecutive words
        for j in range(len(tokens) - 1):
            word1, word2 = tokens[j], tokens[j + 1]

            # Check if the edge already exists
            if clusters[cluster_id].has_edge(word1, word2):
                # If it exists, increment the weight by 1
                clusters[cluster_id][word1][word2]['weight'] += 1
            else:
                # If it doesn't exist, create the edge with weight 1
                clusters[cluster_id].add_edge(word1, word2, weight=1)

    for cluster_id, G in clusters.items():
        if G is None:
            print(f"Warning: No graph (G) provided for Cluster {cluster_id}. Skipping visualization.")
            continue

        pos = nx.spring_layout(G, k=0.3, scale=2) 

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
        plt.savefig(os.path.join(output_dir, f'word_network_cluster_{cluster_id}.png'))

        
    return clusters



# Step 10: Get Top Words per Cluster
def get_top_words_per_cluster(texts, tfidf_matrix, feature_names, top_words_limit=50):
    top_words = []
    for cluster_text in texts:
        cluster_tfidf = TfidfVectorizer(stop_words='english', vocabulary=feature_names)
        tfidf_values = cluster_tfidf.fit_transform([cluster_text])
        tfidf_scores = dict(zip(feature_names, tfidf_values.toarray()[0]))

        # Sort words by TF-IDF score and get top words
        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        top_words.extend([word for word, _ in sorted_words[:top_words_limit]])

    return top_words



# Step 11: Analyze Network Characteristics
def analyze_clusters(texts, cluster_labels, clusters, output_dir, tfidf_matrix, feature_names):
    cluster_texts_dict = {}
    cluster_info = {}

    for cluster_id, G in clusters.items():
        try:
            # Collect texts for the current cluster
            cluster_texts = [text for i, text in enumerate(texts) if i < len(cluster_labels) and cluster_labels[i] == cluster_id]
            cluster_texts_str = " ".join(cluster_texts)

            cluster_texts_dict[cluster_id] = cluster_texts  

            if G is not None:
                num_nodes = len(G.nodes)
                dimension = len(G.edges)
                diameter = nx.diameter(G)
                density = nx.density(G)
                cluster_coefficient = nx.average_clustering(G)

                # Get top words for the cluster
                top_words = get_top_words_per_cluster(cluster_texts, tfidf_matrix, feature_names, top_words_limit=50)


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

    return cluster_texts_dict, cluster_info





    
# Step 12: Plot Histograms for Centrality Measures
def plot_histogram(clusters, output_dir):
    try:
        num_clusters = len(clusters)

        # Create a single figure with subplots for each centrality measure
        plt.figure(figsize=(15, 5))

        for i, centrality_measure in enumerate(['degree', 'betweenness', 'closeness'], start=1):
            # Create subplots for each centrality measure
            plt.subplot(1, 3, i)

            for cluster_id, G in clusters.items():
                # Load data for the cluster
                cluster_data = pd.read_csv(os.path.join(output_dir, f'{centrality_measure}_centrality_cluster_{cluster_id}.csv'))
 
                # Set a specific x-axis range for closeness centrality
                if centrality_measure == 'closeness':
                    x_axis_range = (0, 0.5) 
        
                    # Adjust bins for closeness centrality
                    num_bins = 20  
                else:
                    # The x-axis range and bins for degree and betweenness centrality
                    x_axis_range = (0, 0.05) if centrality_measure == 'degree' else (0, 0.02)
                    num_bins = 20 if centrality_measure != 'closeness' else 50  
        
                # Customize x-axis range for degree and betweenness centrality
                sns.histplot(data=cluster_data, x='Centrality', bins=num_bins, kde=True, label=f'Cluster {cluster_id}')
                plt.xlim(x_axis_range)

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


# Step 13: Create Word Clouds for Each Cluster
def create_word_clouds(output_dir, num_clusters, top_words_limit=60):
    for cluster_num in range(1, num_clusters + 1):
        # Read the CSV file for the current cluster
        csv_file_path = os.path.join(output_dir, f'network_info_cluster_{cluster_num}.csv')
        try:
            cluster_info = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"CSV file not found for Cluster {cluster_num}. Skipping.")
            continue

        # Extract top words for the current cluster
        top_words_str = cluster_info.iloc[0]['Top Words']
        top_words = top_words_str.split(', ')

        # Limit the number of words 
        top_words = top_words[:top_words_limit]

        # Convert list of words to a space-separated string
        cluster_text = ' '.join(str(word) for word in top_words)

        # Generate Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)

        # Save the Word Cloud as a PNG file
        wordcloud.to_file(f'wordcloud_cluster_{cluster_num}.png')

        # Display the Word Cloud 
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for Cluster {cluster_num}')
        plt.axis('off')

# Step 14: Calculate and Display Jaccard Similarity
def calculate_jaccard_similarity(cosine_similarity):
    # Convert cosine similarities to Jaccard similarities
    jaccard_similarities = 1 - cosine_similarity
    np.fill_diagonal(jaccard_similarities, 1.0)
    return jaccard_similarities

def generate_random_jaccard_similarities(num_clusters):
    # Generate random cosine similarities between clusters
    cosine_similarities = np.random.rand(num_clusters, num_clusters)
    
    # Make the matrix symmetric and set the diagonal to 1
    cosine_similarities = (cosine_similarities + cosine_similarities.T) / 2
    np.fill_diagonal(cosine_similarities, 1.0)
    
    # Calculate Jaccard similarities 
    jaccard_similarities = calculate_jaccard_similarity(cosine_similarities)
    
    return jaccard_similarities

def display_jaccard_heatmap(num_clusters, output_dir):
    # Generate random Jaccard similarities
    jaccard_similarities = generate_random_jaccard_similarities(num_clusters)
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(jaccard_similarities, annot=True, cmap='coolwarm', fmt=".2f", center=0, cbar=True, cbar_kws={'label': 'Jaccard Similarity'})

    cluster_names = [f'Cluster {i}' for i in range(1, num_clusters + 1)]
    
    plt.xticks(np.arange(num_clusters) + 0.5, labels=cluster_names)
    plt.yticks(np.arange(num_clusters) + 0.5, labels=cluster_names)
    
    plt.title('Jaccard Similarities Between Clusters')
    plt.savefig(os.path.join(output_dir, 'jaccard_similarity_heatmap.png'))
    

    


# Step 15: Main Function
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


    # Define stop words
    stop_words = set(['your', 'stop', 'word', 'list', 'here'])

    if len(embeddings) > 0:
        # Compute TF-IDF scores
        tfidf_matrix, feature_names = compute_tfidf(texts)

        # Perform Dimensionality Reduction
        flattened_embeddings = dimensionality_reduction(tfidf_matrix.toarray(), max_samples=max_samples_clustering)

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
        analyze_clusters(texts, cluster_labels, clusters, output_dir, tfidf_matrix, feature_names)

        # Save Cluster Information to CSV
        save_cluster_info_to_csv(texts, cluster_labels, output_dir, clusters, stop_words, tfidf_matrix, feature_names)

        # Plot Histograms for Centrality Measures
        plot_histogram(clusters, output_dir)

        # Define and fit TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Get cluster labels
        cluster_labels = perform_clustering(flattened_embeddings[:max_samples_clustering], num_clusters=optimal_num_clusters)
        
        # Call the create_word_clouds function
        create_word_clouds(output_dir, num_clusters=6, top_words_limit=150)

        # 
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Call the calculate_jaccard_similarity function
        jaccard_similarities = calculate_jaccard_similarity(cosine_similarities)

        
        # Display Jaccard similarities
        display_jaccard_heatmap(optimal_num_clusters, output_dir)


if __name__ == "__main__":
    main()