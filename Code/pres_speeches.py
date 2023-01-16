import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.cluster import KMeans

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from seg_algorithm import get_optimal_splits, get_segmented_sentences
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def squared_sum(x):
    """ return 3 rounded square rooted value """
    return round(np.sqrt(sum([a * a for a in x])), 3)


def cos_similarity(x, y):
    """ return cosine similarity between two lists """
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = squared_sum(x) * squared_sum(y)
    return round(numerator / float(denominator), 3)


url = "https://raw.githubusercontent.com/yahya010/DocClustering/main/Pres_Speeches/corpus.csv"
dataset = pd.read_csv(url)
p = 0.65  # increase p = no of segments decreases

stop_words = set(stopwords.words('english'))
transcripts = dataset.transcripts
tokenized_transcripts = pd.DataFrame(index=range(44), columns=['Sentences'])

fullTranscripts = []
filteredtranscript = []
originalTranscript_list = []  # contains 44 transcripts
transcript_list = []  # contains 44 transcripts
for transcript in transcripts[0:1]:
    transcript = sent_tokenize(transcript)
    for sentence in transcript:
        fullTranscripts.append(sentence)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')
embeddings = model.encode(fullTranscripts)

segmentation = get_optimal_splits(embeddings, p)  # Splits.
segmented_txt = get_segmented_sentences(fullTranscripts, segmentation)
segment_list = []
for segment in segmented_txt:
    segment_list.append('Segment ' + str(segmented_txt.index(segment)) + ': ' + str(segment))

seglistDF = pd.DataFrame(segment_list)
seglistDF.to_csv('fullSegmentationList.csv')

heatmap = np.zeros(shape=(len(embeddings), len(embeddings)))
for i in range(len(embeddings)):
    sent = embeddings[i]
    for j in range(len(embeddings)):
        sent2 = embeddings[j]
        cosSim = cos_similarity(sent, sent2)
        heatmap[i, j] = cosSim
# print(heatmap)
heatmap = pd.DataFrame(data=heatmap)
heatmap.to_csv('heatmap.csv')
# print("done")

embedding = embeddings
pca = PCA(n_components=0.95)
reduced_embedding = pca.fit_transform(embedding)
# print(reduced_embedding.shape)

heatmapPost = np.zeros(shape=(len(reduced_embedding), len(reduced_embedding)))
for i in range(len(reduced_embedding)):
    sentPost = reduced_embedding[i]
    for j in range(len(reduced_embedding)):
        sent2Post = reduced_embedding[j]
        cosSimPost = cos_similarity(sentPost, sent2Post)
        heatmapPost[i, j] = cosSimPost
# print(heatmapPost)
heatmapPost = pd.DataFrame(data=heatmapPost)
heatmapPost.to_csv('reduced_heatmap_0.csv')

# Comparison of Within Cluster Sum of Squares (wcss) for different cluster sizes
# wcss = []
# for i in range(1, 50):
#     kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
#     kmeans_pca.fit(reduced_embedding)
#     wcss.append(kmeans_pca.inertia_)
#
# plt.figure(figsize=(10, 8))
# plt.plot(range(1, 50), wcss, marker='o', linestyle='--')
# plt.xlabel('Number of Clusters')
# plt.ylabel('K-means with PCA clustering')
# plt.show()

# KMeans algorithm
kmeans_clusters = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42, max_iter=300, algorithm='auto',
                         copy_x=True, tol=0.0001, verbose=0)

label = kmeans_clusters.fit_predict(reduced_embedding)

cluster_labels = np.unique(label)

# plotting the results:
for i in cluster_labels:
    plt.scatter(reduced_embedding[label == i, 0], reduced_embedding[label == i, 1], label=i)
plt.legend()
plt.show()



