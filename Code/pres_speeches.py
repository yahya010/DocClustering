import pandas as pd
import numpy as np
import matplotlib
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
# Algorithm imports.
from seg_algorithm import get_optimal_splits, get_segmented_sentences
import nlu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#nlu.load('embed_sentence.bert embed_sentence.electra use')
url = "https://raw.githubusercontent.com/yahya010/DocClustering/main/Pres_Speeches/corpus.csv"
dataset = pd.read_csv(url)
print("here")
p = 0.65 #increase p = no of segments decreases

#dataset.to_csv("test1.csv")
#dataset = dataset.rename(columns={"Name", "Party", "Transcript"})
#Washington = dataset.loc[dataset['Unnamed: 0'] == 'George Washington', 'transcripts']

transcripts = dataset.transcripts
tokenized_transcripts = pd.DataFrame(index=range(44), columns=['Sentences'])

transcript_list = [] # contains 44 transcripts
for transcript in transcripts:
    transcript_list.append(sent_tokenize(transcript))
#print(transcript_list[0]) # has washington
#washington = pd.DataFrame(transcript_list[0])
#washington.to_csv('washingtonagain.csv')


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#embeddings = model.encode(transcript_list[10])
embeddings = model.encode(transcript_list[10])
# #embeddings = pd.DataFrame(embeddings)
# #embeddings.to_csv('embeddings.csv')

segmentation  = get_optimal_splits(embeddings, p) # Splits.
segmented_txt = get_segmented_sentences(transcript_list[10], segmentation)
segment_list = []
for segment in segmented_txt:
    segment_list.append('Segment ' + str(segmented_txt.index(segment)) + ': ' + str(segment))
    #print('Segment ' + str(segmented_txt.index(segment)) + ': ' + str(segment))

#pipe = nlu.load('embed_sentence.bert')
seglistDF = pd.DataFrame(segment_list)
seglistDF.to_csv('10SegmentList.csv')
#predictions = embeddings
def squared_sum(x):
  """ return 3 rounded square rooted value """
 
  return round(np.sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """
 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

embeddings0 = model.encode(transcript_list[0]) #fix this to embed everything at once instead of transcript
#print(cos_similarity(embeddings[0], embeddings[1]))
heatmap = np.zeros(shape=(len(embeddings0), len(embeddings0)))
for i in range(len(embeddings0)):
    sent = embeddings0[i]
    for j in range(len(embeddings0)):
        sent2 = embeddings0[j]
        cosSim = cos_similarity(sent, sent2)
        heatmap[i,j] = cosSim
print(heatmap)
heatmap = pd.DataFrame(data=heatmap)
heatmap.to_csv('heatmap.csv')
print("done")