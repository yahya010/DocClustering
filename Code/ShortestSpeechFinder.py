import csv
import string

import nltk
import pandas as pd
import cleanlab
from cleanlab.outlier import OutOfDistribution
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

NUM_SHORTEST_SPEECHES = 7

df = pd.read_csv("https://raw.githubusercontent.com/yahya010/DocClustering/main/Pres_Speeches/presidential_speeches.csv",
                 usecols=['President', 'Speech Title', 'Transcript'])

speechLengthMap = []
for val in df.values:
    speech = val[2]
    if not isinstance(speech, str):
        continue

    sentences = nltk.sent_tokenize(speech)
    # stop_words = set(stopwords.words('english'))
    #
    # for s in sentences:
    #     s = [word for word in s if word.lower() not in stop_words]

    # exclude = string.punctuation.replace(".", "")
    # exclude = exclude.replace("!", "")
    # exclude = exclude.replace("?", "")
    # filtered_words = no_stopwords.translate(str.maketrans('', '', exclude))

    speechLengthMap.append((val[0], sentences, len(sentences)))

speechLengthMap.sort(key=lambda x: x[2])
# store speeches where there are at least 20 sentences
filteredTranscript = [speech[1] for speech in speechLengthMap if speech[2] >= 30]

# for i in range(NUM_SHORTEST_SPEECHES):
#     print(speechLengthMap[i])

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')


unrelated_sent_map = []

for speech in filteredTranscript[:NUM_SHORTEST_SPEECHES - 1] + \
              filteredTranscript[NUM_SHORTEST_SPEECHES + 1:NUM_SHORTEST_SPEECHES + 2]:
    embeddings = model.encode(speech)

    ood = OutOfDistribution()
    train_outlier_scores = ood.fit_score(features=embeddings)
    top_train_outlier_idxs = train_outlier_scores.argsort()[:10]

    unrelated_sent_map.append((speech, [speech[topId] for topId in top_train_outlier_idxs]))

# print(unrelated_sent_map[0])
with open('Data_Output/Speech_Outlier_Data.txt', 'w') as outputFile:
    for i, sent_map in enumerate(unrelated_sent_map):
        outputFile.write(str(i + 1) + ": " + ".".join(sent_map[0]) + "\n\n")

        for j, unrelated_sent in enumerate(sent_map[1]):
            outputFile.write("{0}: {1}\n".format(j + 1, unrelated_sent))

        outputFile.write("\n\n\n")
