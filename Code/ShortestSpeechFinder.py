import csv
import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import cleanlab
from cleanlab.outlier import OutOfDistribution
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Static Variables
NUM_SHORTEST_SPEECHES = 1
MIN_SPEECH_SENTENCES = 80
OUTLIER_COUNT = 20

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
filteredTranscript = [speech[1] for speech in speechLengthMap if speech[2] >= MIN_SPEECH_SENTENCES]

# for i in range(NUM_SHORTEST_SPEECHES):
#     print(speechLengthMap[i])

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')


unrelated_sent_map = []
for speech in filteredTranscript[:NUM_SHORTEST_SPEECHES]:
    embeddings = model.encode(speech)

    ood = OutOfDistribution()
    train_outlier_scores = ood.fit_score(features=embeddings)
    top_train_outlier_idxs = train_outlier_scores.argsort()[:OUTLIER_COUNT]

    speech = [str(train_outlier_scores[i]) + ": " + sentence for i, sentence in enumerate(speech)]

    unrelated_sent_map.append((speech, [speech[topId] for topId in top_train_outlier_idxs],
                               [str(train_outlier_scores[topId]) for topId in top_train_outlier_idxs]))

    # plot histogram
    # plt.hist(train_outlier_scores, bins=15, alpha=0.5, ec='black')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylabel('Frequency', fontsize='16')
    # plt.title('Out of Distribution Scores', fontsize='20')
    # plt.show()
    #
    # # gran plot
    # plt.bar(range(1, len(train_outlier_scores) + 1), sorted(train_outlier_scores, reverse=True), alpha=0.5)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylabel('Score', fontsize='16')
    # plt.title('Out of Distribution Scores', fontsize='20')
    # plt.show()


# print(unrelated_sent_map[0])
with open('Data_Output/Speech_Outlier_Data.txt', 'w') as outputFile:
    for i, sent_map in enumerate(unrelated_sent_map):
        outputFile.write(str(i + 1) + ": " + " ".join(sent_map[0]) + "\n\n")

        for j, unrelated_sent in enumerate(sent_map[1]):
            outputFile.write("{0} {1}\n".format(j + 1, unrelated_sent))

        outputFile.write("\n\n\n")
