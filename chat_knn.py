import random

import numpy as np
import json

from sklearn.neighbors import KNeighborsClassifier

from bert import preprocess, word_piece

# Number of neighbors
k = 1
bot_name = "Chatbot (k-nn)"

with open('training.json', 'r') as f:
    intents = json.load(f)

all_sentences = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # add to our sentence list
        all_sentences.append(pattern)
        # add to xy pair
        xy.append((pattern, tag))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_sentences))

# create training data
all_sentences = preprocess(all_sentences)
X_train = word_piece(all_sentences)
Y_train = []
for (pattern_sentence, tag) in xy:
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# k neighbors classification
neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='cosine')
neigh.fit(X_train, Y_train)


def classify(msg):
    # preprocess message
    X = [msg]
    X = preprocess(X)
    X = word_piece(X)
    X = np.array(X)

    # classify message
    lable = neigh.predict(X)
    # print(tags[lable[0]])
    distance = np.average(neigh.kneighbors(X, n_neighbors=k)[0][0])
    print(distance)

    if distance < 1:
        return tags[lable[0]]

    return "None"


def get_response(msg):
    tag = classify(msg)

    if tag != "None":
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "I do not understand..."
