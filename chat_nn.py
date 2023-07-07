import random
import json

import numpy as np
import torch

from model import NeuralNet
from bert import preprocess, word_piece

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('training.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "neural_net.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_sentences = data["all_sentences"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chatbot (neural_network)"


def classify(msg):
    X = [msg]
    X = preprocess(X)
    X = word_piece(X)
    X = np.array(X)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() > 0.95:
        return tag

    return "None"


def get_response(msg):
    tag = classify(msg)

    if tag != "None":
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "I do not understand..."
