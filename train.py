import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from bert import preprocess, word_piece
from model import NeuralNet

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
    # y: PyTorch CrossEntropyLoss needs only class labels
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# print(X_train)

# Hyper-parameters
num_epochs = 210
batch_size = 16
learning_rate = 0.001
input_size = len(X_train[0])    # Length of the attribute vector - 768
hidden_size = 512
output_size = len(tags)         # Number of tags
print(f"Input size: {input_size} \nOutput size: {output_size}")


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_sentences": all_sentences,
    "tags": tags
}

FILE = "neural_net.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')