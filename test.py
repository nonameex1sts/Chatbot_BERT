import json
from chat_knn import classify

with open('test.json', 'r') as json_data:
    intents = json.load(json_data)

count_total = 0
count_correct = 0
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        count_total += 1
        if classify(pattern) == tag:
            count_correct += 1
        else:
            print(f"Classified: {classify(pattern)} \tActual tag: {tag}")

print(count_correct/count_total)