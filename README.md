# Chatbot

## Training the neural network

Download the pretrained BERT preprocess model [here][preprocess].

[preprocess]: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3

Download one of the BERT pretrained models (L: Number of layers, H: Size of the attribute vector) from the table below:

|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**][2_128]|[2/256][2_256]|[2/512][2_512]|[2/768][2_768]|
| **L=4**  |[4/128][4_128]|[**4/256 (BERT-Mini)**][4_256]|[**4/512 (BERT-Small)**][4_512]|[4/768][4_768]|
| **L=6**  |[6/128][6_128]|[6/256][6_256]|[6/512][6_512]|[6/768][6_768]|
| **L=8**  |[8/128][8_128]|[8/256][8_256]|[**8/512 (BERT-Medium)**][8_512]|[8/768][8_768]|
| **L=10** |[10/128][10_128]|[10/256][10_256]|[10/512][10_512]|[10/768][10_768]|
| **L=12** |[12/128][12_128]|[12/256][12_256]|[12/512][12_512]|[**12/768 (BERT-Base)**][12_768]|


[2_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2
[2_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-2/2
[2_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-2/2
[2_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-2/2
[4_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2
[4_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-2/2
[4_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-2/2
[4_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-2/2
[6_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2
[6_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-2/2
[6_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-2/2
[6_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-2/2
[8_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/2
[8_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-2/2
[8_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-2/2
[8_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-2/2
[10_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/2
[10_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-2/2
[10_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-2/2
[10_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-2/2
[12_128]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/2
[12_256]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-2/2
[12_512]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-2/2
[12_768]: https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-2/2

*Note: You can use the URL of the model without having to download, but downloading is recommended for reducing training time.*

Replace the URLs in **bert.py** by online URLs or according to the location of your local models as follows:

```sh
preprocess_url = "C:/Users/ADMIN/Downloads/bert_en_uncased_preprocess_3"
encoder_url = "C:/Users/ADMIN/Downloads/small_bert_bert_en_uncased_L-6_H-768_A-12_2"
```

Hyper-parameters of the neural network and training process can be changed in **train.py**:

```sh
# Hyper-parameters
num_epochs = 1500
batch_size = 128
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 64
output_size = len(tags)
```

After importing necessary modules in **requirements.txt** and running **train.py**, the neural network is stored in **data.pth**.

```sh
python .\train.py
```

## Running the chatbot

Run the chatbot app by executing **app.py**:
```sh
python .\app.py
```