import tensorflow_hub as hub
import tensorflow_text as text

# Local pretrained model's URL
preprocess_url = "C:/Users/ADMIN/Downloads/bert_en_uncased_preprocess_3"
encoder_url = "C:/Users/ADMIN/Downloads/small_bert_bert_en_uncased_L-6_H-768_A-12_2"

bert_preprocess_model = hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)


def preprocess(sentence):
    # Tokenize a sentence
    return bert_preprocess_model(sentence)


def word_piece(preprocess_sentence):
    # Return the attribute vector of a sentence
    return bert_model(preprocess_sentence)['pooled_output']
