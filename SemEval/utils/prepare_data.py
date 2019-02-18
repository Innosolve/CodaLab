### PEP - 8 Import Standard ###

import re

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

names = ["class", "title", "content"]

nlp = spacy.load('en')


def entity_augmentation(text):
    """
    Surround entities with entity type.

    :param text:
    :return string:
    """
    doc = nlp(text)
    arr = []
    cursor = 0
    for ent in doc.ents:
        start = ent.start_char
        end = ent.end_char
        if start == 0:
            arr = str(ent.label_) + ' ' + text[:end] + ' ' + str(ent.label_)
        else:
            arr = text[cursor:start] + str(ent.label_) + ' ' + text[start:end] + ' ' + str(ent.label_)
        cursor = end
    arr = text[cursor:-1]
    return arr


def entity_replacement(text):
    """
    Replace entities with entity type.

    :param text:
    :return string:
    """
    if type(text) == np.str_:
        print(text)
        text = str(text)
    doc = nlp(text)
    arr = []
    cursor = 0
    for ent in doc.ents:
        start = ent.start_char
        end = ent.end_char
        if start == 0:
            if ent.label_ == 'norp':
                arr = str('group')
            else:
                arr = str(ent.label_)
        else:
            if ent.label_ == 'norp':
                arr = str('group')
            else:
                arr = text[cursor:start] + str(ent.label_)
        cursor = end
    arr = text[cursor:-1]
    return arr


def pos_tagging(text):
    """
    List the entities in a string.

    :param text:
    :return list:
    """
    arr = []
    doc = nlp(text)
    for token in doc:
        arr += [token.pos_]
    return arr


def lemmatizer(text):
    """
    Lemmatize words in a string.

    :param text:
    :return string:
    """
    arr = []
    doc = nlp(text)
    for token in doc:
        arr += [token.lemma_]
    return arr


def max_split(array):
    """
    Return the size of the longest string in a sequence.

    :param array:
    :return int:
    """
    return np.max(np.array(list(map(lambda row: len(row.split()), array))))


def to_one_hot(y, n_class):
    """
    Convert labels to one-hot encoded vectors.

    :param y:
    :param n_class:
    :return vector:
    """
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, path_name, sample_ratio=1, n_class=2, names=names, one_hot=True):
    """
    Load data from csv file.

    :param file_name:
    :param path_name:
    :param sample_ratio:
    :param n_class:
    :param names:
    :param one_hot:
    :return:
    """
    np.random.seed(213)
    csv_file = pd.read_csv(file_name)
    print(csv_file.columns)
    shuffle_csv = csv_file.sample(frac=sample_ratio)

    x = pd.Series(shuffle_csv["tweet"])

    y_1 = pd.Series(shuffle_csv["subtask_a"])
    y_2 = pd.Series(shuffle_csv["subtask_b"])
    y_3 = pd.Series(shuffle_csv["subtask_c"])
    ids = pd.Series(shuffle_csv["ids"])

    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y_1, y_2, y_3, ids


def data_preprocessing(train, test, max_len):
    """
    Early version of data preprocessor.

    :param train:
    :param test:
    :param max_len:
    :return data:
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_

    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)

    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab_size, vocab_processor


def data_preprocessing_v2(train, test, max_words=50000):
    """
    Second version of data preprocessor.

    :param train:
    :param test:
    :param max_words:
    :return data:
    """
    train = np.array([' '.join(tf.keras.preprocessing.text.text_to_word_sequence(row)) for row in train])
    test = np.array([' '.join(tf.keras.preprocessing.text.text_to_word_sequence(row)) for row in test])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token='UNK')
    tokenizer.fit_on_texts(train)

    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)

    return np.array(train_idx), np.array(test_idx), max_words + 2


def data_preprocessing_with_dict(train, test, max_len):
    """
    Data preprocessor.

    :param train:
    :param test:
    :param max_len:
    :return:
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test_org, x_test, x_test_POS, y_test_1, y_test_2, y_test_3, seq_len, dev_ratio=1):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    dev_size = (int)(test_size * dev_ratio)

    x_dev_org = x_test_org[:dev_size]
    x_test_org = x_test_org[dev_size:]

    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]

    x_dev_POS = x_test_POS[:dev_size]
    x_test_POS = x_test_POS[dev_size:]

    y_dev_1 = y_test_1[:dev_size]
    y_dev_2 = y_test_2[:dev_size]
    y_dev_3 = y_test_3[:dev_size]
    y_test_1 = y_test_1[dev_size:]
    y_test_2 = y_test_2[dev_size:]
    y_test_3 = y_test_3[dev_size:]

    seq_len_dev = seq_len[:dev_size]
    seq_len_test = seq_len[dev_size:]
    return x_test_org, x_dev_org, x_test, x_dev, x_test_POS, x_dev_POS, y_test_1, y_dev_1, y_test_2, y_dev_2, y_test_2, \
           y_dev_3, seq_len_test, seq_len_dev, dev_size, test_size - dev_size


def clean_text(text, formal_filter, remove_stop):  # string.punctuation):
    """
    Clean the tweets.

    :param text:
    :param formal_filter:
    :param remove_stop:
    :return cleaned tweets:
    """
    # Remove puncuation
    #     for punc in filters:
    #         text = text.replace(punc, '')
    #     text = ' '.join(re.findall('[A-Z][^A-Z]*', text))
    ###REMOVE URL and EMOJI (partially satisfied)#####
    text = re.sub(r'[a-z]*[:.]+\S+', " url ", text)
    emojipattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags=re.UNICODE)
    text = emojipattern.sub(r'', text)

    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)

    ## Convert words to lower case and split them
    text = text.lower()
    # text = ' '.join(list(map(lambda x: spelling_correction(x.replace('#',''), words) if '#' in x else x,
    # text.split())))

    text = text.split()
    ## Remove stop words
    if remove_stop:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    if formal_filter:
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)
    if type(text) != str:
        print(text)

    return text


def sequence_to_text(seq, reverse_word_map):
    """
    Converts a sequence to text.

    :param seq:
    :param reverse_word_map:
    :return text:
    """
    seq = np.array(seq)
    text = map(lambda x: reverse_word_map.get(x), seq)
    return ' '.join(list(text))


def text_to_glove(dataframe, filters='', dim=None):
    """
    Converts tweets to glove embeddings.

    :param dataframe:
    :param filters:
    :param dim:
    :return embedding:
    """
    if filters == '':
        tokenizer = Tokenizer(filters=filters)  # num_words= vocabulary_size)
    else:
        tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataframe)
    vocabulary_size = len(tokenizer.word_index)

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    sequences = tokenizer.texts_to_sequences(dataframe)
    filtered_text = np.array([sequence_to_text(seq, reverse_word_map) for seq in sequences])
    embeddings_index = {}

    f = open('./data/glove.twitter.27B.' + dim + 'd.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocabulary_size + 1, int(dim)), np.float32)
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                embedding_matrix[index] = np.zeros(int(dim), dtype=np.float32)  # embeddings_index.get('unk')
    n = sum(list(map(lambda row: 0 == np.sum(row), embedding_matrix)))

    return n, embedding_matrix, sequences, filtered_text


def hashtag_splitter(text, words):
    """
    Splits hashtags to interpretable sequence of words.

    :param text:
    :param words:
    :return splitted hashtag string:
    """
    multi_word = []
    text_template = text
    while len(text_template) != 0:
        text = text_template
        word = ''
        current_word = ''
        for char in text:
            word += char
            if word in words:
                current_word = word

        if len(current_word) > 0:
            multi_word += [current_word]
            pointer = len(current_word)
            text_template = text_template[pointer:]
        else:
            text_template = text_template[1:]
    if len(multi_word) != 0:
        result = ' '.join(multi_word)
    else:
        result = 'unk'
    return result