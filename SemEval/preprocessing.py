### PEP-8 Standard for imports ####

import os
import time

import matplotlib

matplotlib.use('Agg')

from utils.prepare_data import *
from utils.model_helper import *


def preprocess(config):
    """
    Preprocess the data.

    :param config:
    :return preprocessed data:
    """
    np.random.seed(123)
    tf.set_random_seed(422)

    ### Create the model directory ###
    path_train = os.path.join(config['model_dir'], config['name'], config['train_data'])
    path_test = os.path.join(config['model_dir'], config['name'], config['test_data'])

    print('Loading test and training data')

    ### Load test data ###
    x_train, y_train_1, y_train_2, y_train_3, ids_train = load_data("./data/" + config['train_data'] + ".csv",
                                                                    one_hot=False, sample_ratio=config['train_ratio'],
                                                                    path_name=path_train + '_shuffled.csv')

    length = len(y_train_1)

    x_test, y_test_1, y_test_2, y_test_3, ids_test = load_data("./data/" + config['test_data'] + ".csv", one_hot=False,
                                                               sample_ratio=config['test_ratio'],
                                                               path_name=path_test + '_shuffled.csv')

    ###  Load the attention Data ###
    print('Loading attention data')
    data_attention = pd.read_csv('./data/nora_output.csv')
    x_attention, y_attention = data_attention['Content'], data_attention['Activation'].map(
        lambda x: np.array(x.split(), dtype=np.float32))

    print('Cleaning the data')
    words = []
    f = open('./data/glove.twitter.27B.' + str(config['fund_embed_dim']) + 'd.txt', encoding='utf8')
    for line in f:
        values = line.split()
        words += [values[0]]
    f.close()

    start_time = time.perf_counter()
    ### Cleaning the data ###
    if config['preprocessing']['clean']:
        x_train = np.array(list(x_train.fillna('unk').map(lambda x: ' '.join(lemmatizer(
            clean_text(x, remove_stop=config['preprocessing']['remove_stop'],
                       formal_filter=config['preprocessing']['formal_filter']))))))

        duration = (time.perf_counter() - start_time)
        print('It took ' + str(duration))
        x_test = np.array(x_test.fillna('unk').map(lambda x: ' '.join(lemmatizer(
            clean_text(x, remove_stop=config['preprocessing']['remove_stop'],
                       formal_filter=config['preprocessing']['formal_filter'])))))

        print('finished cleaning training data')

        x_attention = np.array(list(x_attention.fillna('unk').map(lambda x: ' '.join(lemmatizer(
            clean_text(x, remove_stop=config['preprocessing']['remove_stop'],
                       formal_filter=config['preprocessing']['formal_filter']))))))

    print('Extracting POS')
    ### Extract POS ###
    if config['FIRST_LAYER']['pos_include']:
        x_test_POS = pd.DataFrame({'x': x_test})['x'].fillna('').map(lambda x: ' '.join(pos_tagging(x)))
        x_train_POS = pd.DataFrame({'x': x_train})['x'].fillna('').map(lambda x: ' '.join(pos_tagging(x)))
        x_attention_POS = pd.DataFrame({'x': x_attention})['x'].fillna('').map(lambda x: ' '.join(pos_tagging(x)))
    else:
        x_test_POS = pd.DataFrame({'x': x_test})['x'].fillna('').map(lambda x: '_')
        x_train_POS = pd.DataFrame({'x': x_train})['x'].fillna('').map(lambda x: '_')
        x_attention_POS = pd.DataFrame({'x': x_attention})['x'].fillna('').map(lambda x: '_')

    x_test_POS, x_train_POS, vocab_size, tokenizer = data_preprocessing(x_test_POS, x_train_POS, max_len=1000)

    ### Augmenting the data ###
    assert not (config['preprocessing']['augment'] and config['preprocessing']['replace'])
    if config['preprocessing']['augment']:
        print('Augmenting the data')
        x_train = np.array(list(map(lambda x: entity_augmentation(x).lower(), x_train)))
        x_test = np.array(list(map(lambda x: entity_augmentation(x).lower(), x_test)))
        x_attention = np.array(list(map(lambda x: entity_augmentation(x).lower(), x_attention)))
    if config['preprocessing']['replace']:
        print('Replacing the data')
        x_train = np.array(list(map(lambda x: entity_replacement(x).lower(), x_train)))
        x_test = np.array(list(map(lambda x: entity_replacement(x).lower(), x_test)))
        x_attention = np.array(list(map(lambda x: entity_replacement(x).lower(), x_attention)))

    x_test = np.array(list(map(lambda x: 'unk' if x == '' else x, x_test)))
    x_train = np.array(list(map(lambda x: 'unk' if x == '' else x, x_train)))

    print('Converting the data to tokens')
    ### Converting the data to tokens ###    
    voc_size_test, embed_test, x_test, x_test_org = text_to_glove(x_test, filters='', dim=str(config['fund_embed_dim']))
    voc_size_train, embed_train, x_train, x_train_org = text_to_glove(x_train, filters='',
                                                                      dim=str(config['fund_embed_dim']))

    print('Attention data to tokens')
    ### Attention data to tokens ###
    voc_size_attention, embed_attention, x_attention, x_attention_org = text_to_glove(x_attention, filters='',
                                                                                      dim=str(config['fund_embed_dim']))

    print('Extract the sequence lengths')
    ###   Extract the sequence lengths ###
    seq_len_train = np.array(list(map(lambda x: len(x.split()), x_train_org)))
    seq_len_test = np.array(list(map(lambda x: len(x.split()), x_test_org)))

    seq_len_attention = np.array(list(map(lambda x: len(x.split()), x_attention_org)))

    print("Preprocessing")
    print("train size: ", len(x_train))

    ### Convert all the data to array format ###

    x_train_org = np.array(x_train_org)
    x_test_org = np.array(x_test_org)
    x_attention_org = np.array(x_attention_org)

    x_test = np.array(x_test)
    x_train = np.array(x_train)
    x_attention = np.array(x_attention)

    y_test_1 = np.array(y_test_1)
    y_test_2 = np.array(y_test_2)
    y_test_3 = np.array(y_test_3)
    y_train_1 = np.array(y_train_1)
    y_train_2 = np.array(y_train_2)
    y_train_3 = np.array(y_train_3)
    y_attention = y_attention

    x_attention_POS = np.array(list(tokenizer.transform(x_attention_POS)))

    # split dataset to test and dev
    x_test_org, x_dev_org, x_test, x_dev, x_test_POS, x_dev_POS, y_test_1, y_dev_1, \
    y_test_2, y_dev_2, y_test_3, y_dev_3, seq_len_test, seq_len_dev, dev_size, test_size = \
        split_dataset(x_test_org, x_test, x_test_POS, y_test_1, y_test_2, y_test_3, seq_len_test, 1)

    print("Validation Size: ", dev_size)
    sample_size = len(y_train_1)
    config['sample_size'] = sample_size

    config['reuse'] = False

    return (x_train_org, x_train, x_train_POS, y_train_1, y_train_2, y_train_3, seq_len_train, embed_train,
            x_dev_org, x_dev, x_dev_POS, y_dev_1, y_dev_2, y_dev_3, seq_len_dev, embed_test, x_attention_org,
            x_attention, x_attention_POS, y_attention, seq_len_attention, embed_attention, config, length,
            ids_train, ids_test)
