### PEP-8 Standard for imports ####
import json
import os

import argparse
import matplotlib
import pandas as pd

matplotlib.use('Agg')

from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import *
from utils.helper import *
from utils.model_helper import *
from preprocessing import preprocess


def main(config):
    """
    Do inference on preselected model.

    :param config:
    """
    print("Inference model estimates.")
    i = config['summary_num']
    if config['write_data']:
        x_train_org, x_train, x_train_POS, y_train_1, y_train_2, y_train_3, seq_len_train, embed_train, x_dev_org, \
        x_dev, x_dev_POS, y_dev_1, y_dev_2, y_dev_3, seq_len_dev, embed_test, x_attention_org, x_attention, \
        x_attention_POS, y_attention, seq_len_attention, embed_attention, config, length, _, ids_dev = \
            preprocess(config)
        pd.DataFrame({"x_dev_org": x_dev_org, "x_dev": x_dev, "x_dev_POS": list(map(lambda x: list(x), x_dev_POS)), \
                      "y_dev_1": y_dev_1, "y_dev_2": y_dev_2, "y_dev_3": y_dev_3, "seq_len_dev": seq_len_dev, \
                      'ids': ids_dev}).to_csv('./data/evaluate/dev_csv.csv', index=False)
        pd.DataFrame({'embed_test': list(map(lambda x: x.tolist(), \
                                             embed_test))}).to_csv('./data/evaluate/embed_test_csv.csv')
    else:
        dev_csv = pd.read_csv('./data/evaluate/dev_csv.csv')
        x_dev_org, x_dev, x_dev_POS, y_dev_1, y_dev_2, y_dev_3, seq_len_dev, ids_dev = dev_csv['x_dev_org'].fillna(
            '').values, dev_csv['x_dev'].map(
            lambda x: np.array(x[1:-1].replace('\n', '').split(','), dtype=np.int32) if x != '[]' else [0]).values, \
                                                                                       dev_csv['x_dev_POS'].map(
                                                                                           lambda x: np.asarray(
                                                                                               x[1:-1].replace('\n',
                                                                                                               '').split(
                                                                                                   ','),
                                                                                               dtype=np.int32) if x != '[]' else [
                                                                                               0]).values, dev_csv[
                                                                                           'y_dev_1'].values, dev_csv[
                                                                                           'y_dev_2'].values, dev_csv[
                                                                                           'y_dev_3'].values, dev_csv[
                                                                                           'seq_len_dev'].values, \
                                                                                       dev_csv['ids']

        embed_test = pd.read_csv('./data/evaluate/embed_test_csv.csv')['embed_test'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.float32)).values
        n = len(embed_test)
        embed_test = np.reshape(np.concatenate(embed_test), [n, config['fund_embed_dim']])

        config['sample_size'] = len(x_dev)
        config['reuse'] = False

    _, c = np.unique(y_dev_1, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_a'] = tot / np.array(number_classes * c)

    _, c = np.unique(y_dev_2, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_b'] = tot / np.array(number_classes * c)

    _, c = np.unique(y_dev_3, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_c'] = tot / np.array(number_classes * c)

    print(config)

    ### During CROSS Validation ###

    # Dataset Test
    x_dev_pad = pad_sequences(x_dev, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
    x_dev_POS_pad = pad_sequences(x_dev, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
    dataset_dev = tf.data.Dataset.from_tensor_slices((x_dev_org, x_dev_pad, x_dev_POS_pad, y_dev_1, y_dev_2, y_dev_3, \
                                                      seq_len_dev))
    dataset_dev = dataset_dev.repeat()
    dataset_dev = dataset_dev.batch(2)

    iterator_dev = dataset_dev.make_one_shot_iterator()
    next_element_dev = iterator_dev.get_next()

    config['reuse'] = False

    ###################
    # Inference Model
    ###################
    with tf.name_scope("Validation"):
        # Create model
        inferModel = Model(config)
        inferModel.is_training = False
        inferModel.build_graph()
        inferModel.build_dense()
        inferModel.build_loss()

    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    saver_restore = tf.train.Saver()

    session = tf.Session()
    if config['checkpoint_id'] == None:
        path = tf.train.latest_checkpoint(os.path.join('graphs', config['name'], 'summary_' + str(i)))
    else:
        path = config['checkpoint_id']
    saver_restore.restore(session, path)
    # Create session object
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    # Add the ops to initialize variables.
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually initialize the variables
    # session.run(init_op)

    if config['checkpoint_id'] is None:
        path = tf.train.latest_checkpoint(os.path.join('graphs', config['name'], 'summary_' + str(i)))
    else:
        path = config['checkpoint_id']
    saver_restore.restore(session, path)

    labels = np.array([[], [], []])
    predictions = np.array([[], [], []])
    probabilities = {'a': np.array([]), 'b': np.array([]), 'c': np.array([])}

    if 'b' in config['tasks']:
        probabilities['a'].shape = (0, 2)
        probabilities['b'].shape = (0, 2)
        probabilities['c'].shape = (0, 4)
    else:
        probabilities['a'].shape = (0, 2)
        probabilities['b'].shape = (0, 2)
        probabilities['c'].shape = (0, 3)

    texts = []

    keep_track = int(config['sample_size'] / 2) / 10

    for i in range(int(config['sample_size'] / 2)):
        if i % keep_track == 0:
            print('.', end='')
        batch = session.run(next_element_dev)
        batch = batch + (embed_test,)
        feed_dict = make_test_feed_dict(inferModel, batch)

        num_correct_predictions, alpha, best_example, loss, loss_dict, precision, recall, f1, label, prediction, probability, text = session.run(
            [inferModel.num_correct_predictions,
             inferModel.alpha,
             inferModel.best_example,
             inferModel.loss,
             inferModel.loss_dict,
             inferModel.precision,
             inferModel.recall,
             inferModel.f1, inferModel.labels, inferModel.prediction, inferModel.probabilities,
             inferModel.x_elmo_input], feed_dict=feed_dict)
        labels = np.concatenate([labels, [label['a'], label['b'], label['c']]], axis=1)
        predictions = np.concatenate([predictions, [prediction['a'], prediction['b'], prediction['c']]], axis=1)
        for task in ['a', 'b', 'c']:
            probabilities[task] = np.concatenate([probabilities[task], probability[task]], axis=0)

        texts += text.tolist()

    print("")
    results = {'ids': ids_dev, 'label_a': labels[0], 'label_b': labels[1], 'label_c': labels[2],
               'prediction_a': predictions[0], 'prediction_b': predictions[1], 'prediction_c': predictions[2]}
    for task in ['a', 'b', 'c']:
        probabilities[task] = list(map(lambda x: str(x), probabilities[task]))
    results.update(probabilities)

    results['text'] = texts

    path = str(os.path.join(config['experiment_dir'], 'predictions.csv'))
    print('Writing the predictions to folder ' + path)
    pd.DataFrame(results).to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, help='name of the model')
    parser.add_argument('-model_dir', default='graphs', type=str, help='the directory of the models.')
    parser.add_argument('-checkpoint_id', type=str, default=None, help='checkpoint id (only step number)')
    parser.add_argument('-summary_num', type=int, default=None, help='the fold to recover.')
    parser.add_argument('-data_set', type=str, default=None, help='the data set to evaluate.')
    parser.add_argument('-write_data', action='store_false')
    args = parser.parse_args()

    experiment_dir = os.path.abspath(os.path.join(args.model_dir, args.name, 'summary_' + str(args.summary_num)))
    # Loads config file from experiment folder.
    config = json.load(open(os.path.abspath(os.path.join(args.model_dir, args.name, 'config.json')), 'r'))

    config['experiment_dir'] = experiment_dir
    if args.data_set is not None:
        config['train_data'] = args.data_set
        config['test_data'] = args.data_set
    config['write_data'] = args.write_data
    config['summary_num'] = args.summary_num
    config['checkpoint_id'] = args.checkpoint_id

    if args.checkpoint_id is not None:
        config['checkpoint_id'] = os.path.join(experiment_dir,
                                               'model_' + str(args.summary_num) + '-' + str(args.checkpoint_id))
    else:
        config['checkpoint_id'] = None  # The latest checkpoint will be used.

    main(config)
