### PEP-8 Standard for imports ####
import argparse
import json
import os
import time

import matplotlib
import pandas as pd

matplotlib.use('Agg')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import *
from preprocessing import preprocess
from utils.helper import *
from utils.model_helper import *


def main(config):
    """
    Train the model, write intermediate results to tensorboard and final results to file 'logging.csv'.

    :param config:
    """
    ### Print summary of the network ###
    print("SUMMARY")
    summary_1 = (config['FIRST_LAYER']['layer_1'] + ' + ') if config['FIRST_LAYER']['layer_1_include'] else ''
    print("A summary of the network: " + summary_1 + config['SECOND_LAYER']['layer_2'] + (
        ' with ' if config['SECOND_LAYER']['attention'] else ' without ') + 'attention.')
    print("SUMMARY")
    print("..............................")

    print(np.random.rand(1))

    np.random.seed(123)
    tf.set_random_seed(422)

    ### Create the model directory ###
    os.makedirs(os.path.join(config['model_dir'], config['name']))

    if config['write_data']:
        x_train_org, x_train, x_train_POS, y_train_1, y_train_2, y_train_3, seq_len_train, embed_train, x_dev_org, \
        x_dev, x_dev_POS, y_dev_1, y_dev_2, y_dev_3, seq_len_dev, embed_test, x_attention_org, x_attention, \
        x_attention_POS, y_attention, seq_len_attention, embed_attention, config, length, ids_train, ids_dev = \
            preprocess(config)

        pd.DataFrame({"x_train_org": x_train_org, "x_train": x_train,
                      "x_train_POS": list(map(lambda x: x.tolist(), x_train_POS)), "y_train_1": y_train_1,
                      "y_train_2": y_train_2, "y_train_3": y_train_3, "seq_len_train": seq_len_train,
                      'ids': ids_train}).to_csv('./data/train_csv.csv', index=False)

        pd.DataFrame({"x_dev_org": x_dev_org, "x_dev": x_dev, "x_dev_POS": list(map(lambda x: list(x), x_dev_POS)),
                      "y_dev_1": y_dev_1, "y_dev_2": y_dev_2, "y_dev_3": y_dev_3, "seq_len_dev": seq_len_dev,
                      'ids': ids_dev}).to_csv('./data/dev_csv.csv', index=False)

        pd.DataFrame({"x_attention_org": x_attention_org, "x_attention": x_attention,
                      "x_attention_POS": list(map(lambda x: x.tolist(), x_attention_POS)), "y_attention": y_attention,
                      "seq_len_attention": seq_len_attention}).to_csv('./data/attention_csv.csv', index=False)

        pd.DataFrame({'embed_train': list(map(lambda x: x.tolist(), embed_train))}).to_csv('./data/embed_train_csv.csv')

        pd.DataFrame({'embed_test': list(map(lambda x: x.tolist(), embed_test))}).to_csv('./data/embed_test_csv.csv')

        pd.DataFrame({'embed_attention': list(map(lambda x: x.tolist(), embed_test))}).to_csv(
            './data/embed_attention_csv.csv')
    else:
        train_csv = pd.read_csv('./data/temp/train_csv.csv')
        x_train_org, x_train, x_train_POS, y_train_1, y_train_2, y_train_3, seq_len_train = train_csv[
                                                                                                'x_train_org'].fillna(
            '').values, train_csv['x_train'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.int32).tolist() if x != '[]' else [
                0]).values, train_csv['x_train_POS'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.int32) if x != '[]' else [0]).values, \
                                                                                            train_csv[
                                                                                                'y_train_1'].values, \
                                                                                            train_csv[
                                                                                                'y_train_2'].values, \
                                                                                            train_csv[
                                                                                                'y_train_3'].values, \
                                                                                            train_csv[
                                                                                                'seq_len_train'].values

        dev_csv = pd.read_csv('./data/temp/dev_csv.csv')
        x_dev_org, x_dev, x_dev_POS, y_dev_1, y_dev_2, y_dev_3, seq_len_dev = dev_csv['x_dev_org'].fillna('').values, \
                                                                              dev_csv['x_dev'].map(lambda x: np.array(
                                                                                  x[1:-1].replace('\n', '').split(','),
                                                                                  dtype=np.int32) if x != '[]' else [
                                                                                  0]).values, dev_csv['x_dev_POS'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.int32) if x != '[]' else [0]).values, \
                                                                              dev_csv['y_dev_1'].values, dev_csv[
                                                                                  'y_dev_2'].values, dev_csv[
                                                                                  'y_dev_3'].values, dev_csv[
                                                                                  'seq_len_dev'].values

        attention_csv = pd.read_csv('./data/temp/attention_csv.csv')
        x_attention_org, x_attention, x_attention_POS, y_attention, seq_len_attention = attention_csv[
                                                                                            'x_attention_org'].fillna(
            '').values, attention_csv['x_attention'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.int32) if x != '[]' else [0]).values, \
                                                                                        attention_csv[
                                                                                            'x_attention_POS'].map(
                                                                                            lambda x: np.asarray(
                                                                                                x[1:-1].replace('\n',
                                                                                                                '').split(
                                                                                                    ','),
                                                                                                dtype=np.int32) if x != '[]' else [
                                                                                                0]).values, \
                                                                                        attention_csv[
                                                                                            'y_attention'].map(
                                                                                            lambda x: np.asarray(
                                                                                                x[1:-1].replace('\n',
                                                                                                                '').split(),
                                                                                                dtype=np.float32) if x != '[]' else [
                                                                                                0.0]).values, \
                                                                                        attention_csv[
                                                                                            'seq_len_attention'].values

        embed_train = pd.read_csv('./data/temp/embed_train_csv.csv')['embed_train'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.float32)).values
        n = len(embed_train)
        embed_train = np.reshape(np.concatenate(embed_train), [n, config['fund_embed_dim']])

        embed_test = pd.read_csv('./data/temp/embed_test_csv.csv')['embed_test'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.float32)).values
        n = len(embed_test)
        embed_test = np.reshape(np.concatenate(embed_test), [n, config['fund_embed_dim']])

        embed_attention = pd.read_csv('./data/temp/embed_attention_csv.csv')['embed_attention'].map(
            lambda x: np.asarray(x[1:-1].replace('\n', '').split(','), dtype=np.float32)).values
        n = len(embed_attention)
        embed_attention = np.reshape(np.concatenate(embed_attention), [n, config['fund_embed_dim']])

        config['sample_size'] = len(x_train)
        config['reuse'] = False

        # length = config['sample_size']

    print(np.random.rand(1))

    np.random.seed(123)
    tf.set_random_seed(422)

    _, c = np.unique(y_train_1, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_a'] = tot / np.array(number_classes * c)

    _, c = np.unique(y_train_2, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_b'] = tot / np.array(number_classes * c)

    _, c = np.unique(y_train_3, return_counts=True)
    number_classes = len(c)
    tot = sum(c)
    config['SECOND_LAYER']['weight_c'] = tot / np.array(number_classes * c)

    ##################
    # Training Model
    ##################
    # Create separate graphs for training and validation.
    # Training graph.
    with tf.name_scope("Training"):
        # Create model
        trainModel = Model(config)  # the embed_train is not used here.
        trainModel.build_graph()
        trainModel.build_dense()

        if config['consecutive'] == 'null':
            trainModel.build_loss()
        else:
            holder = config['SECOND_LAYER']['layer_2']
            config['SECOND_LAYER']['layer_2'] = config['consecutive']

            with tf.variable_scope("Parallel"):
                trainModel_2 = Model(config)  # the embed_train is not used here.
                trainModel_2.build_graph()
                trainModel_2.build_dense()

            for task in ['a', 'b', 'c']:
                trainModel.probabilities[task] = tf.reduce_mean(
                    [trainModel.probabilities[task], trainModel_2.probabilities[task]], axis=0)
            trainModel.build_loss()
            config['SECOND_LAYER']['layer_2'] = holder

        if config['SECOND_LAYER']['attention'] and config['SECOND_LAYER']['human_attention']:
            trainModel.build_attention_graph()

        print("\n# of parameters: %s" % trainModel.get_num_parameters())

        ##############
        # Optimization
        ##############
        global_step = tf.Variable(1, name='global_step', trainable=False)
        if config['learning_rate_type'] == 'exponential':
            learning_rate = tf.train.exponential_decay(config['learning_rate'],
                                                       global_step=global_step,
                                                       decay_steps=500,
                                                       decay_rate=config['decay_rate'],
                                                       staircase=False)
        elif config['learning_rate_type'] == 'fixed':
            learning_rate = config['learning_rate']
        else:
            raise Exception("Invalid learning rate type")

        optimizer = tf.train.AdamOptimizer(learning_rate)
        if False:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
        train_op = optimizer.minimize(trainModel.loss, global_step=global_step)
        if config['SECOND_LAYER']['attention'] and config['SECOND_LAYER']['human_attention']:
            train_attention_op = optimizer.minimize(trainModel.loss_attention, global_step=global_step)

    ###################
    # Validation Model
    ###################
    with tf.name_scope("Validation"):
        # Create model
        config['reuse'] = True
        validModel = Model(config)  # The embed_test is not used here
        validModel.build_graph()
        validModel.build_dense()

        if config['consecutive'] == 'null':
            validModel.build_loss()
        else:
            holder = config['SECOND_LAYER']['layer_2']
            config['SECOND_LAYER']['layer_2'] = config['consecutive']

            with tf.variable_scope("Parallel"):
                validModel_2 = Model(config)  # the embed_train is not used here.
                validModel_2.build_graph()
                validModel_2.build_dense()

            for task in ['a', 'b', 'c']:
                validModel.probabilities[task] = tf.reduce_mean(
                    [validModel.probabilities[task], validModel_2.probabilities[task]], axis=0)
            validModel.build_loss()
            config['SECOND_LAYER']['layer_2'] = holder

    ##############
    # Monitoring
    ##############   

    loss_avg_pl = tf.placeholder(tf.float32, name="loss_avg_pl")

    accuracy_avg_pl = {}
    loss_dict_avg_pl = {}
    precision_avg_pl = {}
    recall_avg_pl = {}
    f1_avg_pl = {}

    # Attention words placeholder
    text_pl = tf.placeholder(tf.string, name="best_words")
    # Image
    image_pl = tf.placeholder(tf.float32, name="text_image")

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and plots evaluation results.
    summary_train_loss = tf.summary.scalar('loss', trainModel.loss)
    summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg_pl)

    summary_train_acc = []
    summary_train_precision = []
    summary_train_recall = []

    # summary_train_f1 = tf.summary.scalar('f1_training', trainModel.f1)

    summary_avg_accuracy = []
    summary_avg_loss_dict = []
    summary_avg_precision = []
    summary_avg_recall = []
    summary_avg_f1 = []

    for task in ['a', 'b', 'c']:
        # Create placeholders to provide tensorflow average loss and accuracy.
        accuracy_avg_pl[task] = tf.placeholder(tf.float32, name="accuracy_avg_pl_" + task)
        loss_dict_avg_pl[task] = tf.placeholder(tf.float32, name="loss_dict_avg_pl_" + task)
        precision_avg_pl[task] = tf.placeholder(tf.float32, name="precision_avg_pl_" + task)
        recall_avg_pl[task] = tf.placeholder(tf.float32, name="recall_avg_pl_" + task)
        f1_avg_pl[task] = tf.placeholder(tf.float32, name="f1_avg_pl_" + task)

        summary_train_acc += [tf.summary.scalar('accuracy_training_' + task, trainModel.batch_accuracy[task])]
        summary_train_precision += [tf.summary.scalar('precision_training_' + task, trainModel.precision[task])]
        summary_train_recall += [tf.summary.scalar('recall_training_' + task, trainModel.recall[task])]

        # summary_train_f1 = tf.summary.scalar('f1_training', trainModel.f1)
        summary_avg_accuracy += [tf.summary.scalar('accuracy_avg_' + task, accuracy_avg_pl[task])]
        summary_avg_loss_dict += [tf.summary.scalar('loss_dict_avg_' + task, loss_dict_avg_pl[task])]
        summary_avg_precision += [tf.summary.scalar('precision_avg_' + task, precision_avg_pl[task])]
        summary_avg_recall += [tf.summary.scalar('recall_avg_' + task, recall_avg_pl[task])]
        summary_avg_f1 += [tf.summary.scalar('f1_avg_' + task, f1_avg_pl[task])]

    # Attention words summary
    summary_text = tf.summary.text('list', text_pl)
    # Image
    summary_image = tf.summary.image('image', image_pl)
    summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)

    # Group summaries. summaries_training is used during training and reported after every step.
    summaries_training = tf.summary.merge(np.concatenate(
        [[summary_train_loss], summary_train_acc, summary_train_precision, summary_train_recall,
         [summary_learning_rate]], axis=0).tolist())

    # summaries_evaluation is used by both training and validation in order to report the performance on the dataset.
    summaries_evaluation = tf.summary.merge(np.concatenate(
        [summary_avg_accuracy, [summary_avg_loss], summary_avg_loss_dict, summary_avg_precision, summary_avg_recall,
         summary_avg_f1, [summary_text]], 0).tolist())

    # Create session object
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually initialize the variables
    session.run(init_op)

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = [0.0, 0.0, 0.0]
    counter_loss_training = 0.0
    counter_loss_dict_training = [0.0, 0.0, 0.0]
    counter_precision_training = [0.0, 0.0, 0.0]
    counter_recall_training = [0.0, 0.0, 0.0]
    counter_f1_training = [0.0, 0.0, 0.0]
    best_example_agr_training = []

    counter_correct_predictions_validation = [0.0, 0.0, 0.0]
    counter_loss_validation = 0.0
    counter_loss_dict_validation = [0.0, 0.0, 0.0]
    counter_precision_validation = [0.0, 0.0, 0.0]
    counter_recall_validation = [0.0, 0.0, 0.0]
    counter_f1_validation = [0.0, 0.0, 0.0]
    best_example_agr_validation = []

    del config['SECOND_LAYER']['weight_a']
    del config['SECOND_LAYER']['weight_b']
    del config['SECOND_LAYER']['weight_c']

    # Save configuration in json formats.
    json.dump(config, open(os.path.join(config['model_dir'], config['name'], 'config.json'), 'w'), indent=4,
              sort_keys=True)

    ### tf.data sources ###

    ## Dataset Attention
    x_attention_pad = pad_sequences(x_attention, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
    y_attention_pad = pad_sequences(y_attention, maxlen=None, dtype='float32', padding='pre', truncating='pre',
                                    value=0.0)
    x_attention_POS_pad = pad_sequences(x_attention_POS, maxlen=None, dtype='int32', padding='pre', truncating='pre',
                                        value=0.0)
    dataset_attention = tf.data.Dataset.from_tensor_slices(
        (x_attention_org, x_attention_pad, x_attention_POS_pad, y_attention_pad, seq_len_attention))
    dataset_attention = dataset_attention.repeat()  # Repeat the input indefinitely.
    dataset_attention = dataset_attention.shuffle(32, seed=343)
    dataset_attention = dataset_attention.batch(config["batch_size"])

    iterator_attention = dataset_attention.make_one_shot_iterator()
    next_element_attention = iterator_attention.get_next()

    length = config['sample_size']
    n_fold = config['fold']

    assert n_fold > 1

    skip = int(length / n_fold)
    if args.skip is not None:
        skip = config['skip']
    x_train_org_tmp, x_train_tmp, x_train_POS_tmp, y_train_1_tmp, y_train_2_tmp, y_train_3_tmp, seq_len_train_tmp = \
        x_train_org, x_train, x_train_POS, y_train_1, y_train_2, y_train_3, seq_len_train

    embed_test_tmp = embed_train

    print(np.random.rand(1))
    print(session.run(tf.random_uniform([1])))

    np.random.seed(84034)
    tf.set_random_seed(892)

    best_accuracy_scores = []
    best_loss_scores = []
    best_loss_dict_scores = []
    best_precision_scores = []
    best_recall_scores = []
    best_f1_scores = []
    best_f1_indices = []

    print('Starting the cross validation loop')
    ### Cross Validation Loop ###
    for i in range(n_fold):
        print(np.random.rand(1))
        print(session.run(tf.random_uniform([1])))

        np.random.seed(123)
        tf.set_random_seed(422)

        saver = tf.train.Saver(max_to_keep=100000, save_relative_paths=True)

        session.run(init_op)

        ### Variable for early stopping ###
        early_stopping_counter = 0
        loss_best = 1000.0  ### Large initial loss ###
        accuracy_best = [0.0, 0.0, 0.0]
        loss_dict_best = [1000.0, 1000.0, 1000.0]
        precision_best = [0.0, 0.0, 0.0]
        recall_best = [0.0, 0.0, 0.0]
        f1_best = [0.0, 0.0, 0.0]
        f1_best_index = [0, 0, 0]

        x_train_org = np.delete(x_train_org_tmp, range(i * skip, skip * (i + 1)))
        x_train = np.delete(x_train_tmp, range(i * skip, skip * (i + 1)))
        x_train_POS = np.delete(x_train_POS_tmp, range(i * skip, skip * (i + 1)))

        x_train_POS = x_train_POS_tmp.copy().tolist()
        for _ in range(skip):
            x_train_POS.pop(i * skip)

        y_train_1 = np.delete(y_train_1_tmp, range(i * skip, skip * (i + 1)))
        y_train_2 = np.delete(y_train_2_tmp, range(i * skip, skip * (i + 1)))
        y_train_3 = np.delete(y_train_3_tmp, range(i * skip, skip * (i + 1)))
        seq_len_train = np.delete(seq_len_train_tmp, range(i * skip, skip * (i + 1)))

        ### During CROSS Validation ###

        if config['cross_validation']:
            x_dev_org = x_train_org_tmp[(i * skip):(skip * (i + 1))]
            x_dev = x_train_tmp[(i * skip):(skip * (i + 1))]
            x_dev_POS = x_train_POS_tmp[(i * skip): (skip * (i + 1))]
            y_dev_1 = y_train_1_tmp[(i * skip): (skip * (i + 1))]
            y_dev_2 = y_train_2_tmp[(i * skip): (skip * (i + 1))]
            y_dev_3 = y_train_3_tmp[(i * skip): (skip * (i + 1))]
            seq_len_dev = seq_len_train_tmp[(i * skip): (skip * (i + 1))]

            embed_test = embed_test_tmp

        x_train_pad = pad_sequences(x_train, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
        x_train_POS_pad = pad_sequences(x_train_POS, maxlen=None, dtype='int32', padding='pre', truncating='pre',
                                        value=0.0)
        dataset_train = tf.data.Dataset.from_tensor_slices(
            (x_train_org, x_train_pad, x_train_POS_pad, y_train_1, y_train_2, y_train_3, seq_len_train))
        dataset_train = dataset_train.repeat()  # Repeat the input indefinitely.
        dataset_train = dataset_train.shuffle(32, seed=343)
        dataset_train = dataset_train.batch(config["batch_size"])

        iterator_train = dataset_train.make_one_shot_iterator()
        next_element_train = iterator_train.get_next()

        ### Dataset Dev ###
        x_dev_pad = pad_sequences(x_dev, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
        x_dev_POS_pad = pad_sequences(x_dev_POS, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
        dataset_dev = tf.data.Dataset.from_tensor_slices(
            (x_dev_org, x_dev_pad, x_dev_POS_pad, y_dev_1, y_dev_2, y_dev_3, seq_len_dev))
        dataset_dev = dataset_dev.repeat()  # Repeat the input indefinitely.
        # dataset_dev = dataset_dev.shuffle(32, seed = 343)
        dataset_dev = dataset_dev.batch(config["batch_size"])

        iterator_dev = dataset_dev.make_one_shot_iterator()
        next_element_dev = iterator_dev.get_next()

        # Register summary ops.
        train_summary_dir = os.path.join(config['model_dir'], config['name'], "summary_" + str(i), "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

        valid_summary_dir = os.path.join(config['model_dir'], config['name'], "summary_" + str(i), "valid")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)

        print('Print the update ops collection')
        print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        ##########################
        # Training Loop
        ##########################
        step = 0
        print("Running for " + str(
            config['train_epoch'] * (int(config['sample_size'] / config['batch_size']))) + " batches per fold")

        try:
            while step < config['train_epoch'] * (int(config['sample_size'] / config['batch_size'])) \
                    and early_stopping_counter < config['early_stopping_threshold']:
                step = tf.train.global_step(session, global_step)

                start_time = time.perf_counter()
                batch = session.run(next_element_train)
                batch = batch + (embed_train,)
                feed_dict = make_train_feed_dict(trainModel, batch)
                if config['consecutive'] != 'null':
                    feed_dict.update(make_train_feed_dict(trainModel_2, batch))

                train_summary, alpha, best_example, num_correct_predictions, loss_dict, precision, recall, f1, loss, _ = session.run(
                    [summaries_training, trainModel.alpha, trainModel.best_example, trainModel.num_correct_predictions,
                     trainModel.loss_dict, trainModel.precision, trainModel.recall, trainModel.f1, trainModel.loss,
                     train_op], feed_dict=feed_dict)

                best_example = np.array([str(x).encode('utf-8') for x in alpha])
                best_example = np.array([str(x[0]).encode('utf-8') + ' | '.encode('utf-8') + x[1] for x in
                                         np.transpose([best_example, batch[0]], [1, 0])])

                # Update counters.
                counter_correct_predictions_training += np.array(
                    [num_correct_predictions['a'], num_correct_predictions['b'], num_correct_predictions['c']])
                counter_loss_training += loss
                counter_loss_dict_training += np.array([loss_dict['a'], loss_dict['b'], loss_dict['c']])
                counter_precision_training += np.array([precision['a'], precision['b'], precision['c']])
                counter_recall_training += np.array([recall['a'], recall['b'], recall['c']])
                counter_f1_training += np.array([f1['a'], f1['b'], f1['c']])
                best_example_agr_training += [best_example]
                train_summary_writer.add_summary(train_summary, step)

                if step % 400 == 0:
                    ### Saving the Attention Table ###
                    image = create_attention_table(alpha, batch=batch)
                    summary_image = tf.summary.image("plot", image)
                    summary = session.run(summary_image)
                    train_summary_writer.add_summary(summary, step)

                # Report training performance
                if (step % config['print_every_step']) == 0:
                    # To get a smoother loss plot, we calculate average performance.
                    accuracy_avg = counter_correct_predictions_training / (
                            config['batch_size'] * config['print_every_step'])
                    loss_avg = counter_loss_training / (config['print_every_step'])
                    loss_dict_avg = counter_loss_dict_training / (config['print_every_step'])
                    precision_avg = counter_precision_training / (config['print_every_step'])
                    recall_avg = counter_recall_training / (config['print_every_step'])
                    f1_avg = counter_f1_training / (config['print_every_step'])

                    # Feed average performance.

                    feed_dict = {accuracy_avg_pl['a']: accuracy_avg[0], loss_avg_pl: loss_avg,
                                 loss_dict_avg_pl['a']: loss_dict_avg[0], precision_avg_pl['a']: precision_avg[0],
                                 recall_avg_pl['a']: recall_avg[0], f1_avg_pl['a']: f1_avg[0],
                                 text_pl: best_example_agr_training}

                    feed_dict.update({accuracy_avg_pl['b']: accuracy_avg[1], loss_dict_avg_pl['b']: loss_dict_avg[1],
                                      precision_avg_pl['b']: precision_avg[1], recall_avg_pl['b']: recall_avg[1],
                                      f1_avg_pl['b']: f1_avg[1]})

                    feed_dict.update({accuracy_avg_pl['c']: accuracy_avg[2], loss_dict_avg_pl['c']: loss_dict_avg[2],
                                      precision_avg_pl['c']: precision_avg[2], recall_avg_pl['c']: recall_avg[2],
                                      f1_avg_pl['c']: f1_avg[2]})

                    best_example_agr_training = np.unique(best_example_agr_training)
                    summary_report = session.run(summaries_evaluation,
                                                 feed_dict=feed_dict)
                    train_summary_writer.add_summary(summary_report, step)
                    time_elapsed = (time.perf_counter() - start_time) / config['print_every_step']
                    print(
                        "[Train/%d] Accuracy: %.3f, Loss: %.3f, Precision: %.3f, Recall: %.3f, f1-score: %.3f, "
                        "time/step = %.3f" % (
                            step,
                            accuracy_avg[0],
                            loss_avg,
                            precision_avg[0],
                            recall_avg[0],
                            f1_avg[0],
                            time_elapsed))
                    counter_correct_predictions_training = np.array([0.0, 0.0, 0.0])
                    counter_loss_training = 0.0
                    counter_loss_dict_training = np.array([0.0, 0.0, 0.0])
                    counter_precision_training = np.array([0.0, 0.0, 0.0])
                    counter_recall_training = np.array([0.0, 0.0, 0.0])
                    counter_f1_training = np.array([0.0, 0.0, 0.0])
                    best_example_agr_training = []

                # Report validation performance
                if (step % config['evaluate_every_step']) == 0:
                    if 0 == step % 400:
                        ckpt_save_path = saver.save(session, os.path.join(config['model_dir'], config['name'],
                                                                          "summary_" + str(i), 'model_' + str(i)),
                                                    global_step)
                        print("Model saved in file: %s" % ckpt_save_path)
                    evaluation_steps = int(skip / config['batch_size'])

                    print('The number of evaluation steps')
                    print(evaluation_steps)
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

                    start_time = time.perf_counter()

                    for eval_step in range(evaluation_steps):
                        # Calculate average validation accuracy.
                        batch = session.run(next_element_dev)

                        batch = batch + (embed_test,)
                        feed_dict = make_test_feed_dict(validModel, batch)
                        if not config['consecutive'] == 'null':
                            feed_dict.update(make_test_feed_dict(validModel_2, batch))

                        num_correct_predictions, alpha, best_example, loss, loss_dict, precision, recall, f1, label, prediction, probability, text = session.run(
                            [validModel.num_correct_predictions,
                             validModel.alpha,
                             validModel.best_example,
                             validModel.loss,
                             validModel.loss_dict,
                             validModel.precision,
                             validModel.recall,
                             validModel.f1, validModel.labels, validModel.prediction, validModel.probabilities,
                             validModel.x_elmo_input], feed_dict=feed_dict)

                        # Update counters.
                        counter_correct_predictions_validation += np.array(
                            [num_correct_predictions['a'], num_correct_predictions['b'], num_correct_predictions['c']])

                        counter_loss_validation += loss
                        counter_loss_dict_validation += np.array([loss_dict['a'], loss_dict['b'], loss_dict['c']])

                        counter_precision_validation += np.array([precision['a'], precision['b'], precision['c']])

                        counter_recall_validation += np.array([recall['a'], recall['b'], recall['c']])

                        counter_f1_validation += np.array([f1['a'], f1['b'], f1['c']])

                        if (step % 400) == 0:
                            labels = np.concatenate([labels, [label['a'], label['b'], label['c']]], axis=1)
                            predictions = np.concatenate(
                                [predictions, [prediction['a'], prediction['b'], prediction['c']]], axis=1)

                            for task in ['a', 'b', 'c']:
                                probabilities[task] = np.concatenate([probabilities[task], probability[task]], axis=0)

                            texts += text.tolist()

                        if step % 400 == 0 and step != 0:
                            ### Saving the Text ###
                            best_example = np.array([str(x).encode('utf-8') for x in alpha])

                            best_example = np.array([str(x[0]).encode('utf-8') + ' | '.encode('utf-8') + x[1] for x in
                                                     np.transpose([best_example, batch[0]], [1, 0])])

                            best_example_agr_validation += [best_example]

                            ### Saving the Attention Table ###
                            image = create_attention_table(alpha, batch=batch)
                            summary_image = tf.summary.image("plot", image)
                            summary = session.run(summary_image)
                            valid_summary_writer.add_summary(summary, step)

                    if (step % 400) == 0:
                        results = {'label_a': labels[0], 'label_b': labels[1], 'label_c': labels[2],
                                   'prediction_a': predictions[0], 'prediction_b': predictions[1],
                                   'prediction_c': predictions[2], 'text': texts}

                        for task in ['a', 'b', 'c']:
                            probabilities[task] = list(map(lambda x: str(x), probabilities[task]))
                        results.update(probabilities)

                        pd.DataFrame(results).to_csv(
                            os.path.join('graphs', config['name'], 'summary_' + str(i), 'predictions.csv'), index=False)

                    if config['SECOND_LAYER']['attention'] and config['SECOND_LAYER']['human_attention']:
                        for eval_step in range(config['evaluate_every_step']):
                            ### Attention Training ###
                            batch_attention = session.run(next_element_attention)
                            batch_attention = batch_attention + (embed_attention,)
                            feed_dict_attention = make_attention_feed_dict(trainModel, batch_attention)
                            session.run([train_attention_op], feed_dict=feed_dict_attention)

                    # Report validation performance
                    accuracy_avg = counter_correct_predictions_validation / (config['batch_size'] * evaluation_steps)

                    loss_avg = counter_loss_validation / evaluation_steps

                    loss_dict_avg = counter_loss_dict_validation / evaluation_steps

                    precision_avg = counter_precision_validation / evaluation_steps
                    recall_avg = counter_recall_validation / evaluation_steps
                    f1_avg = counter_f1_validation / evaluation_steps

                    ### Early Stopping and Best Result Tracking ###
                    if loss_avg > loss_best:
                        early_stopping_counter += config['evaluate_every_step']
                    else:
                        loss_best = loss_avg
                        early_stopping_counter = 0

                    for t in range(3):
                        if accuracy_avg[t] > accuracy_best[t]:
                            accuracy_best[t] = accuracy_avg[t]

                        if precision_avg[t] > precision_best[t]:
                            precision_best[t] = precision_avg[t]

                        if recall_avg[t] > recall_best[t]:
                            recall_best[t] = recall_avg[t]

                        if f1_avg[t] > f1_best[t]:
                            f1_best[t] = f1_avg[t]
                            f1_best_index[t] = step

                        if loss_dict_avg[t] > loss_dict_best[t]:
                            loss_dict_best[t] = loss_dict_avg[t]

                    # This was here to account for validating twice or more on the same dataset
                    # best_example_agr_validation = np.unique(best_example_agr_validation)
                    feed_dict = {accuracy_avg_pl['a']: accuracy_avg[0], loss_avg_pl: loss_avg,
                                 loss_dict_avg_pl['a']: loss_dict_avg[0], precision_avg_pl['a']: precision_avg[0],
                                 recall_avg_pl['a']: recall_avg[0], f1_avg_pl['a']: f1_avg[0],
                                 text_pl: best_example_agr_validation}

                    feed_dict.update({accuracy_avg_pl['b']: accuracy_avg[1], loss_dict_avg_pl['b']: loss_dict_avg[1],
                                      precision_avg_pl['b']: precision_avg[1], recall_avg_pl['b']: recall_avg[1],
                                      f1_avg_pl['b']: f1_avg[1]})

                    feed_dict.update({accuracy_avg_pl['c']: accuracy_avg[2], loss_dict_avg_pl['c']: loss_dict_avg[2],
                                      precision_avg_pl['c']: precision_avg[2], recall_avg_pl['c']: recall_avg[2],
                                      f1_avg_pl['c']: f1_avg[2]})

                    summary_report = session.run(summaries_evaluation,
                                                 feed_dict=feed_dict)
                    valid_summary_writer.add_summary(summary_report, step)
                    time_elapsed = (time.perf_counter() - start_time) / config['num_validation_steps']
                    print("")
                    print(
                        "[Valid/%d] Accuracy: %.3f, Loss: %.3f, Precision: %.3f, Recall: %.3f, f1-score: %.3f, "
                        "time/step = %.3f" % (
                            step,
                            accuracy_avg[0],
                            loss_avg,
                            precision_avg[0],
                            recall_avg[0],
                            f1_avg[0],
                            time_elapsed))
                    print("")
                    print(
                        "[Valid/%d] Accuracy: %.3f, Loss: %.3f, Precision: %.3f, Recall: %.3f, f1-score: %.3f, "
                        "time/step = %.3f" % (
                            step,
                            accuracy_avg[1],
                            loss_avg,
                            precision_avg[1],
                            recall_avg[1],
                            f1_avg[1],
                            time_elapsed))
                    print("")
                    print(
                        "[Valid/%d] Accuracy: %.3f, Loss: %.3f, Precision: %.3f, Recall: %.3f, f1-score: %.3f, "
                        "time/step = %.3f" % (
                            step,
                            accuracy_avg[2],
                            loss_avg,
                            precision_avg[2],
                            recall_avg[2],
                            f1_avg[2],
                            time_elapsed))
                    print("")
                    print(
                        "[Best_Valid/%d] Accuracy: %.3f, Loss: %.3f, Precision: %.3f, Recall: %.3f, f1-score: %.3f" % (
                            step,
                            accuracy_best[2],
                            loss_best,
                            precision_best[2],
                            recall_best[2],
                            f1_best[2]))
                    print("")

                    counter_correct_predictions_validation = np.array([0.0, 0.0, 0.0])
                    counter_loss_validation = 0.0
                    counter_loss_dict_validation = np.array([0.0, 0.0, 0.0])
                    counter_precision_validation = np.array([0.0, 0.0, 0.0])
                    counter_recall_validation = np.array([0.0, 0.0, 0.0])
                    counter_f1_validation = np.array([0.0, 0.0, 0.0])
                    best_example_agr_validation = []

        except tf.errors.OutOfRangeError:
            print('Model is trained for %d epochs, %d steps.' % (config['num_epochs'], step))
            print('Done.')

        if early_stopping_counter >= config['early_stopping_threshold']:
            print('Terminated due to early stopping.')

        feed_dict = {accuracy_avg_pl['a']: accuracy_avg[0], loss_avg_pl: loss_avg,
                     loss_dict_avg_pl['a']: loss_dict_avg[0], precision_avg_pl['a']: precision_avg[0],
                     recall_avg_pl['a']: recall_avg[0], f1_avg_pl['a']: f1_avg[0], text_pl: best_example_agr_training}

        feed_dict.update({accuracy_avg_pl['b']: accuracy_avg[1], loss_dict_avg_pl['b']: loss_dict_avg[1],
                          precision_avg_pl['b']: precision_avg[1], recall_avg_pl['b']: recall_avg[1],
                          f1_avg_pl['b']: f1_avg[1]})

        feed_dict.update({accuracy_avg_pl['c']: accuracy_avg[2], loss_dict_avg_pl['c']: loss_dict_avg[2],
                          precision_avg_pl['c']: precision_avg[2], recall_avg_pl['c']: recall_avg[2],
                          f1_avg_pl['c']: f1_avg[2]})

        summary_report = session.run(summaries_evaluation,
                                     feed_dict=feed_dict)
        valid_summary_writer.add_summary(summary_report, step)

        print(np.random.rand(1))
        print(session.run(tf.random_uniform([1])))

        np.random.seed(123)
        tf.set_random_seed(422)
        session.run(init_op)

        best_accuracy_scores.append(accuracy_best)
        best_loss_scores.append(loss_best)
        best_precision_scores.append(precision_best)
        best_recall_scores.append(recall_best)
        best_f1_scores.append(f1_best)
        best_f1_indices.append(f1_best_index)

        # Evaluate model after training and create submission file.
        config['summary_num'] = i

    log_dict = dict(model_name=[config['name']], f1_scores=[best_f1_scores], best_f1_indices=[best_f1_indices],
                    accuracies=[best_accuracy_scores], precisions=[best_precision_scores], recalls=[best_recall_scores],
                    losses=[best_loss_scores], f1=[list(map(lambda x: np.mean(x), np.array(best_f1_scores).T))],
                    accuracy=[list(map(lambda x: np.mean(x), np.array(best_accuracy_scores).T))],
                    precision=[list(map(lambda x: np.mean(x), np.array(best_precision_scores).T))],
                    recall=[list(map(lambda x: np.mean(x), np.array(best_recall_scores).T))],
                    loss=[list(map(lambda x: np.mean(x), np.array(best_loss_scores).T))])

    ### Logging the cross validation performance of the F1 score ###
    log = pd.DataFrame(log_dict)
    if config['log_it']:
        pd.read_csv('logging.csv').append(log).to_csv('logging.csv', index=False)

    print('From all the runs were ' + str(best_f1_scores))
    print("")
    print('The Cross Validation Score is ' + str(np.mean(best_f1_scores)))


if __name__ == '__main__':
    # Process FLAGS
    parser = argparse.ArgumentParser(description='Process the FLAGS.')
    parser.add_argument('-peephole_1', action='store_true')  ### Peephole forward pass first RNN  ###
    parser.add_argument('-peephole_2', action='store_true')  ### Peephole backward pass first RNN ###
    parser.add_argument('-peephole_3', action='store_true')  ### Peephole second forward RNN      ###
    parser.add_argument('-peephole_4', action='store_true')  ### Peephole second backward RNN     ###
    parser.add_argument('-mod', type=int, default=100)
    parser.add_argument('-train', action='store_false')  ### Deactivate training ###
    parser.add_argument('-test_ratio', type=float,
                        default=1.0)  ### Take a ratio of test set, only with write_data TRUE ###
    parser.add_argument('-train_ratio', type=float,
                        default=1.0)  ### Take a ratio of train set, only with write_data TRUE ###
    parser.add_argument('-clean', action='store_false')  ### Clean the data ###
    parser.add_argument('-elmo', action='store_true')  ### Include ELMo Embedding ###
    parser.add_argument('-glove', action='store_false')  ### Include glove embedding ###
    parser.add_argument('-model_dir', type=str, default='graphs')  ### Model Directory ###
    parser.add_argument('-name', type=str, default='default_name')  ### Graph name ###
    parser.add_argument('-attention', action='store_false')  ### Include attention ###
    parser.add_argument('-attention_vectors', type=int, default=20)  ### REDUNDENT ###
    parser.add_argument('-mask', action='store_false')  ### Mask the sentences ###
    parser.add_argument('-fold', type=int, default=5)  ### Number of folds in cross validation ###
    parser.add_argument('-layer_1', type=str, default='conv')  ### Layer 1 types: conv, gru, lstm ###
    parser.add_argument('-layer_2', type=str, default='gru')  ### Layer 1 types: conv, gru, lstm ###
    parser.add_argument('-cross', action='store_false')  ### Activate Cross Validation ###
    parser.add_argument('-train_data', type=str,
                        default='multi_classes')  ### Select train data, only with write_data TRUEE ###
    parser.add_argument('-test_data', type=str,
                        default='multi_classes')  ### Select test data, only with write_data TRUE ###
    parser.add_argument('-layer_1_include', action='store_true')  ### Exclude layer 1 ###
    parser.add_argument('-pool_mean', action='store_false')  ### Mean pool the attention, otherwise max-pool ###
    parser.add_argument('-num_attention', type=int, default=100)  ### Number of attention vectors ###
    parser.add_argument('-attention_prob', type=float, default=0.2)  ### Dropout probability ###
    parser.add_argument('-pos_include', action='store_true')  ###  REDUNDANT ###
    parser.add_argument('-kernel_size', type=int, default=4)  ### Kernel size for convolutional layers ###
    parser.add_argument('-augment',
                        action='store_true')  ### Surround entities with corresponding entity type ### ### REDUNDANT ###
    parser.add_argument('-replace',
                        action='store_true')  ### Replace entities with corresponding entity type ### ### REDUNDANT ###
    parser.add_argument('-formal_filter', action='store_false')  ### Apply a formal filter, e.g. 'I'm' -> "I am" ###
    parser.add_argument('-pos_dimensions', type=int, default=100)  ### Dimension of POS embedding ###
    parser.add_argument('-embed_dimensions', type=int,
                        default=128)  ### Embedding dimension of tweets for self-learned embeddings###
    parser.add_argument('-human_attention',
                        action='store_true')  ### Add a human attention mechanism to training, refer to PAPER ###
    parser.add_argument('-n_class', type=int, default=2)  ### REDUNDANT ###
    parser.add_argument('-early_stopping_threshold', type=int, default=500)  ### Early stopping threshold ###
    parser.add_argument('-write_data', action='store_false')  ### Preprocess the data and write to ./data/ ###
    parser.add_argument('-remove_stop',
                        action='store_false')  ### Remove stopwords in preprocessing, only with write_data TRUE ###
    parser.add_argument('-pos_weight', type=float, default=1.0)  ### REDUNDANT ###
    parser.add_argument('-tasks', type=str, default=['a'], nargs='+',
                        choices=['a', 'b', 'c'])  ### Train Task a), b) or c) ###
    parser.add_argument('-first_cause', action='store_true')  ### REDUNDANT ###
    parser.add_argument('-second_cause', action='store_true')  ### REDUNDANT ###
    parser.add_argument('-hidden_size', type=int,
                        default=512)  ### The hidden size of the RNNs and half the size of the convolutional layer ###
    parser.add_argument('-consecutive', type=str,
                        default='null')  # ## Training a parallel network depending on the value: conv, lstm and gru,
    # and null for no parallel network ###
    parser.add_argument('-learning_rate', type=float, default=8e-3)  ### learning rate ###
    parser.add_argument('-decay_rate', type=float, default=0.97)  ### decay rate ###
    parser.add_argument('-skip', type=int,
                        default=None)  # ## In cross validation use (skip) size as validation set, only relevant when
    # using external data for training ###
    parser.add_argument('-batch_size', type=int, default=32)  ### batch size ###
    parser.add_argument('-fund_embed_dim', type=int,
                        default=200)  ### The embedding dimension used from Glove, options: 50 and 200 ###

    args = parser.parse_args()

    if args.name == 'default_name':
        args.log_it = False
    else:
        args.log_it = True

    timestamp = str(int(time.time()))
    args.name = timestamp + '_' + args.name

    ### Glove goes through the first layer. Therefore, no glove implies no layer_1. ###
    if not args.glove:
        args.layer_1_include = False

    ### Check above assertion ###
    assert args.glove or not args.layer_1_include

    from config import update_config

    config = update_config(args)

    main(config)
