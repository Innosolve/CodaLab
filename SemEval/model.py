## PEP - 8 Import Standard ###

import random

import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.contrib.rnn import GRUCell
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


class Model(object):
    def __init__(self, config):
        # Configurations
        tf.set_random_seed(324)
        self.embedding_matrix = None
        self.best_example = None
        self.num_parameters = 0
        self.loss = tf.constant(0.0)

        self.n_class = config["n_class"]
        self.reuse = config['reuse']
        self.is_training = not config['reuse']
        self.batch_size = config["batch_size"]
        self.hidden_size = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.tasks = config['tasks']
        if 'b' in self.tasks:
            self.classes = {'a': 2, 'b': 2, 'c': 4}
        else:
            self.classes = {'a': 2, 'b': 2, 'c': 3}

        self.layer_1 = config['FIRST_LAYER']['layer_1']
        self.glove_include = config['FIRST_LAYER']['glove']
        self.layer_1_include = config['FIRST_LAYER']['layer_1_include']
        self.peephole_1 = config['FIRST_LAYER']['peephole_1']
        self.peephole_2 = config['FIRST_LAYER']['peephole_2']
        self.pos_include = config['FIRST_LAYER']['pos_include']
        self.kernel_size = config['FIRST_LAYER']['kernel_size']
        self.pos_dimensions = config['FIRST_LAYER']['pos_dimensions']
        self.embed_dimensions = config['FIRST_LAYER']['embed_dimensions']

        self.attention = config['SECOND_LAYER']['attention']
        self.attention_vectors = config['SECOND_LAYER']['attention_vectors']
        self.mask = config['SECOND_LAYER']['mask']
        self.elmo = config['SECOND_LAYER']['elmo']
        self.peephole_3 = config['SECOND_LAYER']['peephole_3']
        self.peephole_4 = config['SECOND_LAYER']['peephole_4']
        self.pool_mean = config['SECOND_LAYER']['pool_mean']
        self.num_attention = config['SECOND_LAYER']['num_attention']
        self.attention_prob = config['SECOND_LAYER']['attention_prob']
        self.layer_2 = config['SECOND_LAYER']['layer_2']
        self.pos_weight = config['SECOND_LAYER']['pos_weight']
        self.weight_a = config['SECOND_LAYER']['weight_a']
        self.weight_b = config['SECOND_LAYER']['weight_b']
        self.weight_c = config['SECOND_LAYER']['weight_c']
        self.first_cause = config['SECOND_LAYER']['first_cause']
        self.second_cause = config['SECOND_LAYER']['second_cause']

        self.weights = [tf.constant(self.weight_a), tf.constant(self.weight_b), tf.constant(self.weight_c)]

        # Placeholders
        self.x_elmo_input = tf.placeholder(tf.string, [None], name='elmo_input')
        self.x = tf.placeholder(tf.int32, [None, None], name='x')
        self.pos = tf.placeholder(tf.int32, [None, None], name='pos')
        self.labels = {}
        for task in ['a', 'b', 'c']:
            self.labels[task] = tf.placeholder(tf.int32, [None], name='label_' + task)
        self.attention_label = tf.placeholder(tf.float32, [None, None], name='attention')
        self.glove = tf.placeholder(tf.float32, [None, config['fund_embed_dim']], name='glove')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.seq_len = tf.placeholder(tf.int32, name='seq_len')

    def build_attention_graph(self):
        """ Create the loss for human attention training."""
        with tf.name_scope('loss_attention'):
            self.loss_attention = tf.reduce_mean(
                tf.losses.mean_squared_error(predictions=self.alpha, labels=self.attention_label[:, -self.shape[1]:]))

    def build_graph(self):
        """ Build the main architecture of the graph. """
        random.seed(310)
        tf.set_random_seed(902)
        print("building graph")

        with tf.variable_scope('model', reuse=self.reuse):
            ### Lookup ELMo Embedding ###
            self.x_elmo = layers.Lambda(lambda inputs: ElmoEmbedding(inputs, elmo_model), output_shape=(1024,))(
                self.x_elmo_input)

            shape = tf.shape(self.x_elmo)
            self.shape = shape
            #            self.glove = tf.Variable(tf.random_uniform([tf.shape(self.glove)[0], self.embed_dimensions], -1.0, 1.0),trainable=True)

            if self.glove_include:
                ### Lookup Glove Vectors ###
                batch_embedded = tf.nn.embedding_lookup(self.glove, self.x)
                batch_embedded = batch_embedded[:, -shape[1]:, :]

                ### Include POS ###
                if self.pos_include:
                    ### POS-TAG Embedding ###
                    embeddings_var = tf.Variable(tf.random_uniform([12, self.pos_dimensions], -1.0, 1.0),
                                                 trainable=True)
                    self.pos_embedding = tf.nn.embedding_lookup(embeddings_var, self.pos)

                    self.pos_embedded = self.pos_embedding[:, -shape[1]:, :]
                    batch_embedded = tf.concat([batch_embedded, self.pos_embedded], axis=2)

                if self.layer_1_include:
                    hid = 2 * self.hidden_size

                    if self.layer_1 == 'lstm':
                        rnn_outputs, _ = bi_rnn(LSTMCell(self.hidden_size, use_peepholes=self.peephole_1),
                                                LSTMCell(self.hidden_size, use_peepholes=self.peephole_2),
                                                inputs=batch_embedded, dtype=tf.float32, scope='rnn_1')

                        fw_outputs, bw_outputs = rnn_outputs
                        layer = tf.concat([fw_outputs, bw_outputs], axis=2)
                    elif self.layer_1 == 'gru':
                        rnn_outputs, _ = bi_rnn(GRUCell(self.hidden_size),
                                                GRUCell(self.hidden_size),
                                                inputs=batch_embedded, dtype=tf.float32, scope='rnn_1')

                        fw_outputs, bw_outputs = rnn_outputs
                        layer = tf.concat([fw_outputs, bw_outputs], axis=2)
                    else:
                        conv_layer = tf.layers.conv1d(inputs=batch_embedded,
                                                      filters=self.hidden_size * 2,
                                                      kernel_size=self.kernel_size,
                                                      strides=1,
                                                      padding="same",
                                                      activation=tf.nn.relu)
                        layer = conv_layer
                else:
                    layer = batch_embedded
                    hid = self.hidden_size
                    if self.pos_include:
                        hid += self.pos_dimensions

            print(self.hidden_size)

            # FLAGS Including ELMO and Glove
            if self.glove_include and self.elmo:
                H_1 = tf.concat([layer, self.x_elmo], axis=2)
                hid += 1024
            elif self.glove_include:
                H_1 = layer
            elif self.elmo:
                H_1 = self.x_elmo
                hid = 1024

            if self.layer_2 == 'lstm':
                rnn_outputs_2, _ = bi_rnn(LSTMCell(hid, use_peepholes=self.peephole_3),
                                          LSTMCell(hid, use_peepholes=self.peephole_4),
                                          inputs=H_1, dtype=tf.float32, scope='rnn_2')

                fw_outputs_2, bw_outputs_2 = rnn_outputs_2
                H = tf.concat([fw_outputs_2, bw_outputs_2], axis=2)
            elif self.layer_2 == 'gru':
                rnn_outputs_2, _ = bi_rnn(GRUCell(hid),
                                          GRUCell(hid),
                                          inputs=H_1, dtype=tf.float32, scope='rnn_2')

                fw_outputs_2, bw_outputs_2 = rnn_outputs_2
                H = tf.concat([fw_outputs_2, bw_outputs_2], axis=2)
            elif self.layer_2 == 'conv':
                conv_layer = tf.layers.conv1d(inputs=H_1,
                                              filters=hid,
                                              kernel_size=self.kernel_size,
                                              strides=1,
                                              padding="same",
                                              activation=tf.nn.relu)
                H = conv_layer
                hid = tf.cast(hid / 2, tf.int32)
            else:
                H = H_1
                hid = tf.cast(hid / 2, tf.int32)

            hid *= 2

            ### Ask whether there is a sequence with length 0 ###
            condition = tf.equal(tf.reduce_min(self.seq_len), 0)

            ### FLAG Including attention ###
            if self.attention:
                with tf.variable_scope('attention', reuse=self.reuse):
                    M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

                    dropout_layer_attention = tf.layers.dropout(inputs=tf.reshape(M, [-1, hid]),
                                                                rate=self.attention_prob, training=self.is_training,
                                                                seed=847)
                    self.dense = tf.layers.dense(inputs=dropout_layer_attention, units=self.num_attention,
                                                 use_bias=False)
                    ### Pool - Max or Mean ###
                    if self.pool_mean:
                        self.pool = tf.reduce_mean(self.dense, axis=1)
                    else:
                        self.pool = tf.reduce_max(self.dense, axis=1)

                    ### Setting for stride 2 ###
                    #self.alpha = tf.exp(tf.reshape(self.pool,
                    #         [-1, tf.cast(tf.round(tf.add(tf.div(tf.cast(shape[1], dtype = tf.float32), 2.0), 0.1)),
                    #                      dtype = tf.int32)]))
                    self.alpha = tf.exp(tf.reshape(self.pool, [-1, shape[1]]))

                    ### Masking the sequences ###
                    if self.mask:
                        with tf.variable_scope('mask', reuse=self.reuse):
                            self.alpha = tf.reverse(self.alpha, axis=[1])
                            mask = tf.sequence_mask(self.seq_len)
                            mask = tf.to_float(mask)

                            self.alpha = tf.cond(condition, lambda: self.alpha, lambda: self.alpha * mask)
                            self.alpha = tf.reverse(self.alpha, axis=[1])

                    #### Softmax ####
                    self.alpha = self.alpha / tf.expand_dims(tf.reduce_sum(self.alpha, axis=1), 1)

                    ### Derive the word with the highest attention ###
                    pos = tf.argmax(self.alpha, axis=1)
                    sparse_tensor = tf.string_split(self.x_elmo_input)
                    dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, '')
                    rg = tf.range(0, shape[0])
                    indices = tf.transpose([rg, tf.cast(pos, tf.int32)], [1, 0])
                    self.best_example = tf.gather_nd(dense_tensor, indices)

                    ### Computing weighted average ###
                    # r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha,
                    #                                                      [-1, tf.cast(tf.round(tf.add(
                    #                                                          tf.div(tf.cast(shape[1], dtype=tf.float32),
                    #                                                                 2.0), 0.1)),
                    #                                                                   dtype=tf.int32), 1]))
                    r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                                 tf.reshape(self.alpha, [-1, shape[1], 1]))
                    r = tf.squeeze(r, axis=2)
            else:
                with tf.variable_scope('rnn_average', reuse=self.reuse):
                    ### Take a simple mean of all the words (INCLUDING padding) ###
                    ### Masking the sequences ###
                    if self.mask:
                        with tf.variable_scope('mask', reuse=self.reuse):
                            self.alpha = tf.cond(condition,
                                                 lambda: tf.tile(tf.expand_dims(shape[1], 0),
                                                                 tf.expand_dims(shape[0], 0)), lambda: self.seq_len)
                            self.alpha = tf.reciprocal(tf.to_float(self.alpha))
                            self.alpha = tf.tile(tf.expand_dims(self.alpha, 1), [1, shape[1]])

                            self.alpha = tf.reverse(self.alpha, axis=[1])
                            mask = tf.sequence_mask(self.seq_len)
                            mask = tf.to_float(mask)

                            self.alpha = tf.cond(condition, lambda: self.alpha, lambda: self.alpha * mask)
                            self.alpha = tf.reverse(self.alpha, axis=[1])
                    else:
                        self.alpha = tf.tile(tf.expand_dims(shape[1], 0), tf.expand_dims(shape[0], 0))
                        self.alpha = tf.reciprocal(tf.to_float(self.alpha))
                        self.alpha = tf.tile(tf.expand_dims(self.alpha, 1), [1, shape[1]])

                    ### Necessarily here but serves no purpose - Derive the word with the highest attention ###
                    pos = tf.argmax(self.alpha, axis=1)
                    sparse_tensor = tf.string_split(self.x_elmo_input)
                    dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, '')
                    rg = tf.range(0, shape[0])
                    indices = tf.transpose([rg, tf.cast(pos, tf.int32)], [1, 0])
                    self.best_example = tf.gather_nd(dense_tensor, indices)

                    ### Computing average ###
                    r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                                  tf.reshape(self.alpha, [-1, shape[1], 1]))
                    r = tf.squeeze(r, axis=2)

            self.h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE)

    def build_dense(self):
        """ Build the output layer of the graph. """
        self.y_hat = {}
        for task, i in zip(['a', 'b', 'c'], range(3)):
            with tf.variable_scope('dense_' + task, reuse=self.reuse):
                dropout_layer = tf.layers.dropout(inputs=self.h_star, rate=self.keep_prob, training=self.is_training,
                                                  seed=345)
                self.y_hat[task] = tf.layers.dense(inputs=dropout_layer, units=self.classes[task])

    def build_loss(self):
        """ Build the loss of the graph"""
        self.prediction = {}
        self.loss_dict = {}
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        self.num_correct_predictions = {}
        self.batch_accuracy = {}
        self.probabilities = {}
        for task, i in zip(['a', 'b', 'c'], range(3)):
            with tf.name_scope(task):
                label = self.labels[task]
                y_hat = self.y_hat[task]
                self.probabilities[task] = tf.nn.softmax(self.y_hat[task])

                with tf.name_scope('loss_' + task):
                    weight = tf.cast(tf.gather_nd(self.weights[i], tf.expand_dims(label, 1)), tf.float32)
                    self.loss_dict[task] = tf.reduce_mean(
                        tf.cast(weight, tf.float32) * tf.losses.sparse_softmax_cross_entropy(logits=y_hat,
                                                                                             labels=label))

                    if task in self.tasks:
                        self.loss += self.loss_dict[task]
                with tf.name_scope('accuracy_' + task):
                    # prediction
                    pred = tf.argmax(tf.nn.softmax(y_hat), 1)

                    if task == 'b' and self.first_cause:
                        bools = self.prediction['a'] < 1
                        pred *= tf.cast(bools, tf.int64)
                    elif task == 'c' and self.second_cause:
                        bools = self.prediction['b'] < 2
                        pred *= tf.cast(bools, tf.int64)

                    self.prediction[task] = pred

                    #             Return a bool tensor with shape [batch_size] that is true for the correct predictions.
                    self.correct_predictions = tf.equal(tf.cast(pred, tf.int32), label)

                    ### Metrics ###
                    self.precision[task] = tf.py_func(precision_func, inp=[label, tf.cast(pred, tf.int32)],
                                                      Tout=tf.float64, name='my_metric_precision')
                    self.recall[task] = tf.py_func(recall_func, inp=[label, tf.cast(pred, tf.int32)], Tout=tf.float64,
                                                   name='my_metric_recall')
                    self.f1[task] = tf.py_func(f1_func, inp=[label, tf.cast(pred, tf.int32)], Tout=tf.float64,
                                               name='my_metric_f1')

                    # Number of correct predictions in order to calculate average accuracy afterwards.
                    self.num_correct_predictions[task] = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))
                    # Calculate the accuracy per mini-batch.
                    self.batch_accuracy[task] = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

                    print("Building loss for task " + task + " built successfully!")

    def get_num_parameters(self):
        """
        Count and return the total number of parameters in the graph.

        :return: total number of trainable parameters.
        """
        # Iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # multiplying dimension values
            self.num_parameters += local_parameters

        return self.num_parameters


def precision_func(label, prediction):
    """
    Calculate and return the precision.


    :param label:
    :param prediction:
    :return precision:
    """
    return precision_score(label, prediction, average='macro')


def recall_func(label, prediction):
    """
    Calculate and return the recall.


    :param label:
    :param prediction:
    :return recall:
    """
    return recall_score(label, prediction, average='macro')


def f1_func(label, prediction):
    """
    Calculate and return the f1 score.

    :param label:
    :param prediction:
    :return f1 score
    """
    return f1_score(label, prediction, average='macro')


def ElmoEmbedding(x, elmo_model):
    """
    Create an ELMo embedding and return its output.

    :param x:
    :param elmo_model:
    :return embedding:
    """
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
