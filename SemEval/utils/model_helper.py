### PEP - 8 Import Standard ###

import numpy as np


def make_train_feed_dict(model, batch):
    """
    Create feed_dict for training.

    :param model:
    :param batch:
    :return feed_dict:
    """
    feed_dict = {model.x_elmo_input: batch[0],
                 model.x: batch[1],
                 model.pos: batch[2],
                 model.labels['a']: batch[3],
                 model.labels['b']: batch[4],
                 model.labels['c']: batch[5],
                 model.seq_len: batch[6],
                 model.glove: batch[7],
                 model.keep_prob: .5}
    return feed_dict


def make_attention_feed_dict(model, batch):
    """
    Create feed_dict for human attention training.

    :param model:
    :param batch:
    :return feed_dict:
    """
    feed_dict = {model.x_elmo_input: batch[0],
                 model.x: batch[1],
                 model.pos: batch[2],
                 model.attention_label: batch[3],
                 model.seq_len: batch[4],
                 model.glove: batch[5],
                 model.keep_prob: .5}
    return feed_dict


def F1_score(x, y):
    return 2 * x * y / (x + y)


def make_test_feed_dict(model, batch):
    """
    Create feed_dict for training.

    :param model:
    :param batch:
    :return feed_dict:
    """
    feed_dict = {model.x_elmo_input: batch[0],
                 model.x: batch[1],
                 model.pos: batch[2],
                 model.labels['a']: batch[3],
                 model.labels['b']: batch[4],
                 model.labels['c']: batch[5],
                 model.seq_len: batch[6],
                 model.glove: batch[7],
                 model.keep_prob: 1.0}

    return feed_dict


def run_train_step(model, sess, batch, summaries):
    """
    Run training step.

    :param model:
    :param sess:
    :param batch:
    :param summaries:
    :return:
    """
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'loss': model.loss,
        'train_op': model.train_op,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    """
    Run evaluation step.

    :param model:
    :param sess:
    :param batch:
    :return:
    """
    feed_dict = make_test_feed_dict(model, batch)
    prediction, loss = sess.run([model.prediction, model.loss], feed_dict)
    eq = np.equal(prediction, batch[2])
    ind = np.where(eq - 1)
    acc = np.sum(eq) / len(prediction)
    wclas = np.take(batch[0], ind)[0]

    ### True Positives, Precision, Recall ###
    tp = np.inner(eq, prediction)
    precision = tp / np.sum(prediction)
    recall = tp / np.sum(batch[2])
    f1 = F1_score(precision, recall)
    return acc, loss, wclas, precision, recall, f1


def get_attn_weight(model, sess, batch):
    """
    Get the attention weights.

    :param model:
    :param sess:
    :param batch:
    :return weights:
    """
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
