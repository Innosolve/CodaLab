### PEP - 8 Import Standard ###

import io
import numpy as np

import matplotlib
import tensorflow as tf

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def is_ascii(s):
    """
    Checks a string whether ascii encoded and return decoded string.

    :param s:
    :return:
    """
    try:
        return s.decode('ascii')
    except:
        return ''


def create_attention_table(alpha, batch):
    """
    Creates a table with the sentence and the corresponding attention coefficient expressed in a color.

    :param alpha:
    :param batch:
    :return:
    """
    split_up = [[''] * (len(alpha[0]) - len(x.split())) + [is_ascii(y) for y in x.split()] for x in batch[0]]
    split_up_strings = split_up

    variables = split_up_strings
    colors = alpha
    normal = Normalize(colors.min() - 1, colors.max() + 1)

    try:
        variables = np.array(variables)
        shape = variables.shape
        variables = np.concatenate(variables)
        variables = list(map(lambda x: x.decode("utf-8"), variables))
        variables = np.reshape(variables, shape)
        table = plt.table(cellText=variables, colWidths=[0.1] * np.array(variables).shape[1], loc='center', \
                          cellColours=plt.cm.hot(normal(colors)))
    except:
        table = plt.table(cellText='empty')

    table.auto_set_font_size(False)
    table.set_fontsize(6)

    plt.axis('off')
    plt.grid('off')

    # prepare for saving:
    # draw canvas once
    plt.gcf().canvas.draw()
    # get bounding box of table
    points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
    # add 10 pixel spacing
    points[0, :] -= 10;
    points[1, :] += 10
    # get new bounding box in inches
    nbbox = matplotlib.transforms.Bbox.from_extents(points / plt.gcf().dpi)
    # save and clip by new bounding box

    buf = io.BytesIO()
    plt.savefig(buf, dpi=900, bbox_inches=nbbox, )
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)

    plt.cla()
    plt.clf()
    buf.close()

    return image
