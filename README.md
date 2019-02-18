# SemEval Task 6

### Author: Koen Vernooij

These instructions will get you a copy of the project up and urinning on your local machine for testing purposes.

### Run the Task a) model by running.

python training.py

### FLAGS

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

    parser.add_argument('-elmo', action='store_false')  ### Include ELMo Embedding ###

    parser.add_argument('-glove', action='store_false')  ### Include glove embedding ###

    parser.add_argument('-model_dir', type=str, default='graphs')  ### Model Directory ###

    parser.add_argument('-name', type=str, default='default_name')  ### Graph name ###

    parser.add_argument('-attention', action='store_false')  ### Include attention ###

    parser.add_argument('-attention_vectors', type=int, default=20)  ### REDUNDENT ###

    parser.add_argument('-mask', action='store_false')  ### Mask the sentences ###

    parser.add_argument('-fold', type=int, default=2)  ### Number of folds in cross validation ###

    parser.add_argument('-layer_1', type=str, default='conv')  ### Layer 1 types: conv, gru, lstm ###

    parser.add_argument('-layer_2', type=str, default='gru')  ### Layer 1 types: conv, gru, lstm ###

    parser.add_argument('-cross', action='store_true')  ### Activate Cross Validation ###

    parser.add_argument('-train_data', type=str,

                        default='multi_classes')  ### Select train data, only with write_data TRUEE ###

    parser.add_argument('-test_data', type=str,

                        default='multi_classes')  ### Select test data, only with write_data TRUE ###

    parser.add_argument('-layer_1_include', action='store_false')  ### Exclude layer 1 ###

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

                        action='store_false')  ### Add a human attention mechanism to training, refer to PAPER ###

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

