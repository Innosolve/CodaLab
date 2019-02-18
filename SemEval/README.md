# SemEval Task 6
  
### Author: Koen Vernooij

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### FLAGS

The options for FLAGS have been specified at the bottom of the `training.py` file.

### Data

The training data is on https://competitions.codalab.org/competitions/20011.
Before execution training the '@USER' mentions need to be removed and the punctuation has to be surrounded by spaces.

The embedding used were the glove embeddings, which can be found at http://nlp.stanford.edu/data/glove.twitter.27B.zip.

Both data sets need to be put in a data folder before running.

### Command.

In order to run the Task a) model run.

```
python training.py -peephole_3 -peephole_4
```

In order to run the Task b) model run.

```
python training.py -layer_2 gru -tasks b -remove_stop -formal_filter -attention -mask
```

In order to run the Task b) model run.

```
python training.py -fund_embed_dim 50 -decay_rate 0.85 -learning_rate 1e-4 -tasks c -remove_stop -num_attention 10 -pool_mean
```