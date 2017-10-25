import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from seq2seq import *

FLAGS = tf.flags.FLAGS

# Model related
tf.flags.DEFINE_integer('num_units'         , 256           , 'Number of units in a LSTM cell')
tf.flags.DEFINE_integer('embed_dim'         , 256           , 'Size of the embedding vector')

# Training related
tf.flags.DEFINE_float('learning_rate'       , 0.001         , 'learning rate for the optimizer')
tf.flags.DEFINE_string('optimizer'          , 'Adam'        , 'Name of the train source file')
tf.flags.DEFINE_integer('batch_size'        , 32            , 'random seed for training sampling')
tf.flags.DEFINE_integer('print_every'       , 100           , 'print records every n iteration')
tf.flags.DEFINE_integer('iterations'        , 10000         , 'number of iterations to train')
tf.flags.DEFINE_string('model_dir'          , 'checkpoints' , 'Directory where to save the model')

tf.flags.DEFINE_integer('input_max_length'  , 30            , 'Max length of input sequence to use')
tf.flags.DEFINE_integer('output_max_length' , 30            , 'Max length of output sequence to use')

tf.flags.DEFINE_bool('use_residual_lstm'    , True          , 'To use the residual connection with the residual LSTM')

# Data related
tf.flags.DEFINE_string('input_filename', 'data/mscoco/train_source.txt', 'Name of the train source file')
tf.flags.DEFINE_string('output_filename', 'data/mscoco/train_target.txt', 'Name of the train target file')
tf.flags.DEFINE_string('vocab_filename', 'data/mscoco/train_vocab.txt', 'Name of the vocab file')


def main(args):
    tf.logging._logger.setLevel(logging.INFO)

    model = Seq2seq(FLAGS)

    input_fn, feed_fn = model.make_input_fn()
    print_inputs = tf.train.LoggingTensorHook( ['source', 'target', 'predict'], every_n_iter=FLAGS.print_every, 
            formatter=model.get_formatter(['source', 'target', 'predict']))

    estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=FLAGS.model_dir, params=FLAGS)
    estimator.train(input_fn=input_fn, hooks=[tf.train.FeedFnHook(feed_fn), print_inputs], steps=FLAGS.iterations)

if __name__ == "__main__":
    tf.app.run()
