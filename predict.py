import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import tensorflow as tf
from seq2seq import Seq2seq
from data_handler import Data
from IPython import embed

FLAGS = tf.app.flags.FLAGS

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

class Predict:
    def __init__(self):
        self.data  = Data(FLAGS)
        model = Seq2seq(self.data.vocab_size, FLAGS)
        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=FLAGS.model_dir)
        def input_fn():
            inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
            output = tf.placeholder(tf.int64, shape=[None, None], name='output')
            tf.identity(inp[0], 'source')
            tf.identity(output[0], 'target')
            dict =  { 'input': inp, 'output': output}
            return tf.estimator.export.ServingInputReceiver(dict, dict)
        self.predictor = tf.contrib.predictor.from_estimator(estimator, input_fn)

    def infer(self, sentence):
        input = self.data.prepare(sentence)
        predictor_prediction = self.predictor({"input": input, "output":input})
        words = [self.data.rev_vocab.get(i, '<UNK>') for i in predictor_prediction['output'][0] if i > 2]
        return ' '.join(words)

def main(args):
    P = Predict()
    print(P.infer('the bike has a clock as a tire'))
    print(P.infer('an old teal colored car parked on the street'))

if __name__ == "__main__":
    tf.app.run()
