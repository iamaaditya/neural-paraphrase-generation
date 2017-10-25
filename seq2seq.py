# Some of the code here is taken from 
# https://github.com/ilblackdragon/tf_examples/blob/master/seq2seq/seq2seq.py

import tensorflow as tf
from tensorflow.contrib import layers

class Seq2seq:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.END_TOKEN = 1
        self.UNK_TOKEN = 2

        # create vocab and reverse vocab maps
        self.vocab     = {}
        self.rev_vocab = {}
        with open(FLAGS.vocab_filename) as f:
            for idx, line in enumerate(f):
                self.vocab[line.strip()] = idx
                self.rev_vocab[idx] = line.strip()

    def make_graph(self,mode, features, labels, params):
        vocab_size = len(self.vocab)
        embed_dim = params.embed_dim
        num_units = params.num_units

        input,output   = features['input'], features['output']
        batch_size     = tf.shape(input)[0]
        start_tokens   = tf.zeros([batch_size], dtype= tf.int64)
        train_output   = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        input_lengths  = tf.reduce_sum(tf.to_int32(tf.not_equal(input, 1)), 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
        input_embed    = layers.embed_sequence(input, vocab_size= vocab_size, embed_dim = embed_dim, scope = 'embed')
        output_embed   = layers.embed_sequence(train_output, vocab_size= vocab_size, embed_dim = embed_dim, scope = 'embed', reuse = True)
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)
        if self.FLAGS.use_residual_lstm:
            cell = tf.contrib.rnn.ResidualWrapper(cell)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)


        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=num_units, memory=encoder_outputs,
                    memory_sequence_length=input_lengths)
                cell = tf.contrib.rnn.GRUCell(num_units=num_units)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=num_units / 2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size, reuse=reuse)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=batch_size))
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=self.FLAGS.output_max_length
                )
                return outputs[0]

        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)
        train_outputs = decode(train_helper, 'decode')
        pred_outputs  = decode(pred_helper, 'decode', reuse=True)

        tf.identity(train_outputs.sample_id[0], name='train_pred')
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
        loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output, output, weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.optimizer,
            learning_rate=params.learning_rate,
            summaries=['loss', 'learning_rate'])

        tf.identity(pred_outputs.sample_id[0], name='predict')
        return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.sample_id, loss=loss, train_op=train_op)


    def tokenize_and_map(self,line):
        return [self.vocab.get(token, self.UNK_TOKEN) for token in line.split(' ')]


    def make_input_fn(self):
        def input_fn():
            inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
            output = tf.placeholder(tf.int64, shape=[None, None], name='output')
            tf.identity(inp[0], 'source')
            tf.identity(output[0], 'target')
            return {
                'input': inp,
                'output': output,
            }, None

        def sampler():
            while True:
                with open(self.FLAGS.input_filename) as finput, open(self.FLAGS.output_filename) as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()
                        yield {
                            'input': self.tokenize_and_map(in_line)[:self.FLAGS.input_max_length - 1] + [self.END_TOKEN],
                            'output': self.tokenize_and_map(out_line)[:self.FLAGS.output_max_length - 1] + [self.END_TOKEN]
                        }

        data_feed = sampler()
        def feed_fn():
            inputs, outputs = [], []
            input_length, output_length = 0, 0
            for i in range(self.FLAGS.batch_size):
                rec = data_feed.next()
                inputs.append(rec['input'])
                outputs.append(rec['output'])
                input_length = max(input_length, len(inputs[-1]))
                output_length = max(output_length, len(outputs[-1]))
            for i in range(self.FLAGS.batch_size):
                inputs[i] += [self.END_TOKEN] * (input_length - len(inputs[i]))
                outputs[i] += [self.END_TOKEN] * (output_length - len(outputs[i]))
            return { 'input:0': inputs, 'output:0': outputs }
        return input_fn, feed_fn

    def get_formatter(self,keys):
        def to_str(sequence):
            tokens = [
                self.rev_vocab.get(x, "<UNK>") for x in sequence]
            return ' '.join(tokens)

        def format(values):
            res = []
            for key in keys:
                res.append("****%s == %s" % (key, to_str(values[key])))
            return '\n'+'\n'.join(res)
        return format

