import tensorflow as tf
from tensorflow.contrib import layers

class Seq2seq:
    def __init__(self, vocab_size, FLAGS):
        self.FLAGS = FLAGS
        self.vocab_size = vocab_size

    def make_graph(self,mode, features, labels, params):
        embed_dim = params.embed_dim
        num_units = params.num_units

        input,output   = features['input'], features['output']
        batch_size     = tf.shape(input)[0]
        start_tokens   = tf.zeros([batch_size], dtype= tf.int64)
        train_output   = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        input_lengths  = tf.reduce_sum(tf.to_int32(tf.not_equal(input, 1)), 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)
        input_embed    = layers.embed_sequence(input, vocab_size=self.vocab_size, embed_dim = embed_dim, scope = 'embed')
        output_embed   = layers.embed_sequence(train_output, vocab_size=self.vocab_size, embed_dim = embed_dim, scope = 'embed', reuse = True)
        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')
        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
        if self.FLAGS.use_residual_lstm:
            cell = tf.contrib.rnn.ResidualWrapper(cell)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, input_embed, dtype=tf.float32)


        def decode(helper, scope, reuse=None):
            # Decoder is partially based on @ilblackdragon//tf_example/seq2seq.py
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=num_units, memory=encoder_outputs,
                    memory_sequence_length=input_lengths)
                cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=num_units / 2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, self.vocab_size, reuse=reuse)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=batch_size))
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=self.FLAGS.output_max_length)
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

