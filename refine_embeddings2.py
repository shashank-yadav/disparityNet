import numpy as np
import tensorflow as tf

rnn = tf.contrib.rnn

class EmbeddingsRefiner(object):
    """ Class to refine embeddings before matching
    """

    def __init__(self):
        self.num_refinement_steps = 5
        self.use_left_refinement = True
        self.use_right_refinement = True
        self.embedding_dimensions = 128

    def refine(self, left_hypercolumns, right_hypercolumns):
        """ refine hypercolumn embeddings of both left and right image
        :param left_hypercolumns - length L list of [batch_size, 128] tensors
        :param right_hypercolumns - length L list of [batch_size, 128] tensors
        """
        right_features_refined = self.fce_right(right_hypercolumns) # (L, batch_size, 128)
        left_features_refined = self.fce_left(left_hypercolumns, right_features_refined)

        return left_features_refined, right_features_refined

    def fce_left(self, left_hypercolumns, right_hypercolumns):
        """ refine hypercolumn for left image
        f(x_i, S) = attLSTM(f'(x_i), g(S), K)
        hypercolumn refinement is done by running LSTM for fixed no. of steps (num_refinement_steps)
        attention over hypercolumns of points on epipolar line in right image used as
        context vector of LSTM
        :param left_hypercolumn - [batch_size, 128] tensor (point feature)
        :param right_hypercolumns - [L, batch_size, 128]
        """
        L = len(left_hypercolumns)

        cells = [None] * L
        prev_states = [None] * L
        outputs = [None] * L

        for i in range(L):
            cells[i] = rnn.BasicLSTMCell(self.embedding_dimensions)
            prev_states[i] = cells[i].zero_state(batch_size, tf.float32)   # state[0] is c, state[1] is h

         for step in xrange(self.num_refinement_steps):
             for i in range(L):
                outputs[i], state = cells[i](left_hypercolumn, prev_state)  # output: (batch_size, 128)

                h_k = tf.add(outputs[i], left_hypercolumns[i]) # (batch_size, 128)

                content_based_attention = tf.nn.softmax(tf.multiply(prev_states[i][1], right_hypercolumns))    # (L, batch_size, 128)
                r_k = tf.reduce_sum(tf.multiply(content_based_attention, right_hypercolumns), axis=0)      # (batch_size, 128)

                prev_states[i] = rnn.LSTMStateTuple(state[0], tf.add(h_k, r_k))

        left_features_refined = tf.convert_to_tensor(outputs, dtype=tf.float32)
        return left_features_refined


    def fce_right(self, right_hypercolumns):
        """ refine hypercolumn for right image
        g(x_i, S) = h_i(->) + h_i(<-) + g'(x_i)
        Set information is incorporated into embedding using bidirectional LSTM
        :param right_hypercolumns - length L list of [batch_size, 128] tensors (point features)
        """
        # dimension of fw and bw is half, so that output has size embedding_dimensions
        fw_cell = rnn.BasicLSTMCell(self.embedding_dimensions / 2)
        bw_cell = rnn.BasicLSTMCell(self.embedding_dimensions / 2)

        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, right_hypercolumns, dtype=tf.float32)

        right_features_refined = tf.add(tf.stack(right_hypercolumns), tf.stack(outputs))
        return right_features_refined
