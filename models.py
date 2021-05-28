import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GRU
from tensorflow.keras.constraints import Constraint


class DiagonalConstraint(Constraint):
    def __init__(self, **kwargs):
        super(DiagonalConstraint, self).__init__(**kwargs)

    def __call__(self, w):
        dim = w.shape[0]
        m = tf.eye(dim)
        w = w * m
        return w


class TemporalDecay(Layer):
    def __init__(self, units, diag=False, **kwargs):
        super(TemporalDecay, self).__init__(**kwargs)

        self.units = units
        self.diag = diag

    def build(self, input_shape):
        if self.diag:
            assert self.units == input_shape[-1]
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_uniform', trainable=True,
                                     constraint=DiagonalConstraint(), name='kernel')
        else:
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='he_uniform', trainable=True,
                                     name='kernel')

        self.b = self.add_weight(shape=self.units, initializer='zeros', trainable=True, name='bias')

    @tf.function
    def call(self, inputs):
        gamma = tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        gamma = tf.math.exp(-gamma)
        return gamma


class GRUD(Layer):
    def __init__(self, hid_dim, dropout_rate, name='GRU-D', **kwargs):
        super(GRUD, self).__init__(name=name, **kwargs)

        self.hid_dim = hid_dim
        self.dropout_rate = dropout_rate
        self.inputs_mean = 0

    def build(self, input_shape):
        self.seq_len = input_shape[2]
        self.var_dim = input_shape[3]

        self.rnn_cell = GRU(units=self.hid_dim, recurrent_dropout=self.dropout_rate, name=self.name + '/gru')

        self.temp_decay_x = TemporalDecay(units=self.var_dim, diag=True, name=self.name + '/temp_decay_x')
        self.temp_decay_h = TemporalDecay(units=self.hid_dim, diag=False, name=self.name + '/temp_decay_h')

        self.out = Dense(units=1, activation='sigmoid', kernel_initializer='he_uniform', name=self.name + '/out')

    @tf.function
    def call(self, inputs):
        """
        Arguments:
            inputs -- shape: (N, (x, m, d, f), t, d)
            inputs:
                x -- normalized by standard gaussian distribution N(0, 1)
                m -- observed 1, missed 0
                d -- time delta
                f -- forward
        Returns:
            outputs -- shape: (N, 1)
        """

        # GRU state
        state_h = tf.zeros(shape=(self.hid_dim,), dtype=tf.float32)

        for t in range(self.seq_len):
            x = inputs[:, 0, t, :]
            m = inputs[:, 1, t, :]
            d = inputs[:, 2, t, :]
            f = inputs[:, 3, t, :]

            gamma_x = self.temp_decay_x(d)
            x_hat = gamma_x * f + (1 - gamma_x) * self.inputs_mean
            x_c = m * x + (1 - m) * x_hat

            inputs_cell = tf.concat([x_c, m], axis=1)

            gamma_h = self.temp_decay_h(d)
            state_h = gamma_h * state_h

            state_h = self.rnn_cell(inputs=tf.expand_dims(inputs_cell, axis=1), initial_state=[state_h])

        outputs = self.out(state_h)

        return outputs
