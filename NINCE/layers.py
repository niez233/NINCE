from keras import activations, initializers, constraints
from keras import regularizers
# import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import Model
# from keras.engine.topology import Layer
import tensorflow as tf

K = tf.keras.backend
# 这里我将Layer改为了Model

'''
Self-Attention Layer
'''
class Self_Attention(tf.keras.Model):

    def __init__(self, nb_head, size_per_head, **kwargs):

        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Self_Attention, self).__init__(**kwargs)


    def build(self, input_shape):

        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim), initializer='uniform', trainable=True)
        super(Self_Attention, self).build(input_shape)

    def call(self, x):

        Q_seq = K.dot(x, self.kernel[0])
        K_seq = K.dot(x, self.kernel[1])
        V_seq = K.dot(x, self.kernel[2])

        # rehsape [B, N, HS] --> [B, N, H, S]
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))

        # transpose [B, N, H, S] --> [B, H, N, S]
        Q_seq = K.permute_dimensions(Q_seq, [0,2,1,3])
        K_seq = K.permute_dimensions(K_seq, [0,2,1,3])
        V_seq = K.permute_dimensions(V_seq, [0,2,1,3])


        # Attention
        QK = tf.matmul(Q_seq, K_seq, transpose_b=True)

        # QK = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) # [B, H, N, N]
        # print(K.int_shape(QK))
        QK = QK / (self.size_per_head**0.5)
        QK = K.softmax(QK)

        z = tf.matmul(QK, V_seq)
        # z = K.batch_dot(QK, V_seq, axes=[3,2]) # [B, H, N, S_v]
        z = K.permute_dimensions(z, [0,2,1,3]) # [B, H, N, S_v] --> [B, N, H, S_v]
        z = K.reshape(z, (-1, K.shape(z)[1], self.output_dim))

        return z

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1], self.output_dim)

'''
GCN layer
clone from https://github.com/tkipf/keras-gcn/tree/master/kegra/layers
'''
class GraphConvolution(tf.keras.Model):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias is not None:  # Change this line
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttLayer(tf.keras.Model):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = tf.keras.backend.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = tf.keras.backend.variable(self.init((self.attention_dim, )), name='b')
        self.u = tf.keras.backend.variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = tf.keras.backend.tanh(tf.keras.backend.bias_add(K.dot(x, self.W), self.b))
        ait = tf.keras.backend.dot(uit, self.u)
        ait = tf.keras.backend.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= tf.keras.backend.cast(mask, tf.keras.backend.floatx())
        ait /= tf.keras.backend.cast(tf.keras.backend.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = tf.keras.backend.expand_dims(ait)
        weighted_input = x * ait
        output = tf.keras.backend.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class CoAttLayer(tf.keras.Model):

    def __init__(self, k, **kwargs):
        super(CoAttLayer, self).__init__(**kwargs)
        self.latent_dim = 64
        self.k = k

    def build(self, input_shape):
        # 使用add_weight来定义权重
        self.Wl = self.add_weight(name='Wl',
                                  shape=(self.latent_dim, self.latent_dim),
                                  initializer='normal',
                                  trainable=True)

        self.Wr = self.add_weight(name='Wr',
                                  shape=(self.k, self.latent_dim),
                                  initializer='normal',
                                  trainable=True)
        self.Wp = self.add_weight(name='Wp',
                                  shape=(self.k, self.latent_dim),
                                  initializer='normal',
                                  trainable=True)

        self.whr = self.add_weight(name='whr',
                                   shape=(1, self.k),
                                   initializer='normal',
                                   trainable=True)
        self.whp = self.add_weight(name='whp',
                                   shape=(1, self.k),
                                   initializer='normal',
                                   trainable=True)
        # 注意，不再需要手动设置 trainable_weights
        super(CoAttLayer, self).build(input_shape)  # 确保这一行在最后

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):

        review_seq = x[0]
        post_seq = x[1]

        review_seq_trans = tf.keras.backend.permute_dimensions(review_seq, (0, 2, 1))
        post_seq_trans = tf.keras.backend.permute_dimensions(post_seq, (0, 2, 1))
        L = tf.keras.backend.tanh(tf.einsum('btd,dD,bDn->btn', review_seq, self.Wl, post_seq_trans))
        L_trans = tf.keras.backend.permute_dimensions(L, (0, 2, 1))

        Hp = tf.keras.backend.tanh(tf.einsum('kd,bdn->bkn', self.Wp, post_seq_trans) + tf.einsum('kd,bdt,btn->bkn', self.Wr, review_seq_trans, L))
        Hr = tf.keras.backend.tanh(tf.einsum('kd,bdt->bkt', self.Wr, review_seq_trans) + tf.einsum('kd,bdn,bnt->bkt', self.Wp, post_seq_trans, L_trans))
        Ap = tf.keras.backend.softmax(tf.einsum('yk,bkn->bn', self.whp, Hp))
        Ar = tf.keras.backend.softmax(tf.einsum('yk,bkt->bt', self.whr, Hr))
        co_p = tf.einsum('bdn,bn->bd', post_seq_trans, Ap)
        co_r = tf.einsum('bdt,bt->bd', review_seq_trans, Ar)
        co_pr = tf.keras.backend.concatenate([co_p, co_r], axis=1)

        return co_pr

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.latent_dim + self.latent_dim)