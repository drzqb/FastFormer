'''
    tf 2.5 for FastFormer
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.initializers import TruncatedNormal, Ones, Zeros
import argparse
import os
from OtherUtils import load_vocab
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--maxword', default=512, type=int, help='The max length of input sequence')
parser.add_argument('--type_vocab_size', default=2, type=int, help='type_vocab_size')
parser.add_argument('--vocab_size', default=21128, type=int, help='type_vocab_size')
parser.add_argument('--drop_rate', default=0.1, type=float, help='rate for dropout')
parser.add_argument('--block', type=int, default=12, help='number of Encoder submodel')
parser.add_argument('--head', type=int, default=12, help='number of multi_head attention')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=200000000, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--hidden_size', type=int, default=768, help='Embedding size for QA words')
parser.add_argument('--intermediate_size', type=int, default=3072, help='Embedding size for QA words')
parser.add_argument('--categories', type=int, default=7, help='number of categories')
parser.add_argument('--ner_labels', type=int, default=13, help='number of ner labels')
parser.add_argument('--check', type=str, default='modelfiles/bert-tf2', help='The path where modelfiles shall be saved')
parser.add_argument('--mode', type=str, default='predict', help='The mode of train or predict as follows: '
                                                                'train0: begin to train or retrain'
                                                                'tran1:continue to train'
                                                                'predict: predict')
parser.add_argument('--per_save', type=int, default=10000, help='save modelfiles for every per_save')

params = parser.parse_args()


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def gelu(x):
    return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_model_weights_from_checkpoint(model,
                                       checkpoint_file,
                                       ):
    """Load trained official modelfiles from checkpoint.

    :param model: Built keras modelfiles.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    """
    loader = checkpoint_loader(checkpoint_file)

    weights = [
        loader('bert/embeddings/word_embeddings'),
        loader('bert/embeddings/token_type_embeddings'),
        loader('bert/embeddings/position_embeddings'),
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ]
    model.get_layer('embeddings').set_weights(weights)

    for i in range(params.block):
        pre = 'bert/encoder/layer_' + str(i) + '/'
        w1 = model.get_layer('attention-' + str(i)).get_weights()

        weights = [
            loader(pre + 'attention/self/query/kernel'),
            loader(pre + 'attention/self/query/bias'),
            loader(pre + 'attention/self/key/kernel'),
            loader(pre + 'attention/self/key/bias'),
            loader(pre + 'attention/self/value/kernel'),
            loader(pre + 'attention/self/value/bias'),
            loader(pre + 'attention/output/dense/kernel'),
            loader(pre + 'attention/output/dense/bias'),
            loader(pre + 'attention/output/LayerNorm/gamma'),
            loader(pre + 'attention/output/LayerNorm/beta'),
        ]
        w1[:6] = weights[:6]
        w1[12:] = weights[6:]

        model.get_layer('attention-' + str(i)).set_weights(w1)

        weights = [
            loader(pre + 'intermediate/dense/kernel'),
            loader(pre + 'intermediate/dense/bias'),
            loader(pre + 'output/dense/kernel'),
            loader(pre + 'output/dense/bias'),
            loader(pre + 'output/LayerNorm/gamma'),
            loader(pre + 'output/LayerNorm/beta'),
        ]
        model.get_layer('feedford-' + str(i)).set_weights(weights)

    weights = [
        loader('cls/predictions/transform/dense/kernel'),
        loader('cls/predictions/transform/dense/bias'),
        loader('cls/predictions/transform/LayerNorm/gamma'),
        loader('cls/predictions/transform/LayerNorm/beta')
    ]
    model.get_layer('sequence').set_weights(weights)

    #
    # weights = [
    #     loader('bert/pooler/dense/kernel'),
    #     loader('bert/pooler/dense/bias'),
    # ]
    # model.get_layer('Pooler').set_weights(weights)

    weights = [
        loader('cls/predictions/output_bias'),
    ]
    model.get_layer('project').set_weights(weights)


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        mask = tf.where(tf.greater(sen, 0),
                        tf.zeros_like(sen, tf.float32),
                        (1.0 - tf.pow(2.0, 31.0)) * tf.ones_like(sen, tf.float32))
        return tf.tile(tf.expand_dims(mask, axis=1), [1, params.head, 1])


class SequenceMask(Layer):
    def __init__(self, **kwargs):
        super(SequenceMask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        return tf.greater(sen, 0)


class Embeddings(Layer):
    def __init__(self, **kwargs):
        super(Embeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        self.word_embeddings = self.add_weight(name='word_embeddings',
                                               shape=[params.vocab_size, params.hidden_size],
                                               dtype=tf.float32,
                                               initializer=create_initializer())
        self.token_embeddings = self.add_weight(name='token_type_embeddings',
                                                shape=[params.type_vocab_size, params.hidden_size],
                                                dtype=tf.float32,
                                                initializer=create_initializer())
        self.position_embeddings = self.add_weight(name='position_embeddings',
                                                   shape=[params.maxword, params.hidden_size],
                                                   dtype=tf.float32,
                                                   initializer=create_initializer())
        self.layernorm = LayerNormalization(name='layernorm-pre', epsilon=1e-6)
        self.dropout = Dropout(rate=params.drop_rate)
        super(Embeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sen, token_type_ids = inputs
        sen_embed = tf.nn.embedding_lookup(self.word_embeddings, sen)
        token_embed = tf.nn.embedding_lookup(self.token_embeddings, token_type_ids)
        seq_length = tf.shape(sen)[1]
        return self.dropout(
            self.layernorm(sen_embed + token_embed + self.position_embeddings[:seq_length])), self.word_embeddings


# class Attention(Layer):
#     def __init__(self, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.dense_q = Dense(params.hidden_size,
#                              name='query',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.dense_k = Dense(params.hidden_size,
#                              name='key',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.dense_v = Dense(params.hidden_size,
#                              name='value',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.dense_o = Dense(params.hidden_size,
#                              name='output',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#         self.w_alpha = self.add_weight(name="w_alpha",
#                                        shape=[params.hidden_size // params.head],
#                                        dtype=tf.float32)
#         self.w_beta = self.add_weight(name="w_beta",
#                                       shape=[params.hidden_size // params.head],
#                                       dtype=tf.float32)
#
#         self.dense_u = Dense(params.hidden_size // params.head,
#                              name='upvalue',
#                              dtype=tf.float32,
#                              kernel_initializer=create_initializer())
#
#         self.dropout1 = Dropout(rate=params.drop_rate)
#         self.dropout2 = Dropout(rate=params.drop_rate)
#         self.dropout3 = Dropout(rate=params.drop_rate)
#         self.layernorm = LayerNormalization(name='layernormattn', epsilon=1e-6)
#
#         super(Attention, self).build(input_shape)
#
#     def softmax(self, a, mask):
#         """
#         :param a: B*ML1*ML2
#         :param mask: B*ML1*ML2
#         """
#         return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)
#
#     def call(self, inputs, **kwargs):
#         x, mask = inputs
#         q = tf.concat(tf.split(self.dense_q(x), params.head, axis=-1), axis=0)
#         k = tf.concat(tf.split(self.dense_k(x), params.head, axis=-1), axis=0)
#         v = tf.concat(tf.split(self.dense_v(x), params.head, axis=-1), axis=0)
#
#         alpha = tf.einsum("ijk,k->ij", q, self.w_alpha) / tf.sqrt(params.hidden_size / params.head)
#         alpha = tf.where(mask, alpha, (1.0 - 2. ** 31) * tf.ones_like(alpha))
#         alphascore = tf.expand_dims(self.dropout1(tf.nn.softmax(alpha, axis=-1)), axis=1)
#
#         q_av = tf.squeeze(tf.matmul(alphascore, q), axis=1)
#
#         p = tf.einsum("ijk,ik->ijk", k, q_av)
#
#         beta = tf.einsum("ijk,k->ij", p, self.w_beta) / tf.sqrt(params.hidden_size / params.head)
#         beta = tf.where(mask, beta, (1.0 - 2. ** 31) * tf.ones_like(beta))
#         betascore = tf.expand_dims(self.dropout2(tf.nn.softmax(beta, axis=-1)), axis=1)
#
#         k_av = tf.squeeze(tf.matmul(betascore, p), axis=1)
#
#         u = tf.einsum("ijk,ik->ijk", v, k_av)
#         r = self.dense_u(u)
#
#         attention_output = self.dense_o(tf.concat(tf.split(r + q, params.head), axis=-1))
#
#         return self.layernorm(x + self.dropout3(attention_output))


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_q = Dense(params.hidden_size,
                             name='query',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_k = Dense(params.hidden_size,
                             name='key',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.dense_v = Dense(params.hidden_size,
                             name='value',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())
        self.alpha = Dense(params.head,
                           name='alpha',
                           dtype=tf.float32,
                           kernel_initializer=create_initializer())
        self.beta = Dense(params.head,
                          name='beta',
                          dtype=tf.float32,
                          kernel_initializer=create_initializer())

        self.dense_u = Dense(params.hidden_size,
                             name='upvalue',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())

        self.dense_o = Dense(params.hidden_size,
                             name='output',
                             dtype=tf.float32,
                             kernel_initializer=create_initializer())

        self.dropout1 = Dropout(rate=params.drop_rate)
        self.dropout2 = Dropout(rate=params.drop_rate)
        self.dropout3 = Dropout(rate=params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormattn', epsilon=1e-6)

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # x: B*N*768 mask:B*12*N
        x, mask = inputs

        batch_size = tf.shape(x)[0]
        seqlen = tf.shape(x)[1]

        # B*N*768
        q = self.dense_q(x)
        k = self.dense_k(x)
        v = self.dense_v(x)

        # B*N*12
        alphascore = self.alpha(q) / (params.hidden_size / params.head) ** 0.5
        # B*12*N
        alphascore = tf.transpose(alphascore, [0, 2, 1])

        # B*12*N
        alphascore += mask

        # B*12*N
        alphaweight = self.dropout1(tf.nn.softmax(alphascore, axis=-1))

        # B*12*1*N
        alphaweight = tf.expand_dims(alphaweight, axis=2)

        # B*N*12*64
        qsplit = tf.reshape(q, [batch_size, seqlen, params.head, params.hidden_size // params.head])

        # B*12*N*64
        qsplit = tf.transpose(qsplit, [0, 2, 1, 3])

        # B*12*1*64-->B*1*12*64
        q_av = tf.transpose(tf.matmul(alphaweight, qsplit), [0, 2, 1, 3])

        # B*1*768
        q_av = tf.reshape(q_av, [-1, 1, params.hidden_size])

        # B*N*768
        q_av = tf.tile(q_av, [1, seqlen, 1])

        #########################################################################

        # B*N*768
        p = k * q_av

        # B*N*12
        betascore = self.beta(p) / (params.hidden_size / params.head) ** 0.5
        # B*12*N
        betascore = tf.transpose(betascore, [0, 2, 1])

        # B*12*N
        betascore += mask

        # B*12*N
        betaweight = self.dropout2(tf.nn.softmax(betascore, axis=-1))

        # B*12*1*N
        betaweight = tf.expand_dims(betaweight, axis=2)

        # B*N*12*64
        psplit = tf.reshape(p, [batch_size, seqlen, params.head, params.hidden_size // params.head])

        # B*12*N*64
        psplit = tf.transpose(psplit, [0, 2, 1, 3])

        # B*12*1*64-->B*1*12*64
        p_av = tf.transpose(tf.matmul(betaweight, psplit), [0, 2, 1, 3])

        # B*1*768
        p_av = tf.reshape(p_av, [-1, 1, params.hidden_size])

        # B*N*768
        p_av = tf.tile(p_av, [1, seqlen, 1])

        # B*N*768
        u = p_av * v

        # B*N*768
        r = self.dense_u(u)

        attention_output = self.dense_o(r + q)

        return self.layernorm(x + self.dropout3(attention_output))


class FeedFord(Layer):
    def __init__(self, **kwargs):
        super(FeedFord, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_ffgelu = Dense(params.intermediate_size,
                                  kernel_initializer=create_initializer(),
                                  dtype=tf.float32,
                                  name='intermediate',
                                  activation=gelu)
        self.dense_ff = Dense(params.hidden_size,
                              kernel_initializer=create_initializer(),
                              dtype=tf.float32,
                              name='output')
        self.dropout = Dropout(rate=params.drop_rate)
        self.layernorm = LayerNormalization(name='layernormffd', epsilon=1e-6)

        super(FeedFord, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.layernorm(x + self.dropout(self.dense_ff(self.dense_ffgelu(x))))


class Project(Layer):
    def __init__(self, **kwargs):
        super(Project, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[params.vocab_size],
                                           dtype=tf.float32)

    def call(self, inputs, **kwargs):
        x, embedmatrix = inputs
        return tf.einsum("ijk,lk->ijl", x, embedmatrix) + self.output_bias


class SplitPooler(Layer):
    def __init__(self, **kwargs):
        super(SplitPooler, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 0]


class SplitSequence(Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class Pooler(Layer):
    def __init__(self, **kwargs):
        super(Pooler, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(params.hidden_size,
                           name='pooler',
                           kernel_initializer=create_initializer(),
                           dtype=tf.float32,
                           activation=tf.tanh)
        super(Pooler, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.dense(x)


class Sequence(Layer):
    def __init__(self, **kwargs):
        super(Sequence, self).__init__(**kwargs)

    def build(self, input_shape):
        self.transformer = Dense(params.hidden_size,
                                 activation=gelu,
                                 kernel_initializer=create_initializer(),
                                 dtype=tf.float32,
                                 name='transformer')
        self.layernorm = LayerNormalization(name='layernormsuf', epsilon=1e-6)

        super(Sequence, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.layernorm(self.transformer(inputs))


class NER(Layer):
    def __init__(self, **kwargs):
        super(NER, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_ner = Dense(params.ner_labels,
                               kernel_initializer=create_initializer(),
                               dtype=tf.float32,
                               name='ner')

    def call(self, x, **kwargs):
        return self.dense_ner(x)


class Classify(Layer):
    def __init__(self, **kwargs):
        super(Classify, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_classify = Dense(params.categories,
                                    kernel_initializer=create_initializer(),
                                    dtype=tf.float32,
                                    name='classify')
        super(Classify, self).build(input_shape)

    def call(self, x, **kwargs):
        return self.dense_classify(x)


class USR:
    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        token_type = Input(shape=[None], name='token_type', dtype=tf.int32)
        mask = Mask()(sen)
        now, embedmatrix = Embeddings(name='embeddings')(inputs=(sen, token_type))
        for layers in range(params.block):
            now = Attention(name='attention-' + str(layers))(inputs=(now, mask))
            now = FeedFord(name='feedford-' + str(layers))(now)
        seq = Sequence(name="sequence")(now)

        logits = Project(name="project")(inputs=(seq, embedmatrix))

        model = Model(inputs=[sen, token_type], outputs=[logits])

        load_model_weights_from_checkpoint(model, 'pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')

        if summary:
            model.summary()
            for tv in model.variables:
                print(tv.name, tv.shape)

        return model

    def predict(self):
        model = self.build_model(summary=False)
        # new_model = Model(inputs=model.inputs,
        #                   outputs=model.get_layer('Pooler').output)  # 你创建新的模型
        prediction = tf.argmax(model.predict([sents, token_types]), axis=-1)[0].numpy()
        print("".join([char_inverse_dict[p] for p in prediction]))


if __name__ == '__main__':
    char_dict = load_vocab('pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')
    char_inverse_dict = {v: k for k, v in char_dict.items()}

    sentence = '所有董事均已出席了审议本报告的董事会会议。'
    sen2id = [[char_dict['[CLS]']] + [
        char_dict[word] if word in char_dict.keys() else char_dict['[UNK]']
        for word in sentence] + [char_dict['[SEP]']]]
    sents = np.array(sen2id, np.int32)
    token_types = np.zeros_like(sents)

    usr = USR()
    usr.predict()