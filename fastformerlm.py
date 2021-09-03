'''
    tf 2.5 for FastFormer
    预训练语言模型
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from official.nlp.optimization import WarmUp, AdamWeightDecay
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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
parser.add_argument('--batch_size', type=int, default=4, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=100, help='Epochs during training')
parser.add_argument('--lr', type=float, default=5.0e-5, help='Initial learing rate')
parser.add_argument('--hidden_size', type=int, default=768, help='Embedding size for QA words')
parser.add_argument('--intermediate_size', type=int, default=3072, help='Embedding size for QA words')
parser.add_argument('--check', type=str, default='model/fastformerlm', help='The path where modelfiles shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
                                                               'train0: begin to train or retrain'
                                                               'tran1:continue to train'
                                                               'predict: predict')
parser.add_argument('--per_save', type=int, default=10000, help='save modelfiles for every per_save')

params = parser.parse_args()


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64)
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    return {"sen": sen}


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
        .repeat()

    return dataset


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

    weights = [
        loader('cls/predictions/output_bias'),
    ]
    model.get_layer('project').set_weights(weights)


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, sen, **kwargs):
        mask = tf.greater(sen, 0)
        mask_label = tf.less(tf.random.uniform(tf.shape(sen), 0.0, 1.0), 0.15)
        mask_label = tf.logical_and(mask_label, mask)

        noise = tf.where(mask_label, 103 * tf.ones_like(sen), sen)

        mask = tf.where(mask,
                        tf.zeros_like(sen, tf.float32),
                        (1.0 - tf.pow(2.0, 31.0)) * tf.ones_like(sen, tf.float32))

        return tf.tile(tf.expand_dims(mask, axis=1), [1, params.head, 1]), tf.cast(mask_label, tf.float32), noise


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

    def call(self, sen, **kwargs):
        token_type_ids = tf.zeros_like(sen)

        sen_embed = tf.nn.embedding_lookup(self.word_embeddings, sen)
        token_embed = tf.nn.embedding_lookup(self.token_embeddings, token_type_ids)
        seq_length = tf.shape(sen)[1]
        return self.dropout(
            self.layernorm(sen_embed + token_embed + self.position_embeddings[:seq_length])), self.word_embeddings


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


class MyLoss(Layer):
    def __init__(self, **kwargs):
        super(MyLoss, self).__init__(**kwargs)

        self.lossobj = SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, **kwargs):
        sen, logits, mask = inputs

        sumls = tf.reduce_sum(mask)

        loss = self.lossobj(sen, logits)
        loss *= mask

        loss = tf.reduce_sum(loss) / sumls

        self.add_loss(loss)

        predict = tf.argmax(logits, axis=-1, output_type=tf.int32)

        acc = tf.cast(tf.equal(predict, sen), tf.float32)
        acc *= mask
        acc = tf.reduce_sum(acc) / sumls

        self.add_metric(acc, name="acc")

        return logits


class CheckCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(params.check + "/fastformerlm.h5")


class USR:
    def build_model(self, summary=True):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)

        mask, masklabel, noise = Mask()(sen)

        now, embedmatrix = Embeddings(name='embeddings')(noise)

        for layers in range(params.block):
            now = Attention(name='attention-' + str(layers))(inputs=(now, mask))
            now = FeedFord(name='feedford-' + str(layers))(now)

        now = Sequence(name="sequence")(now)

        now = Project(name="project")(inputs=(now, embedmatrix))

        logits = MyLoss(name="endpoint")(inputs=(sen, now, masklabel))

        model = Model(inputs=[sen], outputs=[logits])

        tf.keras.utils.plot_model(model, to_file="FastFormerLM.jpg", show_shapes=True, dpi=200)

        if summary:
            model.summary(line_length=200)
            for tv in model.variables:
                print(tv.name, tv.shape)

        return model

    def train(self):
        path = 'D:/pythonwork/SimpleLMTF1/simplelm/data/TFRecordFile/'
        train_file = [
            path + 'trainpyfc1_people0.tfrecord',
            path + 'trainpyfc1_people1.tfrecord',
            path + 'trainpyfc1_people2.tfrecord',
            path + 'trainpyfc1_people3.tfrecord',
            path + 'trainpyfc1_people4.tfrecord',
            path + 'trainpyfc1_people5.tfrecord',
            path + 'trainpyfc1_people6.tfrecord',
            path + 'trainpyfc1_people7.tfrecord',
            path + 'trainpyfc1_people8.tfrecord',
            path + 'trainpyfc1_people9.tfrecord',
        ]

        if not os.path.exists(params.check):
            os.makedirs(params.check)

        batch_data = batched_data(train_file, single_example_parser, params.batch_size,
                                  padded_shapes={"sen": [-1]}, buffer_size=100 * params.batch_size)

        model = self.build_model()
        if params.mode == 'train0':
            load_model_weights_from_checkpoint(model,
                                               'pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')
            # model.save_weights(params.check + '/fastformerlm.h5')
        else:
            model.load_weights(params.check + '/fastformerlm.h5')

        decay_schedule = PolynomialDecay(initial_learning_rate=params.lr,
                                         decay_steps=params.epochs * params.per_save,
                                         end_learning_rate=0.0,
                                         power=1.0,
                                         cycle=False)

        warmup_schedule = WarmUp(initial_learning_rate=params.lr,
                                 decay_schedule_fn=decay_schedule,
                                 warmup_steps=2 * params.per_save)

        optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

        model.fit(batch_data,
                  epochs=params.epochs,
                  steps_per_epoch=params.per_save,
                  callbacks=[CheckCallback()]
                  )

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

    if params.mode.startswith('train'):
        usr.train()
    elif params.mode == "predict":
        usr.predict()
