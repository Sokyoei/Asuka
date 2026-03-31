"""
SimpleRNNCell, SimpleRNN, LSTM and GRU for imdb
"""

import tensorflow as tf

if tf.__version__ >= "2.16.0":
    from keras import datasets, layers, losses, models, optimizers, preprocessing
elif tf.__version__ >= "2.0.0":
    from tensorflow.keras import datasets, layers, losses, models, optimizers, preprocessing
else:
    raise ImportError("TensorFlow version must be >= 2.0.0")


tf.random.set_seed(1)

EPOCHS = 10
BATCH_SIZE = 128
TOTAL_WORDS = 10000
MAX_REVIEW_LEN = 80  # 序列最大长度（时间步数）
EMBEDDING_LEN = 100  # 词向量维度

# preprocess
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=TOTAL_WORDS)
# pad_sequences 将序列填充到相同的长度
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_REVIEW_LEN)  # 统一时间步数
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)
print(x_train.shape)  # (25000, 80)
train = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
)  # drop_remainder(丢弃余数) 删除剩余数据
test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=BATCH_SIZE, drop_remainder=True)


class SimpleRNNCell(models.Model):

    def __init__(self, units):
        super(SimpleRNNCell, self).__init__()

        self.units = units

        self.state0 = [tf.zeros([BATCH_SIZE, self.units])]
        self.state1 = [tf.zeros([BATCH_SIZE, self.units])]

        """
        Embedding 词向量编码  (25000, 80) -> (25000, 80, 100)
            Input shape: 2D tensor with shape: `(batch_size, input_length)`.
            Output shape: 3D tensor with shape: `(batch_size, input_length, output_dim)`.
        """
        self.embedding = layers.Embedding(input_dim=TOTAL_WORDS, output_dim=EMBEDDING_LEN, input_length=MAX_REVIEW_LEN)
        self.rnncell0 = layers.SimpleRNNCell(self.units, dropout=0.5)
        self.rnncell1 = layers.SimpleRNNCell(self.units, dropout=0.5)
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)

        state0 = self.state0
        state1 = self.state1

        # tf.unstack: 以指定的轴 axis, 将一个维度为 R 的张量数组转变成一个维度为 R-1 的张量 [张量版的 tf.squeeze]
        for i in tf.unstack(x, axis=1):
            # 以时间步数轴，将张量从3维降维到2维（以时间维度展开） (batch, timesteps, feature) -> (batch, feature)
            out0, state0 = self.rnncell0(i, state0, training)
            out1, state1 = self.rnncell1(out0, state1, training)

        x = self.outlayer(out1)
        return tf.sigmoid(x)


class SimpleRNN(models.Model):

    def __init__(self, units):
        super(SimpleRNN, self).__init__()

        self.units = units

        self.embedding = layers.Embedding(TOTAL_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN)
        # unroll=True, 网络将展开，否则将使用符号循环。展开可以加速 RNN，尽管它往往更占用内存。展开仅适用于短序列。
        self.rnn = models.Sequential(
            [
                layers.SimpleRNN(units=self.units, dropout=0.5, return_sequences=True, unroll=True),
                layers.SimpleRNN(units=self.units, dropout=0.5, unroll=True),
            ]
        )
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x)
        return tf.sigmoid(x)


class LSTM(models.Model):

    def __init__(self, units):
        super(LSTM, self).__init__()

        self.units = units

        self.embedding = layers.Embedding(TOTAL_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN)
        self.lstm = models.Sequential(
            [
                layers.LSTM(units=self.units, dropout=0.5, return_sequences=True, unroll=True),
                layers.LSTM(units=self.units, dropout=0.5, unroll=True),
            ]
        )
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.outlayer(x)
        return tf.sigmoid(x)


class GRU(models.Model):

    def __init__(self, units):
        super(GRU, self).__init__()

        self.units = units

        self.embedding = layers.Embedding(TOTAL_WORDS, EMBEDDING_LEN, input_length=MAX_REVIEW_LEN)
        self.gru = models.Sequential(
            [
                layers.GRU(units=self.units, dropout=0.5, return_sequences=True, unroll=True),
                layers.GRU(units=self.units, dropout=0.5, unroll=True),
            ]
        )
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        x = self.gru(x)
        x = self.outlayer(x)
        return tf.sigmoid(x)


# model = SimpleRNNCell(units=64)
# model = SimpleRNN(units=64)
# model = LSTM(units=64)
model: models.Model = GRU(units=64)

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.BinaryCrossentropy(), metrics=["accuracy"])
model.fit(train, epochs=EPOCHS, validation_data=test)
print(model.evaluate(test))
