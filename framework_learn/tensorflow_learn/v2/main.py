import os  # noqa: F401

# 启用 Keras 2, pip install tf_keras
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
import keras
import tensorflow as tf

if tf.__version__ >= "2.16.0":
    from keras import applications, callbacks, datasets, layers, losses, models, optimizers, utils
elif tf.__version__ >= "2.0.0":
    from tensorflow.keras import applications, callbacks, datasets, layers, losses, models, optimizers, utils
else:
    raise ImportError("TensorFlow version must be >= 2.0.0")

import matplotlib.pyplot as plt

from Ahri.Asuka.config.config import settings

# 超参数
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.0005
NUM_CLASSES = 10

tf.random.set_seed(1)

# load data and preprocess
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(x_train.shape)  # NHWC
# x_train = tf.cast(x_train, dtype=tf.float32) / 255.0  # 归一化
# x_test = tf.cast(x_test, dtype=tf.float32) / 255.0
x_train = applications.mobilenet.preprocess_input(x_train)
x_test = applications.mobilenet.preprocess_input(x_test)
y_train = utils.to_categorical(y_train, NUM_CLASSES)  # one hot
y_test = utils.to_categorical(y_test, NUM_CLASSES)

base_model: models.Model = applications.MobileNetV2(
    include_top=False,  # 去掉原来的 1000 分类的全连接层
    weights="imagenet",  # 使用 imagenet 预训练的权重
    input_shape=(32, 32, 3),
    pooling="avg",
)
base_model.trainable = False  # 冻结权重，仅训练最后一层分类的全连接层
# 1. Squential API
model = models.Sequential(
    [
        base_model,
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
# 2. Keras 函数式 API
# head = layers.Dense(256, activation="relu")(base_model.output)
# head = layers.Dropout(0.5)(head)
# head = layers.Dense(NUM_CLASSES, activation="softmax")(head)
# model = models.Model(inputs=base_model.input, outputs=head)

model.compile(optimizer=optimizers.Adam(learning_rate=LR), loss=losses.categorical_crossentropy, metrics=["accuracy"])
tensorboard_callback = callbacks.TensorBoard(log_dir=settings.LOG_DIR / "tensorboard", histogram_freq=1)
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("finally_test_accuracy:", test_accuracy)

# save model and weights
if tf.__version__ >= "2.16.0":
    model.save(settings.MODELS_DIR / "MobileNetV2.keras")
    model.save_weights(settings.MODELS_DIR / "MobileNetV2.weights.h5")  # or .weights.json
else:
    model.save(settings.MODELS_DIR / "MobileNetV2.h5")  # or .savemodel
    model.save_weights(settings.MODELS_DIR / "MobileNetV2.weight.h5")

# load model and weights
loaded_model: models.Model = keras.saving.load_model(settings.MODELS_DIR / "MobileNetV2.keras")
loaded_weights = keras.saving.load_weights(model, settings.MODELS_DIR / "MobileNetV2.weights.h5")

# model summary
loaded_model.summary()

# plot loss
loss = history.history["loss"]
plt.plot(loss)
plt.show()
