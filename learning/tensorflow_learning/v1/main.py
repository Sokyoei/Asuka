import random

import tensorflow as tf
import tensorflow.contrib as tc
from loguru import logger
from tensorflow.examples.tutorials.mnist import input_data

from asuka.config import settings

# 超参数
LR = 0.001
EPOCHS = 15
BATCH_SIZE = 100

tf.set_random_seed(1)

mnist_dataset = input_data.read_data_sets(str(settings.DATA_DIR), one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

X_image = tf.reshape(X, [-1, 28, 28, 1])
tf.summary.image("input", X_image, 3)
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("layer1") as layer1:
    w1 = tf.get_variable("w1", shape=[784, 512], initializer=tc.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]))
    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
    tf.summary.histogram("X", X)
    tf.summary.histogram("weight", w1)
    tf.summary.histogram("bias", b1)
    tf.summary.histogram("l1", l1)
with tf.variable_scope("layer2") as layer2:
    w2 = tf.get_variable("w2", shape=[512, 512], initializer=tc.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]))
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)
    tf.summary.histogram("weight", w2)
    tf.summary.histogram("bias", b2)
    tf.summary.histogram("layer", l2)
with tf.variable_scope("layer3") as layer3:
    w3 = tf.get_variable("w3", shape=[512, 512], initializer=tc.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]))
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
    tf.summary.histogram("weight", w3)
    tf.summary.histogram("bias", b3)
    tf.summary.histogram("layer", l3)
with tf.variable_scope("layer4") as layer4:
    w4 = tf.get_variable("w4", shape=[512, 512], initializer=tc.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]))
    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, keep_prob=keep_prob)
    tf.summary.histogram("weight", w4)
    tf.summary.histogram("bias", b4)
    tf.summary.histogram("layer", l4)
with tf.variable_scope("layer5") as layer5:
    w5 = tf.get_variable("w5", shape=[512, 10], initializer=tc.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(l4, w5) + b5
    tf.summary.histogram("weight", w5)
    tf.summary.histogram("bias", b5)
    tf.summary.histogram("hypothesis", hypothesis)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
tf.summary.scalar("loss", loss)
summary = tf.summary.merge_all()  # 合并默认图表中收集的所有摘要

with tf.Session() as ses:
    ses.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(settings.LOG_DIR / "tensorboard")
    writer.add_graph(ses.graph)  # 向事件文件添加一个 Graph

    global_step = 0

    for epoch in range(EPOCHS):
        avg_loss = 0
        total_batch = int(mnist_dataset.train.num_examples / BATCH_SIZE)
        for _i in range(total_batch):
            batch_x, batch_y = mnist_dataset.train.next_batch(BATCH_SIZE)
            s, loss_value, _ = ses.run([summary, loss, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
            writer.add_summary(s, global_step=global_step)
            global_step += 1
            avg_loss += loss_value / total_batch
        logger.info(f"epoch:{(epoch + 1):04d}, loss_value:{avg_loss:.9f}")

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    logger.info(
        "accuracy_value:",
        ses.run(accuracy, feed_dict={X: mnist_dataset.test.images, Y: mnist_dataset.test.labels, keep_prob: 1}),
    )

    r = random.randint(0, mnist_dataset.test.num_examples - 1)
    logger.info("label:", ses.run(tf.argmax(mnist_dataset.test.labels[r : r + 1], 1)))
    logger.info(
        "prediction:",
        ses.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist_dataset.test.images[r : r + 1], keep_prob: 1}),
    )
