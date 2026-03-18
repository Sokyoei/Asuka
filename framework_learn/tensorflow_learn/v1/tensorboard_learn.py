"""
tensorboard --logdir=your_tensorboard_log_dir --host=127.0.0.1
"""

import tensorflow as tf

from Ahri.Asuka.config.config import settings

with tf.name_scope("a"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("b"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
add = tf.add_n([input1, input2], name="add_n")

with tf.Session() as ses:
    ses.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(settings.LOG_DIR / "tensorboard", ses.graph)
    print(ses.run(add))
    writer.close()
