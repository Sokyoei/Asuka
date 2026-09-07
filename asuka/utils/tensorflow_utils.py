import tensorflow as tf

DEVICE = 'cuda' if tf.test.is_gpu_available() else 'cpu'
