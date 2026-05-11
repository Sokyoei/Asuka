"""
tensorflow 1.x 最后一个版本 1.15.5 支持 python3.7
"""

import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def main():
    print(f"tensorflow version: {tf.__version__}")

    # 构建计算图
    c1 = tf.constant(9.5, dtype=tf.float32)

    a = tf.Variable(5.8)
    b = tf.Variable(2.9)
    sum = tf.Variable(0, name="sum")
    result = tf.Variable(1, name="result")

    martix1 = tf.constant([[2.0, 5.0, 4.0], [1.0, 3.0, 6.0]])
    martix2 = tf.constant([[2.0, 3.0], [1.0, 2.0], [3.0, 1.0]])

    vector1 = tf.constant([3.0, 3.0])
    vector2 = tf.constant([1.0, 2.0])

    result1 = tf.multiply(vector1, vector2)
    # result2 = tf.multiply(vector2, vector1)
    print(f"result1: {result1}")  # 只能获得形状，没有执行真正地计算

    # 创建会话，运行计算图
    with tf.Session() as ses:
        # 全局变量初始化
        ses.run(tf.global_variables_initializer())

        print(f"result1: {ses.run(result1)}")
        # print(ses.run(result2))

        # 加减乘除
        print(ses.run(tf.add(a, b)))  # +
        print(ses.run(tf.subtract(a, c1)))  # -
        print(ses.run(tf.multiply(a, b)))  # *
        print(ses.run(tf.divide(a, 2.0)))  # /
        print(ses.run(tf.div(a, 2.0)))  # /

        for i in range(101):
            ses.run(tf.assign(sum, tf.add(sum, i)))
        print("1-100和:", ses.run(sum))

        for i in range(1, 11):
            ses.run(tf.assign(result, tf.multiply(result, i)))
        print("1-10积:", ses.run(result))

        print(ses.run(martix1))
        print(ses.run(tf.transpose(martix1)))  # martix1.T
        print(ses.run(tf.multiply(martix1, 2)))  # 2 * martix1
        print(ses.run(tf.matmul(martix1, martix2)))  # martix1.dot(martix2)
        print(ses.run(tf.reduce_sum(martix1)))
        print(ses.run(tf.reduce_sum(martix1, axis=1)))
        print(ses.run(tf.reduce_sum(martix1, axis=0)))

        print(ses.run(tf.argmax(martix1, axis=1)))
        print(ses.run(tf.cast(1 > 0.5, dtype=tf.float32)))  # 类型转换
        print(ses.run(tf.cast(0.1 > 0.5, dtype=tf.float32)))
        print(ses.run(tf.equal(1.0, 1)))  # 比较
        print(ses.run(tf.equal(1.0, 1.0)))
        print(ses.run(tf.equal(0, False)))
        print(ses.run(tf.reduce_mean(martix1)))
        print(ses.run(tf.reduce_mean(martix1, 0)))
        print(ses.run(tf.reduce_mean(martix1, 1, keepdims=True)))


if __name__ == '__main__':
    main()
