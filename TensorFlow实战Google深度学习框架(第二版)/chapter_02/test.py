#!/usr/bin/python3 
# encoding: utf-8

# 这个是为了使用x86的avx扩展
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import加载tensorflow并查看版本
import tensorflow as tf
print('the version of tf:', tf.__version__)

# 定义两个常量向量
a = tf.constant([1.0,2.0], name="a")
b = tf.constant([2.0,3.0], name="b")

# 定义操作
result = a + b

# 传建一个会话来管理操作的运行
sess = tf.Session()
print("a + b = ", sess.run(result))
