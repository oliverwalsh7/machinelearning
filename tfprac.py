# ai --> ml --> neural network practice
# oliver walsh august 2020

import tensorflow as tf
print(tf.version)

# tf.reshape(x, -1): -1 will calculate tensor size
# types of tensors: variables (mutable), constants, placeholders, sparsetensors
# tensor.eval() finds tensor vals

t = tf.zeros(5, 5, 5, 5)
print(t)
