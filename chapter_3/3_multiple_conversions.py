import tensorflow as tf
import numpy as np

def main():
    a = tf.constant(3)
    b = tf.constant(2)

    c = tf.add(a, b).numpy()
    c_square = np.square(c, dtype = np.float32)
    c_tensor = tf.convert_to_tensor(c_square)

    print("Numpy array: %0.1f, np.square: %0.1f, np.square->tensor: %0.1f" % (c, c_square, c_tensor))

if __name__ == "__main__":
    main()
