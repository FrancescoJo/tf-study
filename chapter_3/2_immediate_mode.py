import tensorflow as tf
import numpy as np

def main():
    a = tf.constant(3)
    b = tf.constant(2)

    # Basic calculations
    print(tf.add(a, b))
    print(tf.subtract(a, b))

    print(tf.multiply(a, b).numpy())
    print(tf.divide(a, b).numpy())

if __name__ == "__main__":
    main()
