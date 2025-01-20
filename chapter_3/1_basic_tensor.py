import tensorflow as tf

def main():
    # Rank 0 (0-D Tensor)
    a = tf.constant(2)
    print(tf.rank(a))

    # Rank 1 (1-D Tensor)
    b = tf.constant([1, 2])
    print(tf.rank(b))

    # Rank 2 (2-D tensor)
    c = tf.constant([[1, 2], [3, 4]])
    print(tf.rank(c))

    # Rank n (n-D tensor), 3D in this case
    d = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print(tf.rank(d))

if __name__ == "__main__":
    main()
