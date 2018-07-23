import tensorflow as tf
import time

if __name__ == "__main__":
    batch_size = 128
    image_size = 5120
    count = 0
    nums_run = 1000
    device = '/device:CPU:0'
    with tf.device(device):
        matrix1 = tf.Variable(tf.random_normal([ 
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m1")
        matrix2 = tf.Variable(tf.random_normal([
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m2")

        result = tf.matmul(matrix1, matrix2)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
    init = tf.global_variables_initializer()
    sess.run(init)
    start = time.time()
    while count < nums_run:    
        result_op = sess.run(result)
        count = count + 1 
    end = time.time()
    print("benchmark time: {}".format(end - start))
