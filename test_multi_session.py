import tensorflow as tf
import threading

def matrix_mul(image_size, graph):
    """
    Return a tensor (N x N), the result of random matrix mul
    """
    with tf.name_scope(graph+"var-1"):
        matrix1 = tf.Variable(tf.random_normal([ 
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m1")
    with tf.name_scope(graph+"var-2"):
        matrix2 = tf.Variable(tf.random_normal([
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m2")

    with tf.name_scope(graph+"mul"):
        result = tf.matmul(matrix1, matrix2)
    return result



def session_thread(graph_name):
    print("entern", graph_name)
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        g1_v = matrix_mul(256, graph_name)
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            sess.run(g1_v)
    print("finish", graph_name)



threading.Thread(target=session_thread, args=("g1",)).start()
threading.Thread(target=session_thread, args=("g2",)).start()
