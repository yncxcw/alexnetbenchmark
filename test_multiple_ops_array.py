import tensorflow as tf
import time

from tensorflow.python.client import timeline
"""
Performance benchmark for a certain number of matrix multiply
# of gourps       time
1                 0.002694  
4                 0.00596
16                0.07386
64                0.27440
256               1.3156
1024              5.15
2048              10.8772
10240             90.4594 
"""


def matrix_mul(image_size):
    """
    Return a tensor (N x N), the result of random matrix mul
    """
    with tf.name_scope("var-1"):
        matrix1 = tf.Variable(tf.random_normal([ 
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m1")
    with tf.name_scope("var-2"):
        matrix2 = tf.Variable(tf.random_normal([
                                            image_size, 
                                            image_size], 
                                          dtype=tf.float32,
                                          stddev=1e-1), name="m2")

    with tf.name_scope("mul"):
        result = tf.matmul(matrix1, matrix2)
    return result



if __name__ == "__main__":
    batch_run  = 4
    image_size = 32
    count = 0
    results = []
    ##define the graph
    while count < batch_run:    
        results.append(matrix_mul(image_size))
        count = count + 1
    # build option for perf
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # build tensorboard 
    tf_writer = tf.summary.FileWriter("./tensorboard_multi_ops_group", graph=tf.get_default_graph())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
    init = tf.global_variables_initializer()
    sess.run(init)
    start = time.time()
    final_result = sess.run(tf.group(*results), options = options, run_metadata = run_metadata)
    end  =  time.time()
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_02_step_%d.json', 'w') as f:
          f.write(chrome_trace)
    print("benchmark time: {}".format(end - start))
