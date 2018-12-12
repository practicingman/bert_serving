import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model has been written.")

flags.DEFINE_string(
    "predict_file", None,
    "The file of predict records")

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.export_dir)
        tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
        tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
        tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
        tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
        tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
        record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.predict_file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            input_ids = example.features.feature['input_ids'].int64_list.value
            input_mask = example.features.feature['input_mask'].int64_list.value
            label_ids = example.features.feature['label_ids'].int64_list.value
            segment_ids = example.features.feature['segment_ids'].int64_list.value
            result = sess.run(tensor_outputs, feed_dict={
                tensor_input_ids: np.array(input_ids).reshape(-1, FLAGS.max_seq_length),
                tensor_input_mask: np.array(input_mask).reshape(-1, FLAGS.max_seq_length),
                tensor_label_ids: np.array(label_ids),
                tensor_segment_ids: np.array(segment_ids).reshape(-1, FLAGS.max_seq_length),
            })
            print(*(result[0]), sep='\t')
