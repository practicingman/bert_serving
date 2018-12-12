## export bert model for serving ##

>
predicting with estimator is slow, use export_savedmodel instead

create virtual environment</br>
```bash
conda env create -f env.yml
```

train a classifier</br>
```bash
bash train.sh
```

use the classifier</br>
```bash
bash predict.sh
```

export bert model</br>
```bash
bash export.sh
```
check out exported model</br>
```bash
saved_model_cli show --all --dir $exported_dir
```

test exported model</br>
```bash
bash test.sh
```

export it yourself</br>

```python
def serving_input_fn():
    label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn
```

and</br>

```python
estimator._export_to_tpu = False
estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
```
