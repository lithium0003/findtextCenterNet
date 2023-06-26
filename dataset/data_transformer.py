import tensorflow as tf

from net.const import feature_dim
from const import encoder_add_dim, max_decoderlen, max_encoderlen, decoder_SOT, decoder_EOT
from const import samples_per_file

encoder_dim = feature_dim + encoder_add_dim

tfdata_path = 'train_data2'

def deserialize_composite(serialized, type_spec):
    serialized = tf.io.parse_tensor(serialized, tf.string)
    component_specs = tf.nest.flatten(type_spec, expand_composites=True)
    components = [
        tf.io.parse_tensor(serialized[i], spec.dtype)
        for i, spec in enumerate(component_specs)
    ]
    return tf.nest.pack_sequence_as(type_spec, components, expand_composites=True)

def parse(serialized):
    return tf.io.parse_example(serialized, features={ 
        "strcode": tf.io.FixedLenFeature([], dtype=tf.string), 
        "features": tf.io.FixedLenFeature([], dtype=tf.string),
        "length": tf.io.FixedLenFeature([], dtype=tf.int64),
    })

def deserialize_data(data):
    rt_spec1 = tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32)
    rt_spec2 = tf.RaggedTensorSpec(shape=[None, None, encoder_dim], dtype=tf.float32, ragged_rank=1)
    deserialized1 = deserialize_composite(data['strcode'], rt_spec1)
    deserialized2 = deserialize_composite(data['features'], rt_spec2)
    return {
        'strcode': deserialized1,
        'features': deserialized2,
        'length': tf.cast(data['length'], tf.int32),
    }

def trim_data(data):
    want_declen = tf.random.uniform([], 1, max_decoderlen - 1, dtype=tf.int32)
    strcode = data['strcode']
    features = data['features']
    f = tf.zeros([0,encoder_dim])
    s = tf.zeros([0,], dtype=tf.int32)
    f1 = f
    s1 = s
    if data['length'] > 2:
        st = tf.random.uniform([], 0, data['length'] - 1, dtype=tf.int32)
    else:
        st = 0
    for i in tf.range(st, data['length']):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[
                (f, tf.TensorShape([None,encoder_dim])),
                (f1, tf.TensorShape([None,encoder_dim])),
                (s, tf.TensorShape([None,])),
                (s1, tf.TensorShape([None,])),
            ]
        )
        s1 = tf.concat([s, strcode[i]], axis=0)
        f1 = tf.concat([f, features[i]], axis=0)
        if tf.shape(s1)[0] < max_decoderlen - 1 and tf.shape(f1)[0] < max_encoderlen:
            s = s1
            f = f1
            if tf.shape(s)[0] >= want_declen:
                break
        else:
            break
    return {
        'strcode': s,
        'features': f,
    }

def encode(data):
    strcode = data['strcode']
    features = data['features']

    decoder_len = tf.shape(strcode)[0]
    encoder_len = tf.shape(features)[0]

    true_str = tf.pad(strcode, [[0, max_decoderlen - decoder_len]])

    strcode = tf.concat([
        tf.cast([decoder_SOT], dtype=tf.int32),
        strcode,
        tf.cast([decoder_EOT], dtype=tf.int32),
    ], axis=0)


    decoder_true = strcode[1:]
    decoder_task = strcode[:-1]
    decoder_true = tf.pad(decoder_true, [[0, max_decoderlen - decoder_len]])
    decoder_task = tf.pad(decoder_task, [[0, max_decoderlen - decoder_len]])

    encoder_inputs = tf.pad(features,[[0, max_encoderlen - encoder_len], [0, 0]])

    return {
        'text': true_str,
        'decoder_true': decoder_true,
        'decoder_task': decoder_task,
        'encoder_inputs': encoder_inputs,
    }

def create_dataset(batch_size, filelist):
    files = tf.data.Dataset.from_tensor_slices(filelist)
    files = files.shuffle(len(filelist))
    ds = files.interleave(lambda x: tf.data.TFRecordDataset(x, 'ZLIB'),
                            num_parallel_calls=tf.data.AUTOTUNE,
                            deterministic=False)
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(filelist)*samples_per_file))
    ds = ds.shuffle(1000)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(deserialize_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(trim_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(encode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_data(batch_size, data_path=''):
    train_files = tf.io.gfile.glob(tf.io.gfile.join(data_path,'train_data2','train*.tfrecords'))
    return create_dataset(batch_size, train_files)

def test_data(batch_size, data_path=''):
    test_files = tf.io.gfile.glob(tf.io.gfile.join(data_path,'train_data2','test*.tfrecords'))
    return create_dataset(batch_size, test_files)

def generate_data(data_path=''):
    test_files = tf.io.gfile.glob(tf.io.gfile.join(data_path,'train_data2','test*.tfrecords'))
    return create_dataset(1, test_files)

if __name__=='__main__':
    for d in test_data(4):
        print(d)
