import tensorflow as tf
import numpy as np

from net.const import feature_dim
from const import encoder_add_dim, max_decoderlen, max_encoderlen, decoder_SOT, decoder_EOT

encoder_dim = feature_dim + encoder_add_dim

tfdata_path = 'train_data2'

npz_file = np.load('charparam.npz')
features = []
feature_idx = []
idx = 0
features.append(np.zeros([1,feature_dim], np.float32))
feature_idx.append([0,0,0,1])
idx = 1
for varname in npz_file.files:
    features.append(npz_file[varname])
    feature_idx.append([int(varname[:-1]),0 if varname[-1] == 'n' else 1,idx,idx+npz_file[varname].shape[0]])
    idx += npz_file[varname].shape[0]
rng = np.random.default_rng()
del npz_file

with tf.device('cpu'):
    features = tf.concat(features, axis=0)
    feature_idx = tf.constant(feature_idx, tf.int64)

def parse(serialized):
    return tf.io.parse_example(serialized, features={ 
        "str": tf.io.FixedLenFeature([], dtype=tf.string), 
        "code": tf.io.FixedLenFeature([], dtype=tf.string),
        "codelen": tf.io.FixedLenFeature([], dtype=tf.int64),
        "strlen": tf.io.FixedLenFeature([], dtype=tf.int64),
    })

def generate_feature(code_vec):
    @tf.function
    def subfun_n(v):
        sample = tf.zeros([feature_dim])
        v0 = tf.cast(v[0], tf.int64)
        if v0 > 0:
            idx = tf.where(tf.logical_and(feature_idx[:,0] == v0,feature_idx[:,1] == 0))
            if tf.size(idx) > 0:
                idx = tf.squeeze(idx)
                st = feature_idx[idx,2]
                ed = feature_idx[idx,3]
            else:
                st = tf.constant(0, tf.int64)
                ed = tf.constant(1, tf.int64)
            index = tf.random.uniform([], minval=st, maxval=ed, dtype=tf.int64)
            sample = features[index,:]
        return tf.concat([sample, tf.cast(v[1:], tf.float32)], axis=0)

    @tf.function
    def subfun_t(v):
        sample = tf.zeros([feature_dim])
        v0 = tf.cast(v[0], tf.int64)
        if v0 > 0:
            idx = tf.where(tf.logical_and(feature_idx[:,0] == v0,feature_idx[:,1] == 1))
            if tf.size(idx) > 0:
                idx = tf.squeeze(idx)
                st = feature_idx[idx,2]
                ed = feature_idx[idx,3]
            else:
                idx2 = tf.where(tf.logical_and(feature_idx[:,0] == v0,feature_idx[:,1] == 0))
                if tf.size(idx2) > 0:
                    idx2 = tf.squeeze(idx2)
                    st = feature_idx[idx2,2]
                    ed = feature_idx[idx2,3]
                else:
                    st = tf.constant(0, tf.int64)
                    ed = tf.constant(1, tf.int64)
            index = tf.random.uniform([], minval=st, maxval=ed, dtype=tf.int64)
            sample = features[index,:]
        return tf.concat([sample, tf.cast(v[1:], tf.float32)], axis=0)

    if tf.random.uniform([]) < 0.25:
        return tf.map_fn(subfun_t, code_vec, fn_output_signature=tf.float32)
    else:
        return tf.map_fn(subfun_n, code_vec, fn_output_signature=tf.float32)

def process_data(data):
    batch = 8
    str_data = data['str']
    code = data['code']
    strlen_data = data['strlen']
    codelen_data = data['codelen']
    max_len = tf.random.uniform([], 1, max_encoderlen, dtype=tf.int64)
    pad_ln = tf.random.uniform([batch])

    result_strlen = tf.constant(0, dtype=tf.int64)
    result_codelen = tf.constant(0, dtype=tf.int64)
    j = tf.constant(0, tf.int64)
    while j < batch:
        if result_strlen + strlen_data[j] < max_decoderlen - 2 and result_codelen < max_len and result_codelen + codelen_data[j] < max_encoderlen:
            result_strlen += strlen_data[j]
            result_codelen += codelen_data[j]
            if pad_ln[j] < 0.1:
                result_strlen += 1
                result_codelen += 1
        else:
            break
        j += 1
    if j == 0:
        j = tf.constant(1, tf.int64)
        
    def loop1(result,i):
        result = tf.concat([result, tf.io.parse_tensor(code[i], tf.int32)], axis=0)
        if pad_ln[i] < 0.1:
            result = tf.concat([result, tf.constant([[0,0,0,0,1]], tf.int32)], axis=0)
        return result, i+1

    result_code,_ = tf.while_loop(lambda r,i: i < j, loop1, 
                                  loop_vars=[tf.zeros([0,5], tf.int32), tf.constant(0,tf.int64)],
                                  shape_invariants=[tf.TensorShape([None,5]),tf.TensorShape([])])

    def loop2(result,i):
        result = tf.strings.join([result, str_data[i]])
        if pad_ln[i] < 0.1:
            result = tf.strings.join([result, tf.constant("\n")])
        return result, i+1

    result_str,_ = tf.while_loop(lambda r,i: i < j, loop2, loop_vars=[tf.constant(""), tf.constant(0,tf.int64)])

    if tf.random.uniform([]) < 0.5:
        result_str = tf.strings.substr(result_str, 0, tf.strings.length(result_str)-1)
        result_code = result_code[:-1,:]

    if tf.shape(result_code, out_type=tf.int64)[0] > max_encoderlen:
        return {
            'text': tf.constant(""),
            'decoder_true': tf.zeros([max_decoderlen], dtype=tf.int32),
            'decoder_task': tf.zeros([max_decoderlen], dtype=tf.int32),
            'encoder_inputs': tf.zeros([max_encoderlen, encoder_dim]),
        }

    decoder_code = tf.strings.unicode_decode(result_str, input_encoding='UTF-8')
    if tf.shape(decoder_code, out_type=tf.int64)[0] > max_decoderlen - 2:
        return {
            'text': tf.constant(""),
            'decoder_true': tf.zeros([max_decoderlen], dtype=tf.int32),
            'decoder_task': tf.zeros([max_decoderlen], dtype=tf.int32),
            'encoder_inputs': tf.zeros([max_encoderlen, encoder_dim]),
        }

    decoder_code = tf.concat([
        tf.cast([decoder_SOT], dtype=tf.int32),
        decoder_code,
        tf.cast([decoder_EOT], dtype=tf.int32),
    ], axis=0)
    encoder_input = generate_feature(result_code)
    decoder_len = tf.shape(decoder_code)[0]
    encoder_len = tf.shape(encoder_input)[0]

    decoder_true = decoder_code[1:]
    decoder_task = decoder_code[:-1]
    decoder_len = tf.shape(decoder_true)[0]

    decoder_true = tf.pad(decoder_true, [[0, max_decoderlen - decoder_len]])
    decoder_task = tf.pad(decoder_task, [[0, max_decoderlen - decoder_len]])
    encoder_inputs = tf.pad(encoder_input, [[0, max_encoderlen - encoder_len], [0, 0]])

    return {
        'text': result_str,
        'decoder_true': decoder_true,
        'decoder_task': decoder_task,
        'encoder_inputs': encoder_inputs,
    }


def create_dataset(batch_size, filelist):
    fs = tf.data.Dataset.from_tensor_slices(filelist)
    fs = fs.shuffle(len(filelist), reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(filenames=fs)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.batch(8, drop_remainder=True)
    ds = ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.shuffle(10000)
    ds = ds.repeat()
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
    for d in generate_data().take(10):
        print([b.decode() for b in d['text'].numpy()])
