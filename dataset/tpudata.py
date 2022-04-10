import tensorflow as tf

def make_example(images, labels, ids):
    images = images.numpy().ravel().tolist()
    labels = labels.numpy().ravel().tolist()
    ids = ids.numpy().ravel().tolist()
    
    return tf.train.Example(features=tf.train.Features(feature={
        'images' : tf.train.Feature(float_list=tf.train.FloatList(value=images)),
        'labels' : tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
        'ids' : tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
    }))

def make_tfrecodes(data, savepath, filecount, batchcount, batchsize, train=True):
    if train:
        print('make train data')
        ds = data.train_data(batchsize)
        ftempl = 'train%05d.tfrecords'
    else:
        print('make test data')
        ds = data.test_data(batchsize)
        ftempl = 'test%05d.tfrecords'
    
    tf.io.gfile.makedirs(savepath)
    for i in range(filecount):
        filename = tf.io.gfile.join(savepath, ftempl%i)
        print(filename)
        with tf.io.TFRecordWriter(filename,'GZIP') as writer:
            for j, d in enumerate(ds.take(batchcount)):
                print(j, '/', batchcount)
                images, labels, ids = d
                for b in range(batchsize):
                    ex = make_example(images[b,...],labels[b,...],ids[b,...])
                    writer.write(ex.SerializeToString())

def parse_batch_example(example): 
    from .data import height, width, scale

    features = tf.io.parse_example(example, features={ 
        'images' : tf.io.FixedLenFeature([height * width * 3], tf.float32),
        'labels' : tf.io.FixedLenFeature([(height // scale) * (width // scale) * 7], tf.float32),
        'ids' : tf.io.FixedLenFeature([(height // scale) * (width // scale)], tf.int64),
    }) 
    return (
        tf.reshape(features['images'], [-1, height, width, 3]), 
        tf.reshape(features['labels'], [-1, height // scale, width // scale, 7]), 
        tf.cast(tf.reshape(features['ids'], [-1, height // scale, width // scale]), tf.int32),
    )

def train_data(savepath, batchsize):
    files = tf.io.gfile.glob(tf.io.gfile.join(savepath, 'train*.tfrecords'))
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=batchsize)
    ds = ds.shuffle(1000)
    ds = ds.repeat()
    ds = ds.batch(batchsize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(parse_batch_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def test_data(savepath, batchsize):
    files = tf.io.gfile.glob(tf.io.gfile.join(savepath, 'test*.tfrecords'))
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=batchsize)
    ds = ds.shuffle(1000)
    ds = ds.repeat()
    ds = ds.batch(batchsize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(parse_batch_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
