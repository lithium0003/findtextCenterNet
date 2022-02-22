#!/usr/bin/env python3
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

save_target = 'result/'
batchsize = 8

import os
import tqdm
import time

import net
import dataset

def is_alive():
    home = os.environ['HOME']
    jid = os.environ.get('JOB_ID', None)
    if jid:
        return os.path.exists('%s/run.%s'%(home,jid))
    else:
        return True

def make_run_lock(step):
    home = os.environ['HOME']
    jid = os.environ.get('JOB_ID', '0')
    with open('%s/lock.%s'%(home,jid), 'w') as f:
        f.write('%d'%step)

def rm_run_lock():
    home = os.environ['HOME']
    jid = os.environ.get('JOB_ID', '0')
    os.remove('%s/lock.%s'%(home,jid))

def wait_run_lock(step):
    home = os.environ['HOME']
    jid = os.environ.get('JOB_ID', '0')
    path = '%s/lock.%s'%(home,jid)
    run_step = -1
    while run_step != step:
        while not os.path.exists(path):
            time.sleep(5)

        try:
            with open(path) as f:
                run_step = int(f.read())
        except:
            continue

        if run_step != step:
            time.sleep(5)

def calc_predid(*args):
    m = net.modulo_list
    b = args
    assert(len(m) == len(b))
    t = []

    for k in range(len(m)):
        u = 0
        for j in range(k):
            w = t[j]
            for i in range(j):
                w *= m[i]
            u += w
        tk = b[k] - u
        for j in range(k):
            tk *= pow(m[j], m[k]-2, m[k])
            #tk *= pow(m[j], -1, m[k])
        tk = tk % m[k]
        t.append(tk)
    x = 0
    for k in range(len(t)):
        w = t[k]
        for i in range(k):
            w *= m[i]
        x += w
    mk = 1
    for k in range(len(m)):
        mk *= m[k]
    x = x % mk
    return x

def heatmap_loss(true, logits):
    alpha = 2
    beta = 4
    pos_th = 1.0

    predict = tf.math.sigmoid(logits)

    pos_mask = tf.cast(true >= pos_th, dtype=tf.float32)
    neg_mask = tf.cast(true < pos_th, dtype=tf.float32)

    neg_weights = tf.math.pow(1. - true, beta)

    pos_loss = tf.math.softplus(-logits) * tf.math.pow(1 - predict, alpha) * pos_mask
    neg_loss = (logits + tf.math.softplus(-logits)) * tf.math.pow(predict, alpha) * neg_weights * neg_mask

    loss = tf.math.reduce_sum(pos_loss) + tf.math.reduce_sum(neg_loss)
    loss = loss / tf.maximum(tf.reduce_sum(pos_mask), 1.0)

    return loss

class TextDetector2:
    def __init__(self, pretrain=None, step=1, lr=1e-4):
        self.step = step
        self.save_dir = save_target+'step%d/'%step
        checkpoint_dir = self.save_dir

        self.max_epoch = 4000
        self.steps_per_epoch = 1000
        self.batch_size = batchsize

        self.key_th1 = 0.8

        self.dtcloss_mean = tf.keras.metrics.Mean()
        self.decloss_mean = tf.keras.metrics.Mean()
        self.keymap_loss_mean = tf.keras.metrics.Mean()
        self.keymap0_loss_mean = tf.keras.metrics.Mean()
        self.size_loss_mean = tf.keras.metrics.Mean()
        self.offset_loss_mean = tf.keras.metrics.Mean()
        self.textline_loss_mean = tf.keras.metrics.Mean()
        self.separator_loss_mean = tf.keras.metrics.Mean()
        self.id_loss_mean = tf.keras.metrics.Mean()
        self.id_acc = tf.keras.metrics.Accuracy()

        self.detector = net.CenterNetDetectionBlock()
        self.decoder = net.SimpleDecoderBlock()
        opt = tf.keras.optimizers.Adam(lr, epsilon=1e-4)
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(opt)

        self.detector.summary()
        self.decoder.summary()

        self.trainable_variables = self.detector.trainable_variables
        self.trainable_variables += self.decoder.trainable_variables

        self.mu_loss = {}
        self.mu_lt = {}
        self.M_lt = {}
        self.mu_loss2 = {}
        self.mu_lt2 = {}
        self.M_lt2 = {}
        for key in ['keymap','size','offset','textline','separator','id']:
            self.mu_loss[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.mu_lt[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.M_lt[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.mu_loss2[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.mu_lt2[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.M_lt2[key] = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)

        checkpoint = tf.train.Checkpoint(
                            optimizer=self.optimizer,
                            decoder=self.decoder,
                            detector=self.detector,
                            mu_loss=self.mu_loss,
                            mu_lt=self.mu_lt,
                            M_lt=self.M_lt,
                            mu_loss2=self.mu_loss2,
                            mu_lt2=self.mu_lt2,
                            M_lt2=self.M_lt2)
        last = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(last)
        self.manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_dir, max_to_keep=2)
        if not last is None:
            self.init_epoch = int(os.path.basename(last).split('-')[1])
            print('loaded %d epoch'%self.init_epoch)
        else:
            self.init_epoch = 0

            if pretrain:
                last = tf.train.latest_checkpoint(pretrain)
                checkpoint0 = tf.train.Checkpoint(
                                    decoder=self.decoder,
                                    detector=self.detector)
                checkpoint0.restore(last).expect_partial()
                if last is not None:
                    print('pretrain loaded from',last)

        self.trainstep = self.init_epoch * self.steps_per_epoch
        self.teststep = self.init_epoch * self.steps_per_epoch

        if self.init_epoch < self.max_epoch:
            self.summary_writer_train = tf.summary.create_file_writer(
                    os.path.join(self.save_dir, "train")) if hvd.rank() == 0 else None
            self.summary_writer_test = tf.summary.create_file_writer(
                    os.path.join(self.save_dir, "test")) if hvd.rank() == 0 else None

    def CoV_Weight(self, training=False, **kwarg):
        t = 100
        a = {}
        eps = 1e-7
        if training:
            for key in kwarg:
                self.mu_loss[key].assign((1 - 1/t) * self.mu_loss[key] + 1/t * kwarg[key])
                l_t = kwarg[key] / (self.mu_loss[key] + eps)
                mu_lt_1 = self.mu_lt[key]
                self.mu_lt[key].assign((1 - 1/t) * mu_lt_1 + 1/t * l_t)
                self.M_lt[key].assign((1 - 1/t) * self.M_lt[key] + 1/t * (l_t - mu_lt_1) * (l_t - self.mu_lt[key]))
                sigma_lt = tf.sqrt(self.M_lt[key])
                mu_lt = self.mu_lt[key]
                a[key] = sigma_lt / (mu_lt + eps)
        else:
            for key in kwarg:
                self.mu_loss2[key].assign((1 - 1/t) * self.mu_loss2[key] + 1/t * kwarg[key])
                l_t = kwarg[key] / (self.mu_loss2[key] + eps)
                mu_lt_1 = self.mu_lt2[key]
                self.mu_lt2[key].assign((1 - 1/t) * mu_lt_1 + 1/t * l_t)
                self.M_lt2[key].assign((1 - 1/t) * self.M_lt2[key] + 1/t * (l_t - mu_lt_1) * (l_t - self.mu_lt2[key]))
                sigma_lt = tf.sqrt(self.M_lt2[key])
                mu_lt = self.mu_lt2[key]
                a[key] = sigma_lt / (mu_lt + eps)

        z = 0.
        for key in kwarg:
            z = z + a[key]
        
        for key in kwarg:
            a[key] = a[key] / (z + eps)

        return a

    def loss_func(self, labels, target_id, maps, decoder_outputs, mask1, training=False):
        mask3 = tf.boolean_mask(labels[...,0], mask1) > 0.99

        keymap_loss = heatmap_loss(true=labels[...,0], logits=maps[...,0])

        xsize_loss = tf.keras.losses.huber(
            tf.boolean_mask(labels[...,1], mask1),
            tf.boolean_mask(maps[...,1], mask1))
        ysize_loss = tf.keras.losses.huber(
            tf.boolean_mask(labels[...,2], mask1),
            tf.boolean_mask(maps[...,2], mask1))
        size_loss = xsize_loss + ysize_loss
        size_loss = tf.math.reduce_mean(size_loss)

        xoffset_loss = tf.keras.losses.huber(
            tf.boolean_mask(labels[...,3], mask1),
            tf.boolean_mask(maps[...,3], mask1))
        yoffset_loss = tf.keras.losses.huber(
            tf.boolean_mask(labels[...,4], mask1),
            tf.boolean_mask(maps[...,4], mask1))
        offset_loss = xoffset_loss + yoffset_loss
        offset_loss = tf.math.reduce_mean(offset_loss)

        textline_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels[...,5], maps[...,5])
        textline_loss = tf.math.reduce_mean(textline_loss)

        separator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels[...,6], maps[...,6])
        separator_loss = tf.math.reduce_mean(separator_loss)

        target_ids = []
        for modulo in net.modulo_list:
            target_id1 = tf.boolean_mask(target_id % modulo, mask1)
            target_ids.append(target_id1)

        id_loss = 0.
        for target_id1, decoder_id1 in zip(target_ids, decoder_outputs):
            id1_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target_id1, decoder_id1)
            id_loss += tf.math.reduce_mean(id1_loss)

        pred_ids = []
        for decoder_id1 in decoder_outputs:
            prob_id1 = tf.nn.softmax(decoder_id1, -1)
            pred_id1 = tf.math.argmax(prob_id1, axis=-1)
            pred_ids.append(pred_id1)

        pred_id = calc_predid(*pred_ids)
        pred_id = tf.boolean_mask(pred_id, mask3)
        target_id = tf.boolean_mask(target_id, mask1)
        target_id = tf.boolean_mask(target_id, mask3)

        keymap_loss *= 1e2
        offset_loss *= 10
        textline_loss *= 1e2
        separator_loss *= 1e3
        size_loss *= 1e3
        
        self.keymap_loss_mean.update_state(keymap_loss)
        self.keymap0_loss_mean.update_state(keymap_loss)
        self.size_loss_mean.update_state(size_loss)
        self.offset_loss_mean.update_state(offset_loss)
        self.textline_loss_mean.update_state(textline_loss)
        self.separator_loss_mean.update_state(separator_loss)
        self.id_loss_mean.update_state(id_loss)
        self.id_acc.update_state(target_id, pred_id)
        self.decloss_mean.update_state(id_loss)

        alpha = self.CoV_Weight(training=training,
            keymap=keymap_loss, size=size_loss, offset=offset_loss, 
            textline=textline_loss, separator=separator_loss,
            id=id_loss)
        
        keymap_loss *= alpha['keymap']
        size_loss *= alpha['size']
        offset_loss *= alpha['offset']
        textline_loss *= alpha['textline']
        separator_loss *= alpha['separator']
        id_loss *= alpha['id']

        loss = keymap_loss + size_loss + offset_loss + textline_loss + separator_loss + id_loss

        self.dtcloss_mean.update_state(loss)

        return loss

    @tf.function
    def train_step(self, iterator):
        step0 = self.optimizer.iterations == 0

        grads = self.train_substep(next(iterator))
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        if step0:
            hvd.broadcast_variables(self.trainable_variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0) 

    @tf.function
    def train_substep(self, inputs):
        image, labels, ids = inputs
        with tf.GradientTape() as tape:
            maps, feature = self.detector(image, training=True)
            mask1 = labels[...,0] > self.key_th1
            random_mask = tf.random.uniform(tf.shape(ids)) < 0.1
            mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
            feature = tf.boolean_mask(feature, mask1)
            decoder_outputs = self.decoder(feature, training=True)
            loss = self.loss_func(labels, ids, maps, decoder_outputs, mask1, training=True)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        scaled_grad = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grad)

        return grads

    @tf.function
    def test_step(self, iterator):
        self.test_substep(next(iterator))

    @tf.function
    def test_substep(self, inputs):
        image, labels, ids = inputs
        maps, feature = self.detector(image)
        mask1 = labels[...,0] > self.key_th1
        random_mask = tf.random.uniform(tf.shape(ids)) < 0.1
        mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
        feature = tf.boolean_mask(feature, mask1)
        decoder_outputs = self.decoder(feature)
        self.loss_func(labels, ids, maps, decoder_outputs, mask1, training=False)

    def fit(self, train_ds, test_ds, init_epoch=0, total_train=0, total_test=0):
        train_iterator = iter(train_ds)
        test_iterator = iter(test_ds)
        for epoch in range(init_epoch, self.max_epoch):
            if not is_alive():
                print('break')
                break

            make_run_lock(epoch)
            try:
                with tqdm.tqdm(range(total_train), desc="[train %d]"%(epoch+1)) as pbar:
                    for _ in pbar:
                        self.train_step(train_iterator)
                        pbar.set_postfix({
                            'dtc': self.dtcloss_mean.result().numpy(),
                            'key': self.keymap_loss_mean.result().numpy(),
                            'dec': self.decloss_mean.result().numpy(),
                            'cur': self.keymap0_loss_mean.result().numpy(),
                            'acc': self.id_acc.result().numpy(),
                            })
                        self.decloss_mean.reset_states()
                        self.keymap0_loss_mean.reset_states()

                with self.summary_writer_train.as_default():
                    i = self.optimizer.iterations
                    tf.summary.scalar('detector_loss', self.dtcloss_mean.result(), step=i)
                    tf.summary.scalar('keymap_loss', self.keymap_loss_mean.result(), step=i)
                    tf.summary.scalar('size_loss', self.size_loss_mean.result(), step=i)
                    tf.summary.scalar('offset_loss', self.offset_loss_mean.result(), step=i)
                    tf.summary.scalar('textline_loss', self.textline_loss_mean.result(), step=i)
                    tf.summary.scalar('separator_loss', self.separator_loss_mean.result(), step=i)
                    tf.summary.scalar('id_loss', self.id_loss_mean.result(), step=i)
                    tf.summary.scalar('id_acc', self.id_acc.result(), step=i)

                self.manager.save()

                self.dtcloss_mean.reset_states()
                self.decloss_mean.reset_states()
                self.keymap_loss_mean.reset_states()
                self.keymap0_loss_mean.reset_states()
                self.size_loss_mean.reset_states()
                self.offset_loss_mean.reset_states()
                self.textline_loss_mean.reset_states()
                self.separator_loss_mean.reset_states()
                self.id_loss_mean.reset_states()
                self.id_acc.reset_states()

                with tqdm.tqdm(range(total_test), desc="[test %d]"%(epoch+1)) as pbar:
                    for _ in pbar:
                        self.test_step(test_iterator)
                        pbar.set_postfix({
                            'dtc': self.dtcloss_mean.result().numpy(),
                            'dec': self.decloss_mean.result().numpy(),
                            'key': self.keymap_loss_mean.result().numpy(),
                            'acc': self.id_acc.result().numpy(),
                            })

                with self.summary_writer_test.as_default():
                    i = self.optimizer.iterations
                    tf.summary.scalar('detector_loss', self.dtcloss_mean.result(), step=i)
                    tf.summary.scalar('keymap_loss', self.keymap_loss_mean.result(), step=i)
                    tf.summary.scalar('size_loss', self.size_loss_mean.result(), step=i)
                    tf.summary.scalar('offset_loss', self.offset_loss_mean.result(), step=i)
                    tf.summary.scalar('textline_loss', self.textline_loss_mean.result(), step=i)
                    tf.summary.scalar('separator_loss', self.separator_loss_mean.result(), step=i)
                    tf.summary.scalar('id_loss', self.id_loss_mean.result(), step=i)
                    tf.summary.scalar('id_acc', self.id_acc.result(), step=i)

                self.dtcloss_mean.reset_states()
                self.decloss_mean.reset_states()
                self.keymap_loss_mean.reset_states()
                self.keymap0_loss_mean.reset_states()
                self.size_loss_mean.reset_states()
                self.offset_loss_mean.reset_states()
                self.textline_loss_mean.reset_states()
                self.separator_loss_mean.reset_states()
                self.id_loss_mean.reset_states()
                self.id_acc.reset_states()
            finally:
                rm_run_lock()

    def fit2(self, train_ds, init_epoch=0, total_train=0):
        train_iterator = iter(train_ds)
        for epoch in range(init_epoch, self.max_epoch):
            if not is_alive():
                break

            wait_run_lock(epoch)
            for _ in range(total_train):
                self.train_step(train_iterator)
    
    def train(self, data):
        train_ds = data.train_data(self.batch_size)
        test_ds = data.test_data(self.batch_size)

        if hvd.rank() == 0:
            self.fit(train_ds, test_ds, init_epoch=self.init_epoch, total_train=self.steps_per_epoch, total_test=self.steps_per_epoch // 4)
        else:
            self.fit2(train_ds, init_epoch=self.init_epoch, total_train=self.steps_per_epoch)


def train2(data, pretrain=None, step=1, lr=1e-4):
    detector = TextDetector2(pretrain=pretrain, step=step, lr=lr)
    if detector.init_epoch < detector.max_epoch:
        detector.train(data)

if __name__ == '__main__':
    data = dataset.FontData()
    train2(data, step=1, lr=1e-4)
    #train2(data, pretrain='pretrain/', step=1, lr=1e-5)
    #train2(data, pretrain=save_target+'step1/', step=2, lr=1e-5)
