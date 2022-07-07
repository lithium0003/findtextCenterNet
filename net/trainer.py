import tensorflow as tf
import numpy as np
import os

from .detector import width, height
from .detector import modulo_list, CenterNetDetectionBlock, SimpleDecoderBlock

def calc_predid(*args):
    m = modulo_list
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


class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        lockBase = kwargs.pop('lockBase', False) 
        average = kwargs.pop('average', 1)
        syncbn = kwargs.pop('syncbn', False)
        logdir = kwargs.pop('logdir', None)
        pre_weight = kwargs.pop('pre_weight', True)
        super().__init__(**kwargs)

        self.key_th1 = 0.8

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.regloss_tracker = tf.keras.metrics.Mean(name='regloss')
        self.keymap_loss_tracker = tf.keras.metrics.Mean(name='keymap_loss')
        self.size_loss_tracker = tf.keras.metrics.Mean(name='size_loss')
        self.offset_loss_tracker = tf.keras.metrics.Mean(name='offset_loss')
        self.textline_loss_tracker = tf.keras.metrics.Mean(name='textline_loss')
        self.separator_loss_tracker = tf.keras.metrics.Mean(name='separator_loss')
        self.id_loss_tracker = tf.keras.metrics.Mean(name='id_loss')
        self.id_acc_tracker = tf.keras.metrics.Accuracy(name='id_acc')

        self.detector = CenterNetDetectionBlock(pre_weight=pre_weight, average=average, syncbn=syncbn, lockBase=lockBase)
        self.decoder = SimpleDecoderBlock()

        inputs = tf.keras.Input(shape=(height,width,3))
        self.detector(inputs)

        if logdir:
            self.train_writer = tf.summary.create_file_writer(os.path.join(logdir,'train'))
            self.test_writer = tf.summary.create_file_writer(os.path.join(logdir,'validation'))
        else:
            self.train_writer = None
            self.test_writer = None

        if average > 1:
            self.average = tf.constant(average, dtype=tf.float32)
            self.n_gradients = tf.constant(average, dtype=tf.int32)
            self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            self.detector_gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.detector.trainable_variables]
            self.train_step = self.train_step2
        else:
            self.train_step = self.train_step1

    def compile(self, detector_optimizer, decoder_optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = self._get_optimizer(detector_optimizer)
        self.decoder_optimizer = self._get_optimizer(decoder_optimizer)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.regloss_tracker,
            self.keymap_loss_tracker,
            self.size_loss_tracker,
            self.offset_loss_tracker,
            self.textline_loss_tracker,
            self.separator_loss_tracker,
            self.id_loss_tracker,
            self.id_acc_tracker,
            ]

    def train_step1(self, data):
        image, labels, ids = data

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            maps, feature = self.detector(image, training=True)
            mask1 = labels[...,0] > self.key_th1
            random_mask = tf.random.uniform(tf.shape(ids)) < 0.01
            mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
            raw_feature = feature
            feature = tf.boolean_mask(feature, mask1)
            decoder_outputs = self.decoder(feature, training=True)
            main_loss, decoder_loss = self.loss_func(labels, ids, maps, raw_feature, decoder_outputs, mask1, training=True)

        self.optimizer.minimize(main_loss, self.detector.trainable_variables, tape=tape1)
        self.decoder_optimizer.minimize(decoder_loss, self.decoder.trainable_variables, tape=tape2)

        if self.train_writer is not None:
            step = self._train_counter
            with tf.summary.record_if(tf.equal(step % 1000, 0)):
                with self.train_writer.as_default(step=step):
                    tf.summary.histogram("feature", raw_feature)

        return {
            "loss": self.loss_tracker.result(),
            "regloss": self.regloss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
        }

    def train_step2(self, data):
        self.n_acum_step.assign_add(1)
        image, labels, ids = data

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            maps, feature = self.detector(image, training=True)
            mask1 = labels[...,0] > self.key_th1
            random_mask = tf.random.uniform(tf.shape(ids)) < 0.01
            mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
            raw_feature = feature
            feature = tf.boolean_mask(feature, mask1)
            decoder_outputs = self.decoder(feature, training=True)
            main_loss, decoder_loss = self.loss_func(labels, ids, maps, raw_feature, decoder_outputs, mask1, training=True)

        self.decoder_optimizer.minimize(decoder_loss, self.decoder.trainable_variables, tape=tape2)

        # Calculate batch gradients
        detector_gradients = tape1.gradient(main_loss, self.detector.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.detector_gradient_accumulation)):
            self.detector_gradient_accumulation[i].assign_add(detector_gradients[i] / self.average)

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        if self.train_writer is not None:
            step = self._train_counter
            with tf.summary.record_if(tf.equal(step % 1000, 0)):
                with self.train_writer.as_default(step=step):
                    tf.summary.histogram("feature", raw_feature)

        return {
            "loss": self.loss_tracker.result(),
            "regloss": self.regloss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
        }

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.detector_gradient_accumulation, self.detector.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.detector_gradient_accumulation)):
            self.detector_gradient_accumulation[i].assign(tf.zeros_like(self.detector.trainable_variables[i], dtype=tf.float32))

    def test_step(self, data):
        image, labels, ids = data

        maps, feature = self.detector(image)
        mask1 = labels[...,0] > self.key_th1
        random_mask = tf.random.uniform(tf.shape(ids)) < 0.01
        mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
        raw_feature = feature
        feature = tf.boolean_mask(feature, mask1)
        decoder_outputs = self.decoder(feature)
        self.loss_func(labels, ids, maps, raw_feature, decoder_outputs, mask1, training=False)

        if self.test_writer is not None:
            step = self._train_counter + self._test_counter
            with tf.summary.record_if(tf.equal(step % 1000, 0)):
                with self.test_writer.as_default(step=step):
                    tf.summary.histogram("feature", raw_feature)

        return {
            "loss": self.loss_tracker.result(),
            "regloss": self.regloss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
        }

    def loss_func(self, labels, target_id, maps, feature, decoder_outputs, mask1, training=False):
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
        for modulo in modulo_list:
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

        feature_loss = tf.math.reduce_mean(tf.maximum(tf.square(feature) - 100., 0.))

        self.regloss_tracker.update_state(feature_loss)
        self.keymap_loss_tracker.update_state(keymap_loss)
        self.size_loss_tracker.update_state(size_loss)
        self.offset_loss_tracker.update_state(offset_loss)
        self.textline_loss_tracker.update_state(textline_loss)
        self.separator_loss_tracker.update_state(separator_loss)
        self.id_loss_tracker.update_state(id_loss)
        self.id_acc_tracker.update_state(target_id, pred_id)

        decoder_loss = id_loss

        main_loss = feature_loss + offset_loss + id_loss + keymap_loss + size_loss + textline_loss + separator_loss
        self.loss_tracker.update_state(main_loss)

        return main_loss, decoder_loss
