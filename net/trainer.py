import tensorflow as tf

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
        renorm = kwargs.pop('renorm', False)
        syncbn = kwargs.pop('syncbn', False)
        super().__init__(**kwargs)

        self.key_th1 = 0.8

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.keymap_loss_tracker = tf.keras.metrics.Mean(name='keymap_loss')
        self.size_loss_tracker = tf.keras.metrics.Mean(name='size_loss')
        self.offset_loss_tracker = tf.keras.metrics.Mean(name='offset_loss')
        self.textline_loss_tracker = tf.keras.metrics.Mean(name='textline_loss')
        self.separator_loss_tracker = tf.keras.metrics.Mean(name='separator_loss')
        self.id_loss_tracker = tf.keras.metrics.Mean(name='id_loss')
        self.id_acc_tracker = tf.keras.metrics.Accuracy(name='id_acc')

        self.detector = CenterNetDetectionBlock(renorm=renorm, syncbn=syncbn)
        self.decoder = SimpleDecoderBlock()

        self.mu_loss = {}
        self.mu_lt = {}
        self.M_lt = {}
        for key in ['keymap','size','offset','textline','separator','id']:
            self.mu_loss[key] = self.add_weight(name=key+'_mu_loss', initializer="zeros", dtype=tf.float32, trainable=False)
            self.mu_lt[key] = self.add_weight(name=key+'_mu_lt', initializer="zeros", dtype=tf.float32, trainable=False)
            self.M_lt[key] = self.add_weight(name=key+'_M_lt', initializer="zeros", dtype=tf.float32, trainable=False)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.keymap_loss_tracker,
            self.size_loss_tracker,
            self.offset_loss_tracker,
            self.textline_loss_tracker,
            self.separator_loss_tracker,
            self.id_loss_tracker,
            self.id_acc_tracker,
            ]

    def train_step(self, data):
        image, labels, ids = data

        with tf.GradientTape() as tape:
            maps, feature = self.detector(image, training=True)
            mask1 = labels[...,0] > self.key_th1
            random_mask = tf.random.uniform(tf.shape(ids)) < 0.1
            mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
            feature = tf.boolean_mask(feature, mask1)
            decoder_outputs = self.decoder(feature, training=True)
            loss = self.loss_func(labels, ids, maps, decoder_outputs, mask1, training=True)
            loss /= self.distribute_strategy.num_replicas_in_sync

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {
            "loss": self.loss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
        }

    def test_step(self, data):
        image, labels, ids = data

        maps, feature = self.detector(image)
        mask1 = labels[...,0] > self.key_th1
        random_mask = tf.random.uniform(tf.shape(ids)) < 0.1
        mask1 = tf.where(tf.reduce_sum(tf.cast(mask1, tf.float32)) > 0, mask1, random_mask)
        feature = tf.boolean_mask(feature, mask1)
        decoder_outputs = self.decoder(feature)
        self.loss_func(labels, ids, maps, decoder_outputs, mask1, training=False)

        return {
            "loss": self.loss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
        }

    def CoV_Weight(self, training=True, **kwarg):
        t = 20
        a = {}
        for key in kwarg:
            if training:
                self.mu_loss[key].assign((1 - 1/t) * self.mu_loss[key] + 1/t * kwarg[key])
                l_t = kwarg[key] / self.mu_loss[key]
                mu_lt_1 = self.mu_lt[key]
                self.mu_lt[key].assign((1 - 1/t) * mu_lt_1 + 1/t * l_t)
                self.M_lt[key].assign((1 - 1/t) * self.M_lt[key] + 1/t * (l_t - mu_lt_1) * (l_t - self.mu_lt[key]))
            sigma_lt = tf.sqrt(self.M_lt[key])
            mu_lt = self.mu_lt[key]
            a[key] = sigma_lt / mu_lt

        z = 0.
        for key in kwarg:
            z = z + a[key]
        
        for key in kwarg:
            a[key] = a[key] / z

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

        self.keymap_loss_tracker.update_state(keymap_loss)
        self.size_loss_tracker.update_state(size_loss)
        self.offset_loss_tracker.update_state(offset_loss)
        self.textline_loss_tracker.update_state(textline_loss)
        self.separator_loss_tracker.update_state(separator_loss)
        self.id_loss_tracker.update_state(id_loss)
        self.id_acc_tracker.update_state(target_id, pred_id)

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
        self.loss_tracker.update_state(loss)

        return loss
