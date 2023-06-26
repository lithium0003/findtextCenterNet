import tensorflow as tf

from .const import width, height, modulo_list, feature_dim
from .detector import CenterNetDetectionBlock, SimpleDecoderBlock

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


# Multi-Loss Weighting with Coefficient of Variations
# https://arxiv.org/abs/2009.01717
class CoW_Weighting:
    def __init__(self, items, t) -> None:
        self.t = t
        self.mu_Lt = {}
        self.mu_lt = {}
        self.M_lt = {}
        self.clt = {}
        self.items = items
        for item in items:
            self.mu_Lt[item] = tf.Variable(0., trainable=False, dtype=tf.float32, name='%s_mu_Lt'%item)
            self.mu_lt[item] = tf.Variable(0., trainable=False, dtype=tf.float32, name='%s_mu_lt'%item)
            self.M_lt[item] = tf.Variable(0., trainable=False, dtype=tf.float32, name='%s_M_lt'%item)
            self.clt[item] = tf.Variable(0., trainable=False, dtype=tf.float32, name='%s_clt'%item)

    def process(self, losses):
        for item in self.items:
            Lt = tf.stop_gradient(losses[item])
            mu_Lt = self.mu_Lt[item]
            lt = tf.where(mu_Lt != 0, Lt / mu_Lt, Lt)
            self.mu_Lt[item].assign((1 - 1/self.t)*mu_Lt + Lt/self.t)
            mu0_lt = self.mu_lt[item]
            self.mu_lt[item].assign((1 - 1/self.t)*mu0_lt + lt/self.t)
            mu_lt = self.mu_lt[item]
            M_lt = self.M_lt[item]
            self.M_lt[item].assign((1 - 1/self.t)*M_lt + (lt - mu0_lt)*(lt - mu_lt)/self.t)
            sigma_lt = tf.math.sqrt(self.M_lt[item])
            self.clt[item].assign(sigma_lt / mu_lt)
        zt = 0.
        for item in self.items:
            zt += self.clt[item]
        scaled_loss = 0.
        for item in self.items:
            scaled_loss += losses[item] * self.clt[item] / zt
        return scaled_loss

    def test_call(self, losses):
        zt = 0.
        for item in self.items:
            zt += self.clt[item]
        scaled_loss = 0.
        for item in self.items:
            scaled_loss += losses[item] * self.clt[item] / zt
        return scaled_loss


class TextDetectorModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        pre_weight = kwargs.pop('pre_weight', True)
        frozen_backbone = kwargs.pop('frozen_backbone', False)
        super().__init__(**kwargs)

        self.loss_weight = CoW_Weighting([
            'keymap',
            'size',
            'offset',
            'textline',
            'separator',
            *['code%d'%i for i in range(4)],
            'id'
        ], 100)

        self.key_th2 = 0.9
        self.key_th3 = 0.99

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.main_loss_tracker = tf.keras.metrics.Mean(name='main_loss')
        self.keymap_loss_tracker = tf.keras.metrics.Mean(name='keymap_loss')
        self.size_loss_tracker = tf.keras.metrics.Mean(name='size_loss')
        self.offset_loss_tracker = tf.keras.metrics.Mean(name='offset_loss')
        self.textline_loss_tracker = tf.keras.metrics.Mean(name='textline_loss')
        self.separator_loss_tracker = tf.keras.metrics.Mean(name='separator_loss')
        self.code_loss_tracker = []
        for i in range(4):
            self.code_loss_tracker.append(tf.keras.metrics.Mean(name='code%d_loss'%(2**(i))))
        self.id_loss_tracker = tf.keras.metrics.Mean(name='id_loss')
        self.id_acc_tracker = tf.keras.metrics.Accuracy(name='id_acc')
        self.feature_std_tracker = tf.keras.metrics.Mean(name='feat_std')

        self.detector = CenterNetDetectionBlock(pre_weight=pre_weight, frozen_backbone=frozen_backbone)
        self.decoder = SimpleDecoderBlock()

        inputs = tf.keras.Input(shape=(height,width,3))
        self.detector(inputs)

    def compile(self, optimizer, backbone_optimizer, decoder_optimizer):
        with self.distribute_strategy.scope():
            self.optimizer = self._get_optimizer(optimizer)
            if backbone_optimizer is not None:
                self.backbone_optimizer = self._get_optimizer(backbone_optimizer)
            else:
                self.backbone_optimizer = None
            self.decoder_optimizer = self._get_optimizer(decoder_optimizer)
            self._reset_compile_cache()
            self._is_compiled = True

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.main_loss_tracker,
            self.keymap_loss_tracker,
            self.size_loss_tracker,
            self.offset_loss_tracker,
            self.textline_loss_tracker,
            self.separator_loss_tracker,
            *self.code_loss_tracker,
            self.id_loss_tracker,
            self.id_acc_tracker,
            self.feature_std_tracker,
            ]

    def train_step(self, data):
        image, labelmaps, code = data['image'], data['maps'], data['code']

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            maps, feature = self.detector(image, training=True)
            decoder_outputs = self.decoder(tf.boolean_mask(feature, labelmaps[...,0] > self.key_th3), training=True)
            loss, id_loss = self.loss_func(labelmaps, code, maps, decoder_outputs)
            num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
            if num_replicas > 1:
                loss *= (1 / num_replicas)
                id_loss *= (1 / num_replicas)
                
        std_feature = tf.math.reduce_std(tf.boolean_mask(feature, labelmaps[...,0] > self.key_th3))
        tf.cond(tf.math.is_finite(std_feature),
            true_fn=lambda: self.feature_std_tracker.update_state(std_feature),
            false_fn=lambda: tf.identity(std_feature))

        detector_weights = self.detector.trainable_weights
        if self.backbone_optimizer is not None:
            backbone_weights = self.detector.backbone.trainable_weights
            detector_weights = [item for item in detector_weights if item.name not in [w.name for w in backbone_weights]]

        self.optimizer.minimize(loss, detector_weights, tape=tape1)
        if self.backbone_optimizer is not None:
            self.backbone_optimizer.minimize(loss, backbone_weights, tape=tape2)
        self.decoder_optimizer.minimize(id_loss, self.decoder.trainable_weights, tape=tape3)

        code_loss = {}
        for i in range(4):
            code_loss['code%d_loss'%(2**(i))] = self.code_loss_tracker[i].result()

        return {
            "loss": self.loss_tracker.result(),
            "main_loss": self.main_loss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
            **code_loss,
            "feat_std": self.feature_std_tracker.result(),
        }

    def test_step(self, data):
        image, labelmaps, code = data['image'], data['maps'], data['code']

        maps, feature = self.detector(image)
        decoder_outputs = self.decoder(tf.boolean_mask(feature, labelmaps[...,0] > self.key_th3))
        self.loss_func(labelmaps, code, maps, decoder_outputs, train=False)

        std_feature = tf.math.reduce_std(tf.boolean_mask(feature, labelmaps[...,0] > self.key_th3))
        tf.cond(tf.math.is_finite(std_feature),
            true_fn=lambda: self.feature_std_tracker.update_state(std_feature),
            false_fn=lambda: tf.identity(std_feature))

        code_loss = {}
        for i in range(4):
            code_loss['code%d_loss'%(2**(i))] = self.code_loss_tracker[i].result()

        return {
            "loss": self.loss_tracker.result(),
            "main_loss": self.main_loss_tracker.result(),
            "id_acc": self.id_acc_tracker.result(),
            "id_loss": self.id_loss_tracker.result(),
            "keymap_loss": self.keymap_loss_tracker.result(),
            "size_loss": self.size_loss_tracker.result(),
            "offset_loss": self.offset_loss_tracker.result(),
            "textline_loss": self.textline_loss_tracker.result(),
            "separator_loss": self.separator_loss_tracker.result(),
            **code_loss,
            "feat_std": self.feature_std_tracker.result(),
        }

    def loss_func(self, labels, code, maps, decoder_outputs, train=True):
        weight1 = tf.minimum(labels[...,0] * 5.0, 1.0)
        weight2 = tf.maximum(labels[...,0] - self.key_th2, 0.) / (1 - self.key_th2)
        weight2 = tf.boolean_mask(weight2, labels[...,0] > self.key_th2)
        weight3 = tf.maximum(labels[...,0] - self.key_th3, 0.) / (1 - self.key_th3)
        weight3 = tf.boolean_mask(weight3, labels[...,0] > self.key_th3)
        weight2_count = tf.maximum(1., tf.reduce_sum(weight2))

        keymap_loss = heatmap_loss(true=labels[...,0], logits=maps[...,0])

        def huber(y_pred, y_true):
            delta = 1.0
            error = tf.subtract(y_pred, y_true)
            abs_error = tf.abs(error)
            half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
            return tf.where(
                    abs_error <= delta,
                    half * tf.square(error),
                    delta * abs_error - half * tf.square(delta),
                )

        xsize_loss = huber(
            tf.boolean_mask(labels[...,1], labels[...,0] > self.key_th2), 
            tf.boolean_mask(maps[...,1], labels[...,0] > self.key_th2))
        ysize_loss = huber(
            tf.boolean_mask(labels[...,2], labels[...,0] > self.key_th2), 
            tf.boolean_mask(maps[...,2], labels[...,0] > self.key_th2))
        size_loss = (xsize_loss + ysize_loss) * weight2
        size_loss = tf.math.reduce_sum(size_loss) / weight2_count

        xoffset_loss = huber(
            tf.boolean_mask(labels[...,3], labels[...,0] > self.key_th2), 
            tf.boolean_mask(maps[...,3], labels[...,0] > self.key_th2))
        yoffset_loss = huber(
            tf.boolean_mask(labels[...,4], labels[...,0] > self.key_th2), 
            tf.boolean_mask(maps[...,4], labels[...,0] > self.key_th2))
        offset_loss = (xoffset_loss + yoffset_loss) * weight2
        offset_loss = tf.math.reduce_sum(offset_loss) / weight2_count

        textline_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels[...,5], maps[...,5])
        textline_loss = tf.math.reduce_mean(textline_loss)

        separator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels[...,6], maps[...,6])
        separator_loss = tf.math.reduce_mean(separator_loss)

        code_losses = {}
        for i in range(4):
            label_map = tf.cast((code[...,1] & 2**(i)) > 0, tf.float32) * weight1
            predict_map = maps[...,7+i]
            code_loss = tf.nn.sigmoid_cross_entropy_with_logits(label_map, predict_map)
            code_loss = tf.math.reduce_mean(code_loss)
            self.code_loss_tracker[i].update_state(code_loss)
            code_losses['code%d'%i] = code_loss

        target_id = tf.boolean_mask(code[...,0], labels[...,0] > self.key_th3)
        target_ids = []
        for modulo in modulo_list:
            target_id1 = target_id % modulo
            target_ids.append(target_id1)

        id_loss = 0.
        for target_id1, decoder_id1 in zip(target_ids, decoder_outputs):
            id1_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(target_id1, decoder_id1)
            id1_loss = id1_loss * weight3
            weight2_count = tf.maximum(1., tf.math.reduce_sum(weight3))
            id_loss += tf.math.reduce_sum(id1_loss) / weight2_count

        pred_ids = []
        for decoder_id1 in decoder_outputs:
            prob_id1 = tf.nn.softmax(decoder_id1, -1)
            pred_id1 = tf.math.argmax(prob_id1, axis=-1)
            pred_ids.append(pred_id1)

        pred_id = calc_predid(*pred_ids)
        pred_id = tf.boolean_mask(pred_id, weight3 > 0.999)
        target_id = tf.boolean_mask(target_id, weight3 > 0.999)

        self.keymap_loss_tracker.update_state(keymap_loss)
        self.size_loss_tracker.update_state(size_loss)
        self.offset_loss_tracker.update_state(offset_loss)
        self.textline_loss_tracker.update_state(textline_loss)
        self.separator_loss_tracker.update_state(separator_loss)
        self.id_loss_tracker.update_state(id_loss)
        self.id_acc_tracker.update_state(target_id, pred_id)

        loss = keymap_loss + size_loss + offset_loss + textline_loss + separator_loss + id_loss
        for c_loss in code_losses.values():
            loss += c_loss

        if train:
            main_loss = self.loss_weight.process({
                'keymap': keymap_loss,
                'size': size_loss,
                'offset': offset_loss,
                'textline': textline_loss,
                'separator': separator_loss,
                **code_losses,
                'id': id_loss,
            })
        else:
            main_loss = self.loss_weight.test_call({
                'keymap': keymap_loss,
                'size': size_loss,
                'offset': offset_loss,
                'textline': textline_loss,
                'separator': separator_loss,
                **code_losses,
                'id': id_loss,
            })


        self.loss_tracker.update_state(loss)
        self.main_loss_tracker.update_state(main_loss)

        return main_loss, id_loss
