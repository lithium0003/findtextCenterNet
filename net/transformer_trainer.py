import tensorflow as tf
import numpy as np
import struct

from .transformer import TextTransformer
from dataset.data_transformer import encoder_add_dim, max_decoderlen, decoder_SOT, decoder_EOT
from .detector_trainer import calc_predid
from .const import modulo_list, feature_dim

encoder_dim = feature_dim + encoder_add_dim

def padded_cross_entropy_loss(logits, labels, masks, smoothing=0.05):
    """Calculate cross entropy loss while ignoring padding.
    Args:
        logits: Tensor of size [batch_size, length_logits, vocab_size]
        labels: Tensor of size [batch_size, length_labels]
        masks: Tensor of size [batch_size, length_labels], 1 means vaild, 0 means masked
        smoothing: Label smoothing constant, used to determine the on and off values
        vocab_size: int size of the vocabulary
    Returns:
        Returns the cross entropy loss and weight tensors: float32 tensors with
            shape [batch_size, max(length_logits, length_labels)]
    """
    with tf.name_scope("loss"):
        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy"):
            num_classes = tf.cast(tf.shape(logits)[-1], tf.float32)
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / (num_classes - 1)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=tf.cast(num_classes, tf.int32),
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                confidence * tf.math.log(tf.clip_by_value(confidence, tf.keras.backend.epsilon(), 1.0)) +
                (num_classes - 1) * low_confidence * tf.math.log(tf.clip_by_value(low_confidence, tf.keras.backend.epsilon(), 1.0)))
            xentropy -= normalizing_constant

        weights = tf.cast(masks == True, tf.float32)
        return xentropy * weights, weights

class TransformerDecoderModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = TextTransformer()
        embedded = tf.keras.Input(shape=(None,encoder_dim))
        decoderinput = tf.keras.Input(shape=(None,))
        self.transformer((embedded, decoderinput))

        self.transformer.summary()
    
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.illegal_tracker = tf.keras.metrics.Mean(name='illegal')
        self.id_tracker = tf.keras.metrics.Mean(name='id')
        self.acc_tracker = tf.keras.metrics.Accuracy(name='id_acc')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.illegal_tracker,
            self.id_tracker,
            self.acc_tracker,
        ]

    def loss_func(self, inputs, outputs):
        label = inputs['decoder_true']
        mask = tf.logical_or(inputs['decoder_task'] > 0, inputs['decoder_true'] > 0)

        target_ids = []
        for modulo in modulo_list:
            target_id1 = label % modulo
            target_ids.append(target_id1)

        predictions = [tf.nn.softmax(predict, axis=-1) for predict in outputs]
        indices = [tf.argmax(predict, axis=-1) for predict in predictions]
        pred_idx = calc_predid(*indices)

        self.acc_tracker.update_state(
            tf.boolean_mask(label, mask),
            tf.boolean_mask(pred_idx, mask),
        )

        failed_idx = pred_idx >= 0x10FFFF

        illegal_count = tf.cast(failed_idx, dtype=tf.int64)
        illegal_count = tf.where(inputs['decoder_task'] > 0, illegal_count, 0)
        text_count = tf.cast(inputs['decoder_task'] > 0, dtype=tf.int64)
        illegal_count = tf.math.reduce_sum(illegal_count, axis=1) / tf.math.reduce_sum(text_count, axis=1)
        illegal_count = tf.math.reduce_mean(illegal_count)
        self.illegal_tracker.update_state(illegal_count)

        loss = 0.
        id_entropy = 0.
        for target_id1, decoder_id1 in zip(target_ids, outputs):
            xentropy, weights = padded_cross_entropy_loss(logits=decoder_id1, labels=target_id1, masks=mask)
            id_entropy += tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
            loss += tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        self.id_tracker.update_state(id_entropy)
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self.transformer((data['encoder_inputs'], data['decoder_task']), training=True)
            loss = self.loss_func(data, outputs)
        
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
            "illegal": self.illegal_tracker.result(),
            "id": self.id_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def test_step(self, data):
        outputs = self.transformer((data['encoder_inputs'], data['decoder_task']))
        loss = self.loss_func(data, outputs)

        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
            "illegal": self.illegal_tracker.result(),
            "id": self.id_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def generate(self, ds):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1,1), dtype=tf.int64),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int64),
            ])
        def call_loop(new_input, i, result, score):
            #tf.print(i)
            decoder_output = self.transformer.decoder([new_input, i])

            out1091, out1093, out1097 = decoder_output
            p1091 = tf.math.softmax(out1091)
            p1093 = tf.math.softmax(out1093)
            p1097 = tf.math.softmax(out1097)
            i1091 = tf.reshape(tf.argmax(p1091, axis=-1),[-1])
            i1093 = tf.reshape(tf.argmax(p1093, axis=-1),[-1])
            i1097 = tf.reshape(tf.argmax(p1097, axis=-1),[-1])
            code = calc_predid(i1091,i1093,i1097)
            p = tf.reduce_mean([tf.math.log(p1091[i1091]), tf.math.log(p1093[i1093]), tf.math.log(p1097[i1097])])
            return code[None,:], i+1, tf.concat([result, code], axis=-1), tf.concat([score, p], axis=-1)

        with open('generate.log','a') as f:
            result_text = tf.constant([['true','p','predict']])
            for j, inputs in enumerate(ds):
                print(j, flush=True)
                print(j, flush=True, file=f)
                if j >= 4:
                    break
                encoder_output = self.transformer.encoder(inputs['encoder_inputs'])
                self.transformer.decoder.create_cache(encoder_output)
                decoder_input = tf.constant([[decoder_SOT]], dtype=tf.int32),
                i0 = tf.constant(0)
                result = tf.zeros([0], dtype=tf.int64)
                score = tf.zeros([0])
                c = lambda n, i, r, s: tf.logical_and(i < max_decoderlen, r[-1] != decoder_EOT)
                with tf.device('cpu'):
                    _,_,output,score = tf.while_loop(
                        c, call_loop, loop_vars=[decoder_input, i0, result, score],
                        shape_invariants=[decoder_input.get_shape(), i0.get_shape(), tf.TensorShape([None,]), tf.TensorShape([None,])])

                score = tf.math.exp(tf.reduce_mean(score))
                pred_bytes = b''.join([i.to_bytes(4, 'little') for i in output])
                input_bytes = b''.join([int(i).to_bytes(4, 'little') for i in inputs['text'][0].numpy() if i > 0])
                pred_text = pred_bytes.decode("utf-32le", "backslashreplace")
                input_text = input_bytes.decode("utf-32le", "backslashreplace")
                pred_int = output
                input_int = input_text.encode('utf-32le')
                input_int = struct.unpack('I' * (len(input_int)//4), input_int)

                if pred_text == input_text:
                    print(input_text)
                    print('*%f'%score)
                    print(pred_text)
                    print(input_text, file=f)
                    print('*%f'%score, file=f)
                    print(pred_text, file=f)
                    result_text = tf.concat([result_text, tf.constant([[input_text, '*%f'%score , pred_text]])], axis=0)
                else:
                    print(input_text)
                    print(' %f'%score)
                    print(pred_text)
                    print(input_int)
                    print(pred_int)
                    print(input_text, file=f)
                    print(' %f'%score, file=f)
                    print(pred_text, file=f)
                    print(input_int, file=f)
                    print(pred_int, file=f)
                    result_text = tf.concat([result_text, tf.constant([[input_text, '%f'%score , pred_text]])], axis=0)
        return result_text
