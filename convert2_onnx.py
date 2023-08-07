#!/usr/bin/env python3
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime
import numpy as np

from const import max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT
from net.const import hidden_dim
from net.transformer_trainer import encoder_dim
from net.transformer import TextTransformer
from net.detector_trainer import calc_predid

def convert_encoder(model):
    print('encoder')

    embedded = tf.keras.Input(shape=(max_encoderlen,encoder_dim), name='encoder_input')

    encoder_output = tf.keras.layers.Lambda(lambda x: x, name='encoder_output', dtype='float32')(model.transformer.encoder(embedded))

    transformer_encoder = tf.keras.Model(embedded, encoder_output, name='TransformerEncoder')

    tf2onnx.convert.from_keras(transformer_encoder, output_path='TransformerEncoder.onnx')
    onnx.checker.check_model('TransformerEncoder.onnx')

def convert_decoder(model):
    decoder_input = tf.keras.Input(shape=(max_decoderlen,), name='decoder_input')
    encoder_output = tf.keras.Input(shape=(max_encoderlen,hidden_dim), name='encoder_output')
    encoder_input = tf.keras.Input(shape=(max_encoderlen,encoder_dim), name='encoder_input')

    class Decoder(tf.keras.models.Model):
        def __init__(
            self,
            decoder,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.decoder = decoder

        def call(self, inputs):
            decoder_input, encoder_output, encoder_input = inputs
            decoder_output = self.decoder([decoder_input, encoder_output, encoder_input])

            out1091, out1093, out1097 = decoder_output
            p1091 = out1091[0,:,:]
            p1093 = out1093[0,:,:]
            p1097 = out1097[0,:,:]
            return p1091, p1093, p1097

    decoder = Decoder(model.transformer.decoder)
    inputs = [decoder_input, encoder_output, encoder_input]

    p1091, p1093, p1097 = decoder(inputs)
    outputs = [
        tf.keras.layers.Lambda(lambda x: x, name='mod1091', dtype='float32')(p1091),
        tf.keras.layers.Lambda(lambda x: x, name='mod1093', dtype='float32')(p1093),
        tf.keras.layers.Lambda(lambda x: x, name='mod1097', dtype='float32')(p1097),
    ]

    transformer_decoder = tf.keras.Model(inputs, outputs, name='TransformerDecoder')
    tf2onnx.convert.from_keras(transformer_decoder, output_path='TransformerDecoder.onnx')
    onnx.checker.check_model('TransformerDecoder.onnx')

# class TransformerDecoderModel(tf.keras.models.Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.transformer = TextTransformer()
#         embedded = tf.keras.Input(shape=(max_encoderlen,encoder_dim))
#         decoderinput = tf.keras.Input(shape=(max_decoderlen,))
#         self.transformer((embedded, decoderinput))

#         self.transformer.summary()

def convert2():
    model = TransformerDecoderModel()
    last = tf.train.latest_checkpoint('ckpt2')
    print(last)
    model.load_weights(last).expect_partial()

    convert_encoder(model)
    convert_decoder(model)
    #return last

def testmodel():
    print('load char param')
    npz_file = np.load('charparam.npz')
    codes = []
    for varname in npz_file.files:
        codes.append(int(varname[:-1]))
    codes = sorted(codes)
    features = {}
    for code in codes:
        feature = npz_file['%dn'%code]
        features[chr(code)] = feature
    rng = np.random.default_rng()

    print('load')
    onnx_encoder = onnxruntime.InferenceSession("TransformerEncoder.onnx")
    onnx_decoder = onnxruntime.InferenceSession("TransformerDecoder.onnx")

    print('make input')
    encoder_input = [
        np.concatenate([rng.choice(features['t']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['e']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['s']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['t']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['o']), np.asarray([1, 0, 0, 0])]),
        np.concatenate([rng.choice(features['u']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['t']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['p']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['u']), np.asarray([0, 0, 0, 0])]),
        np.concatenate([rng.choice(features['t']), np.asarray([0, 0, 0, 1])]),
    ]
    encoder_input = np.pad(encoder_input, [[0, max_encoderlen - len(encoder_input)],[0,0]])
    encoder_input = np.expand_dims(encoder_input, 0).astype(np.float32)
    print('encoder')

    encoder_output, = onnx_encoder.run(['encoder_output'], { 'encoder_input': encoder_input })

    print('decoder')
    decoder_input = np.zeros([1,max_decoderlen], dtype=np.float32)
    decoder_input[0,0] = decoder_SOT
    count = 0
    while count < max_decoderlen - 1 and decoder_input[0,count] != decoder_EOT:
        mod1091, mod1093, mod1097 = onnx_decoder.run(['mod1091','mod1093','mod1097'], { 'decoder_input': decoder_input, 'encoder_output': encoder_output, 'encoder_input': encoder_input })
        i1091 = np.argmax(mod1091[count,:])
        i1093 = np.argmax(mod1093[count,:])
        i1097 = np.argmax(mod1097[count,:])
        code = calc_predid(i1091,i1093,i1097)
        count += 1
        decoder_input[0,count] = code

    code = decoder_input[0].astype(np.int32)
    print(code)
    str_code = code[1:count]
    str_text = ''.join([chr(c) if c < 0x110000 else '\uFFFD' for c in str_code])
    print(str_text)

if __name__ == '__main__':
    #convert2()
    testmodel()
