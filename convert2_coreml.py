#!/usr/bin/env python3
import tensorflow as tf
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
import os
import time
import glob
from datetime import datetime

from const import max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT
from net.const import hidden_dim, head_num, hopping_num_decoder
from net.transformer import TextTransformer
from net.transformer_trainer import encoder_dim
from net.detector_trainer import calc_predid

def convert_encoder(model):
    print('encoder')

    embedded = tf.keras.Input(shape=(max_encoderlen,encoder_dim), name='encoder_input')

    encoder_output = model.transformer.encoder(embedded)

    transformer_encoder = tf.keras.Model(embedded, encoder_output, name='TransformerEncoder')

    mlmodel_transformer_encoder = ct.convert(transformer_encoder,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name='encoder_input', shape=ct.Shape(shape=(1, max_encoderlen, encoder_dim))),
            ],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16)
    mlmodel_transformer_encoder.version = datetime.now().strftime("%Y%m%d%H%M%S")
    spec = mlmodel_transformer_encoder.get_spec()

    # get output names
    output_names = [out.name for out in spec.description.output]

    ct.utils.rename_feature(spec, output_names[0], 'encoder_output')
    mlmodel_transformer_encoder_fix = ct.models.MLModel(spec, weights_dir=mlmodel_transformer_encoder.weights_dir)
    mlmodel_transformer_encoder_fix.save("TransformerEncoder.mlpackage")

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

    transformer_decoder = tf.keras.Model(inputs, decoder(inputs), name='TransformerDecoder')

    mlmodel_transformer_decoder = ct.convert(transformer_decoder,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name='decoder_input', shape=ct.Shape(shape=(1, max_decoderlen))),
                ct.TensorType(name='encoder_output', shape=ct.Shape(shape=(1, max_encoderlen, hidden_dim))),
                ct.TensorType(name='encoder_input', shape=ct.Shape(shape=(1, max_encoderlen, encoder_dim))),
            ],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16)

    mlmodel_transformer_decoder.version = datetime.now().strftime("%Y%m%d%H%M%S")
    spec = mlmodel_transformer_decoder.get_spec()

    ct.utils.rename_feature(spec, 'Identity', 'mod1091')
    ct.utils.rename_feature(spec, 'Identity_1', 'mod1093')
    ct.utils.rename_feature(spec, 'Identity_2', 'mod1097')

    mlmodel_transformer_decoder = ct.models.MLModel(spec, weights_dir=mlmodel_transformer_decoder.weights_dir)
    mlmodel_transformer_decoder.save("TransformerDecoder.mlpackage")

class TransformerDecoderModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = TextTransformer()
        embedded = tf.keras.Input(shape=(max_encoderlen,encoder_dim))
        decoderinput = tf.keras.Input(shape=(max_decoderlen,))
        self.transformer((embedded, decoderinput))

        self.transformer.summary()

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
    mlmodel_encoder = ct.models.MLModel('TransformerEncoder.mlpackage')
    mlmodel_decoder = ct.models.MLModel('TransformerDecoder.mlpackage')

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
    encoder_input = np.expand_dims(encoder_input, 0)
    print('encoder')
    out1 = mlmodel_encoder.predict({ 'encoder_input': encoder_input })

    print('decoder')
    decoder_input = np.zeros([1,max_decoderlen], dtype=np.float32)
    decoder_input[0,0] = decoder_SOT
    count = 0
    while count < max_decoderlen - 1 and decoder_input[0,count] != decoder_EOT:
        out2 = mlmodel_decoder.predict({ 'decoder_input': decoder_input, **out1, 'encoder_input': encoder_input })
        mod1091 = out2['mod1091']
        mod1093 = out2['mod1093']
        mod1097 = out2['mod1097']
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
    convert2()
    testmodel()