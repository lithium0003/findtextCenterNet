#!/usr/bin/env python3
import coremltools as ct
import torch
import numpy as np
import os
from datetime import datetime
import itertools

from models.transformer import ModelDimensions, Transformer, TransformerEncoderPredictor, TransformerDecoderPredictor
from util_func import feature_dim, modulo_list, calc_predid
from const import encoder_add_dim, max_decoderlen, max_encoderlen, decoder_PAD, decoder_SOT, decoder_EOT, decoder_MSK

def convert3():
    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    if os.path.exists('model3.pt'):
        data = torch.load('model3.pt', map_location="cpu", weights_only=True)
        config = ModelDimensions(**data['config'])
        model = Transformer(**config.__dict__)
        model.load_state_dict(data['model_state_dict'])
        print('loaded')
    else:
        config = ModelDimensions()
        model = Transformer(**config.__dict__)
        print('empty model')
    model.eval()
    encoder = TransformerEncoderPredictor(model.encoder)
    decoder = TransformerDecoderPredictor(model.decoder)
    encoder.eval()
    decoder.eval()


    #########################################################################
    print('encoder')

    encoder_dim = feature_dim+encoder_add_dim
    encoder_input = torch.rand(1, max_encoderlen, encoder_dim)
    key_mask = torch.all(encoder_input == 0, dim=-1)
    key_mask = torch.where(key_mask[:,None,None,:], float("-inf"), 0)
    traced_model = torch.jit.trace(encoder, [encoder_input, key_mask])

    # def op_selector(op):
    #     print(op.op_type, [o.name for o in op.outputs])
    #     return True

    mlmodel_detector = ct.convert(traced_model,
            inputs=[
                ct.TensorType(name='encoder_input', shape=(1, max_encoderlen, encoder_dim)),
                ct.TensorType(name='key_mask', shape=(1, 1, 1, max_encoderlen)),
            ],
            outputs=[
                ct.TensorType(name='encoder_output'),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18)
            # compute_precision=ct.precision.FLOAT32)
    mlmodel_detector.version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlmodel_detector.save("TransformerEncoder.mlpackage")

    ############################################################################
    print('decoder')

    encoder_output = torch.rand(1, max_encoderlen, config.embed_dim)
    decoder_input = torch.randint(0, 1000, size=(1, max_decoderlen), dtype=torch.long)
    traced_model = torch.jit.trace(decoder, (encoder_output, decoder_input, key_mask))

    mlmodel_decoder = ct.convert(traced_model,
                                 convert_to="mlprogram",
                                 inputs=[
                                    ct.TensorType(name='encoder_output', shape=(1, max_encoderlen, config.embed_dim)),
                                    ct.TensorType(name='decoder_input', shape=(1, max_decoderlen), dtype=np.int32),
                                    ct.TensorType(name='key_mask', shape=(1, 1, 1, max_encoderlen)),
                                 ],
                                 outputs=[
                                    ct.TensorType(name='modulo_1091'),
                                    ct.TensorType(name='modulo_1093'),
                                    ct.TensorType(name='modulo_1097'),
                                 ],
                                 minimum_deployment_target=ct.target.iOS18)
                                #  compute_precision=ct.transform.FP16ComputePrecision(op_selector=op_selector))
                                #  compute_precision=ct.precision.FLOAT32)
    mlmodel_decoder.version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlmodel_decoder.save("TransformerDecoder.mlpackage")

def test3():
    print('load')
    mlmodel_encoder = ct.models.MLModel('TransformerEncoder.mlpackage')
    mlmodel_decoder = ct.models.MLModel('TransformerDecoder.mlpackage')

    rng = np.random.default_rng()
    train_data3 = 'train_data3'

    encoder_dim = feature_dim+encoder_add_dim
    encoder_input = np.zeros(shape=(1, max_encoderlen, encoder_dim), dtype=np.float32)
    SP_token = np.zeros([encoder_dim], dtype=np.float32)
    SP_token[0:feature_dim:2] = 5
    SP_token[1:feature_dim:2] = -5
    encoder_input[0,0,:] = SP_token
    with np.load(os.path.join(train_data3, 'features.npz')) as data:
        for i,c in enumerate('test'):
            code = ord(c)
            value = data['hori_%d'%code]
            feat = rng.choice(value, replace=False)
            encoder_input[0,i+1,:feature_dim] = feat
    encoder_input[0,i+2,:] = -SP_token

    key_mask = np.where((encoder_input == 0).all(axis=-1)[:,None,None,:], float("-inf"), 0)
    print('encoder')
    encoder_output = mlmodel_encoder.predict({'encoder_input': encoder_input, 'key_mask': key_mask})['encoder_output']

    print('decoder')
    decoder_input = np.zeros(shape=(1, max_decoderlen), dtype=np.int32)
    decoder_input[0,:] = decoder_MSK
    rep_count = 8
    for k in range(rep_count):
        output = mlmodel_decoder.predict({
            'encoder_output': encoder_output,
            'decoder_input': decoder_input,
            'key_mask': key_mask,
        })

        listp = []
        listi = []
        for m in modulo_list:
            pred_p1 = output['modulo_%d'%m]
            topi = np.argpartition(-pred_p1, 4, axis=-1)[...,:4]
            topp = np.take_along_axis(pred_p1, topi, axis=-1)
            listp.append(np.transpose(topp, (2,0,1)))
            listi.append(np.transpose(topi, (2,0,1)))

        pred_ids = np.stack([np.stack(x) for x in itertools.product(*listi)])
        pred_p = np.stack([np.stack(x) for x in itertools.product(*listp)])
        pred_ids = np.transpose(pred_ids, (1,0,2,3))
        pred_p = np.transpose(pred_p, (1,0,2,3))
        pred_p = np.exp(np.mean(np.log(np.maximum(pred_p, 1e-10)), axis=0))
        decoder_output = calc_predid(*pred_ids)
        pred_p[decoder_output > 0x3FFFF] = 0
        maxi = np.argmax(pred_p, axis=0)
        decoder_output = np.take_along_axis(decoder_output, maxi[None,...], axis=0)[0]
        pred_p = np.take_along_axis(pred_p, maxi[None,...], axis=0)[0]
        decoder_output = np.where(decoder_input == decoder_MSK, decoder_output, decoder_input)
        if np.all(pred_p[np.logical_and(decoder_input == decoder_MSK, decoder_output > 0)] > 0.99):
            print(f'---[{k} early stop]---')
            break
        if k < rep_count-1:
            r = int(max_decoderlen * (k+1) / rep_count)
            remask = np.arange(max_decoderlen) > r
            remask = np.logical_or(remask, decoder_output > 0x3FFFF)
            if r > 0:
                remask = np.logical_or(remask, np.logical_and(decoder_input == decoder_MSK, pred_p < 0.9))
            if not np.any(remask):
                break
            decoder_output = np.where(remask, decoder_MSK, decoder_output)
            decoder_input[:,:] = decoder_output[:,:]
    print(decoder_output[0])
    predstr = ''
    for p in decoder_output[0]:
        if p == decoder_SOT:
            continue
        if p == decoder_PAD or p == decoder_EOT:
            break
        if p >= 0xD800 and p <= 0xDFFF:
            predstr += '\uFFFD'
        elif p < 0x3FFFF:
            predstr += chr(p)
        else:
            predstr += '\uFFFD'
    try:
        print(predstr)
    except UnicodeEncodeError:
        pass

def test32():
    from models.transformer import TransformerPredictor

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)
    rng = np.random.default_rng()

    if os.path.exists('model3.pt'):
        data = torch.load('model3.pt', map_location="cpu", weights_only=True)
        config = ModelDimensions(**data['config'])
        model = Transformer(**config.__dict__)
        model.load_state_dict(data['model_state_dict'])
    else:
        config = ModelDimensions()
        model = Transformer(**config.__dict__)
    model2 = TransformerPredictor(model.encoder, model.decoder)
    model2.to(device)
    model2.eval()

    rng = np.random.default_rng()
    train_data3 = 'train_data3'

    encoder_dim = feature_dim+encoder_add_dim
    encoder_input = np.zeros(shape=(1, max_encoderlen, encoder_dim), dtype=np.float32)
    SP_token = np.zeros([encoder_dim], dtype=np.float32)
    SP_token[0:feature_dim:2] = 5
    SP_token[1:feature_dim:2] = -5
    with np.load(os.path.join(train_data3, 'features.npz')) as data:
        encoder_input[0,0,:] = SP_token
        for i,c in enumerate('test'):
            code = ord(c)
            value = data['hori_%d'%code]
            feat = rng.choice(value, replace=False)
            encoder_input[0,i+1,:feature_dim] = feat
        encoder_input[0,i+2,:] = -SP_token

    encoder_input = torch.tensor(encoder_input).to(device)
    pred = model2(encoder_input).squeeze(0).cpu().numpy()
    predstr = ''
    for p in pred:
        if p == decoder_SOT:
            continue
        if p == decoder_PAD or p == decoder_EOT:
            break
        if p >= 0xD800 and p <= 0xDFFF:
            predstr += '\uFFFD'
        elif p < 0x3FFFF:
            predstr += chr(p)
        else:
            predstr += '\uFFFD'
    print('------------------')
    try:
        print(predstr)
    except UnicodeEncodeError:
        pass
    print('==================')
    print(pred)        

if __name__ == '__main__':
    convert3()
    test3()
    # test32()
