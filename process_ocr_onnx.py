import numpy as np
import itertools

from process_ocr_base import OCR_Processer, max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT, decoder_MSK, modulo_list, calc_predid

class OCR_onnx_Processer(OCR_Processer):
    def __init__(self):
        super().__init__()
        import onnxruntime
        import os

        print('load')
        if os.path.exists("TextDetector.quant.onnx"):
            print('quant')
            onnx_detector = onnxruntime.InferenceSession("TextDetector.quant.onnx")
        else:
            onnx_detector = onnxruntime.InferenceSession("TextDetector.onnx")
        self.onnx_detector = onnx_detector
        self.onnx_transformer_encoder = onnxruntime.InferenceSession("TransformerEncoder.onnx")
        self.onnx_transformer_decoder = onnxruntime.InferenceSession("TransformerDecoder.onnx")

    def call_detector(self, image_input):
        images = (image_input / 255.).transpose(0,3,1,2).astype(np.float32)
        heatmap, features = self.onnx_detector.run(['heatmap','feature'], {'image': images})
        return heatmap, features

    def call_transformer(self, encoder_input):
        key_mask = np.where((encoder_input == 0).all(axis=-1)[:,None,None,:], float("-inf"), 0).astype(np.float32)
        encoder_output, = self.onnx_transformer_encoder.run(['encoder_output'], {'encoder_input': encoder_input.astype(np.float32), 'key_mask': key_mask.astype(np.float32)})

        decoder_input = np.zeros(shape=(1, max_decoderlen), dtype=np.int64)
        decoder_input[0,:] = decoder_MSK
        rep_count = 8
        for k in range(rep_count):
            output = self.onnx_transformer_decoder.run(['modulo_%d'%m for m in modulo_list], {
                'encoder_output': encoder_output,
                'decoder_input': decoder_input,
                'key_mask': key_mask,
            })

            listp = []
            listi = []
            for pred_p1 in output:
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
            if np.all(pred_p[decoder_output > 0] > 0.99):
                print(f'[{k} early stop]')
                break

            remask = decoder_output > 0x3FFFF
            remask = np.logical_or(remask, pred_p < 0.9)
            if not np.any(remask):
                print(f'---[{k} early stop]---')
                break

            decoder_input[:,:] = np.where(remask, decoder_MSK, decoder_output)

        pred = decoder_output[0]
        return pred