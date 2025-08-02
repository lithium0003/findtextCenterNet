from PIL import Image
import numpy as np
import itertools

from process_ocr_base import OCR_Processer, max_encoderlen, max_decoderlen, decoder_SOT, decoder_EOT, decoder_MSK, modulo_list, calc_predid

class OCR_coreml_Processer(OCR_Processer):
    def __init__(self):
        super().__init__()
        import coremltools as ct

        print('load')
        self.mlmodel_detector = ct.models.MLModel('TextDetector.mlpackage')

        self.mlmodel_transformer_encoder = ct.models.MLModel('TransformerEncoder.mlpackage')
        self.mlmodel_transformer_decoder = ct.models.MLModel('TransformerDecoder.mlpackage')

    def call_detector(self, image_input):
        input_image = Image.fromarray(image_input.squeeze(0).astype(np.uint8), mode="RGB")

        output = self.mlmodel_detector.predict({'image': input_image})
        heatmap = output['heatmap']
        features = output['feature']
        return heatmap, features

    def call_transformer(self, encoder_input):
        key_mask = np.where((encoder_input == 0).all(axis=-1)[:,None,None,:], float("-inf"), 0).astype(np.float32)
        encoder_output = self.mlmodel_transformer_encoder.predict({
            'encoder_input': encoder_input, 
            'key_mask': key_mask,
        })['encoder_output']

        decoder_input = np.zeros(shape=(1, max_decoderlen), dtype=np.int32)
        decoder_input[0,:] = decoder_MSK
        rep_count = 8
        for k in range(rep_count):
            output = self.mlmodel_transformer_decoder.predict({
                'encoder_output': encoder_output,
                'decoder_input': decoder_input,
                'key_mask': key_mask,
            })

            listp = []
            listi = []
            for m in modulo_list:
                pred_p1 = output['modulo_%d'%m]
                topi = np.argpartition(-pred_p1, 5, axis=-1)[...,:5]
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