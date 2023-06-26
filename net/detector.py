import tensorflow as tf

backbone_model = 'efficientnet'

if backbone_model == 'mscan':
    from .detector_mscan import CenterNetDetectionBlock, SimpleDecoderBlock
elif backbone_model == 'mit':
    from .detector_mit import CenterNetDetectionBlock, SimpleDecoderBlock
elif backbone_model == 'efficientnet':
    from .detector_efficientnet import CenterNetDetectionBlock, SimpleDecoderBlock
