import tensorflow as tf

from .const import width, height, feature_dim, modulo_list
from .mscan import mscan_t, mscan_s, mscan_b, mscan_l, load_pretrain

class LeafmapModel(tf.keras.Model):
    def __init__(self, out_dim=1, mid_dim=64, **kwarg):
        super().__init__(**kwarg)

        self.conv1 = []
        conv_dims1 = [s for s in [8,8,16,32]]
        for i, dim in enumerate(conv_dims1):
            self.conv1.append([
                tf.keras.layers.Activation('gelu', name=self.name+'_%s_act'%('abcd'[i])),
                tf.keras.layers.Conv2D(dim,
                                       kernel_size=5,
                                       padding='same',
                                       use_bias=False,
                                       name=self.name+'_%s_conv'%('abcd'[i])),
                tf.keras.layers.BatchNormalization(momentum=0.9, name=self.name+'_%s_bn'%('abcd'[i])),
                tf.keras.layers.UpSampling2D(size=2**(i+1), interpolation='bilinear', name=self.name+'_%s_up'%('abcd'[i])),
            ])
        
        self.top_conv = [
            tf.keras.layers.Activation('gelu', name=self.name+'_top1_act'),
            tf.keras.layers.Conv2D(mid_dim,
                                   kernel_size=3,
                                   padding='same',
                                   use_bias=False,
                                   name=self.name+'_top_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, name=self.name+'_top_bn'),
            tf.keras.layers.Activation('gelu', name=self.name+'_top2_act'),
        ]

        self.convout = tf.keras.layers.Conv2D(out_dim,
                                              kernel_size=3,
                                              padding='same',
                                              name=self.name+'_out_conv')
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs, **kwarg):
        P_in = inputs

        x = []
        for x_in, conv in zip(P_in, self.conv1):
            x1 = x_in
            for layer in conv:
                x1 = layer(x1, **kwarg)
            x.append(x1)
        x = tf.concat(x, axis=-1)

        for layer in self.top_conv:
            x = layer(x, **kwarg)

        x = self.convout(x)
        x = self.outfloat32(x)
        return x

class ClassModuloModel(tf.keras.Model):
    def __init__(self, out_dim=97, **kwarg):
        super().__init__(**kwarg)
        self.dense_layers = [
            tf.keras.layers.Dense(1024,
                                  name=self.name+'dense_1'),
            tf.keras.layers.Activation('gelu', name=self.name+'_act1'),
        ]
        self.dense_out = tf.keras.layers.Dense(out_dim, 
                                               name=self.name+'dense_out')
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_out(x)
        x = self.outfloat32(x)
        return x

class CenterNetDetectionModel(tf.keras.Model):
    def __init__(self, pre_weight=True, frozen_backbone=False, **kwarg):
        super().__init__(**kwarg)
        self.backbone = mscan_b()
        self.pre_weight = pre_weight
        if frozen_backbone:
            self.backbone.trainable = False

        self.keyheatmap = LeafmapModel(out_dim=1, name='keyheatmap')
        self.sizes = LeafmapModel(out_dim=2, mid_dim=16, name='sizes')
        self.offsets = LeafmapModel(out_dim=2, mid_dim=16, name='offsets')
        self.textline = LeafmapModel(out_dim=1, mid_dim=8, name='textline')
        self.sepatator = LeafmapModel(out_dim=1, mid_dim=8, name='sepatator')
        self.codes = LeafmapModel(out_dim=4, name='codes')
        self.feature = LeafmapModel(out_dim=feature_dim, name='feature')

    def build(self, input_shape):
        super().build(input_shape)
        if self.pre_weight:
            print('load pre_weight', self.backbone.name)

            path_to_downloaded_file = tf.keras.utils.get_file(
                origin='https://bucket.lithium03.info/mscan_preweights/%s.pickle'%self.backbone.name)

            load_pretrain(self.backbone, path_to_downloaded_file)

    def call(self, inputs, **kwarg):
        x = tf.keras.applications.imagenet_utils.preprocess_input(inputs, mode='torch')
        backbone_out = self.backbone(x)

        maps = tf.keras.layers.Concatenate(dtype='float32')([
            self.keyheatmap(backbone_out), 
            self.sizes(backbone_out), 
            self.offsets(backbone_out), 
            self.textline(backbone_out), 
            self.sepatator(backbone_out),
            self.codes(backbone_out),
        ])
        return maps, self.feature(backbone_out)

def CenterNetDetectionBlock(pre_weight=True, frozen_backbone=False):
    return CenterNetDetectionModel(pre_weight=pre_weight, frozen_backbone=frozen_backbone, name='CenterNetBlock')

def SimpleDecoderBlock():
    embedded = tf.keras.Input(shape=(feature_dim,))

    outputs = []
    for modulo in modulo_list:
        dense = ClassModuloModel(out_dim=modulo, name='modulo%d'%modulo)(embedded)
        outputs.append(dense)

    return tf.keras.Model(embedded, outputs, name='SimpleDecoderBlock')

if __name__ == '__main__':
    # model = BackboneModel(pre_weight=False)
    # inputs = tf.keras.Input(shape=(height,width,3))
    # outputs = model(inputs)
    # model.summary()
    # print(outputs)
    # exit()    

    model = CenterNetDetectionBlock()
    inputs = tf.keras.Input(shape=(height,width,3))
    outputs = model(inputs)
    model.summary()

    decoder = SimpleDecoderBlock()
    decoder.summary()

    def print_layers(layers):
        print(layers.name, layers.trainable)
        if hasattr(layers, 'layers'):
            for l in layers.layers:
                print_layers(l)
        if hasattr(layers, 'weights'):
            print(layers.weights)

    print_layers(model)
    