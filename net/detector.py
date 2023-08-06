import tensorflow as tf

from .efficientnetv2xl import EfficientNetV2XL
from .const import width, height, feature_dim, modulo_list

class BackboneModel(tf.keras.Model):
    def __init__(self, pre_weight=True, model='s', frozen_bn=False, **kwarg):
        super().__init__(**kwarg)
        if model == 's':
            base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False, weights=None)
            model_name = 'efficientnetv2-s'
        elif model == 'm':
            base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(include_top=False, weights=None)
            model_name = 'efficientnetv2-m'
        elif model == 'l':
            base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False, weights=None)
            model_name = 'efficientnetv2-l'
        elif model == 'xl':
            base_model = EfficientNetV2XL()
            model_name = 'efficientnetv2-xl'

        outlayers = [layer for layer in base_model.layers if layer.name.endswith('_add')]
        mid_output3 = [layer for layer in outlayers if 'block2' in layer.name][-1]
        mid_output2 = [layer for layer in outlayers if 'block3' in layer.name][-1]
        mid_output1 = [layer for layer in outlayers if 'block5' in layer.name][-1]
        if model == 's':
            mid_output0 = [layer for layer in outlayers if 'block6' in layer.name][-1]
        else:
            mid_output0 = [layer for layer in outlayers if 'block7' in layer.name][-1]
        self.extract_model = tf.keras.Model(base_model.input, [mid_output3.output, mid_output2.output, mid_output1.output, mid_output0.output], name=model_name)

        if pre_weight:
            path_to_downloaded_file = tf.keras.utils.get_file(
                'efficientnetv2-%s-21k-ft1k'%model, 
                'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-%s-21k-ft1k.tgz'%model,
                untar=True)

            ckpt = tf.train.latest_checkpoint(path_to_downloaded_file)
        else:
            ckpt = None

        if ckpt:
            print(ckpt)

            reader = tf.train.load_checkpoint(ckpt)
            #variable_map = reader.get_variable_to_shape_map()

            blocknames = {}         
            for v in self.extract_model.variables:
                destname = v.name.split(':')[0]
                bname = destname.split('/')[0]
                bname, lname = bname.split('_', 1)
                if bname not in blocknames:
                    blocknames[bname] = []
                if lname not in blocknames[bname]:
                    blocknames[bname].append(lname)

            for block in blocknames:
                l1 = blocknames[block]
                blocknames[block] = []
                for lname in l1:
                    blocknames[block].append([lname])

            for b, block in enumerate(blocknames):
                if b == 0:
                    basename = 'efficientnetv2-%s/stem/'%model
                elif block != 'top':
                    basename = 'efficientnetv2-%s/blocks_%d/'%(model, b-1)
                else:
                    basename = 'efficientnetv2-%d/head/'%model

                convcount = 0
                bncount = 0
                for lname in blocknames[block]:
                    if lname[0] == 'conv' or lname[0].endswith('_conv'):
                        layername = 'conv2d'
                        if convcount > 0:
                            layername += '_%d'%convcount
                        convcount += 1
                    elif lname[0] == 'bn' or lname[0].endswith('_bn'):
                        layername = 'tpu_batch_normalization'
                        if bncount > 0:
                            layername += '_%d'%bncount
                        bncount += 1
                    elif lname[0] == 'dwconv2':
                        layername = 'depthwise_conv2d'
                    elif lname[0] == 'se_reduce':
                        layername = 'se/conv2d'
                    elif lname[0] == 'se_expand':
                        layername = 'se/conv2d_1'
                    else:
                        print(lname)
                        exit(1)

                    target = basename + layername
                    lname.append(target)

            #print(blocknames)

            for v in self.extract_model.variables:
                #print(v.name, v.shape)
                destname = v.name.split(':')[0]
                bname = destname.split('/')[0]
                bname, lname = bname.split('_', 1)
                for l, n in blocknames[bname]:
                    if lname == l:
                        k = destname.split('/')[-1]
                        key = n+'/'+k
                        #print(key, variable_map[key])
                            
                        tensor = reader.get_tensor(key)
                        v.assign(tensor)

            self.finalize_state()

        #print([layer.weights for layer in self.extract_model.layers])

        if frozen_bn:
            for layer in self.extract_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False          

    def finalize_state(self):
        for layer in self.extract_model.layers:
            layer.finalize_state()

    def call(self, inputs, **kwarg):
        x = inputs
        mid_output3,mid_output2,mid_output,final_output = self.extract_model(x, **kwarg)
        outputs = [mid_output3,mid_output2,mid_output,final_output]

        return outputs

class LeafmapModel(tf.keras.Model):
    def __init__(self, out_dim=1, mid_dim=64, **kwarg):
        super().__init__(**kwarg)

        self.conv1 = []
        conv_dims1 = [s for s in [8,8,16,32]]
        for i, dim in enumerate(conv_dims1):
            self.conv1.append([
                tf.keras.layers.Activation('swish', name=self.name+'_%s_act'%('abcd'[i])),
                tf.keras.layers.Conv2D(dim,
                                       kernel_size=5,
                                       padding='same',
                                       use_bias=False,
                                       name=self.name+'_%s_conv'%('abcd'[i])),
                tf.keras.layers.BatchNormalization(momentum=0.9, name=self.name+'_%s_bn'%('abcd'[i])),
                tf.keras.layers.UpSampling2D(size=2**(i+1), interpolation='bilinear', name=self.name+'_%s_up'%('abcd'[i])),
            ])
        
        self.top_conv = [
            tf.keras.layers.Activation('swish', name=self.name+'_top1_act'),
            tf.keras.layers.Conv2D(mid_dim,
                                   kernel_size=3,
                                   padding='same',
                                   use_bias=False,
                                   name=self.name+'_top_conv'),
            tf.keras.layers.BatchNormalization(momentum=0.9, name=self.name+'_top_bn'),
            tf.keras.layers.Activation('swish', name=self.name+'_top2_act'),
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
            tf.keras.layers.Activation('swish', name=self.name+'_act1'),
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
        self.backbone = BackboneModel(pre_weight=pre_weight, model='xl')
        if frozen_backbone:
            self.backbone.trainable = False

        self.bn = [
            tf.keras.layers.BatchNormalization(momentum=0.9, name='backbone_%s_bn'%('abcd'[i])) for i in range(4)
        ]

        self.keyheatmap = LeafmapModel(out_dim=1, name='keyheatmap')
        self.sizes = LeafmapModel(out_dim=2, mid_dim=16, name='sizes')
        self.offsets = LeafmapModel(out_dim=2, mid_dim=16, name='offsets')
        self.textline = LeafmapModel(out_dim=1, mid_dim=8, name='textline')
        self.sepatator = LeafmapModel(out_dim=1, mid_dim=8, name='sepatator')
        self.codes = LeafmapModel(out_dim=4, name='codes')
        self.feature = LeafmapModel(out_dim=feature_dim, name='feature')

    def call(self, inputs, **kwarg):
        backbone_out = self.backbone(inputs)
        backbone_out = [bn(x) for (bn, x) in zip(self.bn, backbone_out)]

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
        # if hasattr(layers, 'weights'):
        #     print(layers.weights)

    print_layers(model)
