import tensorflow as tf

import copy

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import layer_utils
from keras.applications.efficientnet_v2 import round_filters, CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER, MBConvBlock, FusedMBConvBlock, round_repeats

width = 512
height = 512
scale = 2

feature_dim = 384

modulo_list = [37,41,43,47,53]

def EfficientNetV2(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args="default",
    model_name="efficientnetv2",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
):
  """Instantiates the EfficientNetV2 architecture using given scaling coefficients.
  Args:
    width_coefficient: float, scaling coefficient for network width.
    depth_coefficient: float, scaling coefficient for network depth.
    default_size: integer, default input image size.
    dropout_rate: float, dropout rate before final classifier layer.
    drop_connect_rate: float, dropout rate at skip connections.
    depth_divisor: integer, a unit of network width.
    min_depth: integer, minimum number of filters.
    bn_momentum: float. Momentum parameter for Batch Normalization layers.
    activation: activation function.
    blocks_args: list of dicts, parameters to construct block modules.
    model_name: string, model name.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    weights: one of `None` (random initialization), `"imagenet"` (pre-training
      on ImageNet), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) or
      numpy array to use as image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is
      False. It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction when `include_top` is
      `False`. - `None` means that the output of the model will be the 4D tensor
      output of the last convolutional layer. - "avg" means that global average
      pooling will be applied to the output of the last convolutional layer, and
      thus the output of the model will be a 2D tensor. - `"max"` means that
      global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True, and if no `weights` argument is
      specified.
    classifier_activation: A string or callable. The activation function to use
      on the `"top"` layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the `"top"` layer.
    include_preprocessing: Boolean, whether to include the preprocessing layer
      (`Rescaling`) at the bottom of the network. Defaults to `True`.
  Returns:
    A `keras.Model` instance.
  Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `"softmax"` or `None` when
      using a pretrained top layer.
  """

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

  x = img_input

  if include_preprocessing:
    # Apply original V1 preprocessing for Bx variants
    # if number of channels allows it
    num_channels = input_shape[bn_axis - 1]
    if model_name.split("-")[-1].startswith("b") and num_channels == 3:
      x = layers.Rescaling(scale=1. / 255)(x)
      x = layers.Normalization(
          mean=[0.485, 0.456, 0.406],
          variance=[0.229**2, 0.224**2, 0.225**2],
          axis=bn_axis,
      )(x)
    else:
      x = layers.Rescaling(scale=1. / 128.0, offset=-1)(x)

  # Build stem
  stem_filters = round_filters(
      filters=blocks_args[0]["input_filters"],
      width_coefficient=width_coefficient,
      min_depth=min_depth,
      depth_divisor=depth_divisor,
  )
  x = layers.Conv2D(
      filters=stem_filters,
      kernel_size=3,
      strides=2,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      use_bias=False,
      name="stem_conv",
  )(x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="stem_bn",
  )(x)
  x = layers.Activation(activation, name="stem_activation")(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)
  b = 0
  blocks = float(sum(args["num_repeat"] for args in blocks_args))

  for (i, args) in enumerate(blocks_args):
    assert args["num_repeat"] > 0

    # Update block input and output filters based on depth multiplier.
    args["input_filters"] = round_filters(
        filters=args["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor)
    args["output_filters"] = round_filters(
        filters=args["output_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor)

    # Determine which conv type to use:
    block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
    repeats = round_repeats(
        repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient)
    for j in range(repeats):
      # The first block needs to take care of stride and filter size increase.
      if j > 0:
        args["strides"] = 1
        args["input_filters"] = args["output_filters"]

      c_str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
      x = block(
          activation=activation,
          bn_momentum=bn_momentum,
          survival_probability=drop_connect_rate * b / blocks,
          name="block{}{}_".format(i + 1, c_str[j]),
          **args,
      )(x)

  # Build top
  top_filters = round_filters(
      filters=1280,
      width_coefficient=width_coefficient,
      min_depth=min_depth,
      depth_divisor=depth_divisor)
  x = layers.Conv2D(
      filters=top_filters,
      kernel_size=1,
      strides=1,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      data_format="channels_last",
      use_bias=False,
      name="top_conv",
  )(x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="top_bn",
  )(x)
  x = layers.Activation(activation=activation, name="top_activation")(x)

  if include_top:
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        bias_initializer=tf.constant_initializer(0),
        name="predictions")(x)
  else:
    if pooling == "avg":
      x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
      x = layers.GlobalMaxPooling2D(name="max_pool")(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name=model_name)

  return model

class BackboneModel(tf.keras.Model):
    def __init__(self, pre_weight=True, renorm=False, syncbn=False, **kwarg):
        super().__init__(**kwarg)
        blocks_args=[
            {
                "kernel_size": 3,
                "num_repeat": 4,
                "input_filters": 32,
                "output_filters": 32,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 32,
                "output_filters": 64,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 64,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 16,
                "input_filters": 96,
                "output_filters": 192,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 24,
                "input_filters": 192,
                "output_filters": 256,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 32,
                "input_filters": 256,
                "output_filters": 512,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 512,
                "output_filters": 640,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
        ]        
        base_model = EfficientNetV2(
            width_coefficient=1.0,
            depth_coefficient=1.0,
            default_size=480,
            model_name="efficientnetv2-xl",
            include_top=False,
            weights=None,
            blocks_args=blocks_args,
            )


        outlayers = [layer for layer in base_model.layers if layer.name.endswith('_add')]
        mid_output3 = [layer for layer in outlayers if 'block2' in layer.name][-1]
        mid_output2 = [layer for layer in outlayers if 'block3' in layer.name][-1]
        mid_output1 = [layer for layer in outlayers if 'block5' in layer.name][-1]
        self.extract_model = tf.keras.Model(base_model.input, [mid_output3.output, mid_output2.output, mid_output1.output, base_model.output], name='efficientnetv2-xl')

        if syncbn:
            import horovod.tensorflow as hvd
            model_config = self.extract_model.get_config()
            for layer, layer_config in zip(self.extract_model.layers, model_config['layers']):
                if type(layer) == tf.keras.layers.BatchNormalization:
                    layer_config['class_name'] = 'SyncBatchNormalization'
            self.extract_model = tf.keras.models.Model.from_config(model_config, custom_objects={'SyncBatchNormalization': hvd.SyncBatchNormalization})
        elif renorm:
            model_config = self.extract_model.get_config()
            for layer, layer_config in zip(self.extract_model.layers, model_config['layers']):
                if type(layer) == tf.keras.layers.BatchNormalization:
                    layer_config['config']['renorm'] = True
            self.extract_model = tf.keras.models.Model.from_config(model_config)

        if pre_weight:
            path_to_downloaded_file = tf.keras.utils.get_file(
                'efficientnetv2-xl-21k-ft1k', 
                'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-xl-21k-ft1k.tgz',
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
                    basename = 'efficientnetv2-xl/stem/'
                elif b < 101:
                    basename = 'efficientnetv2-xl/blocks_%d/'%(b-1)
                else:
                    basename = 'efficientnetv2-xl/head/'

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
                        if renorm and k == 'moving_stddev':
                            key = n+'/'+'moving_variance'
                            tensor = reader.get_tensor(key)
                            tensor = tf.sqrt(tensor)
                            v.assign(tensor)
                        elif renorm and k == 'renorm_mean':
                            key = n+'/'+'moving_mean'
                            tensor = reader.get_tensor(key)
                            v.assign(tensor)
                        elif renorm and k == 'renorm_stddev':
                            key = n+'/'+'moving_variance'
                            tensor = reader.get_tensor(key)
                            tensor = tf.sqrt(tensor)
                            v.assign(tensor)
                        else:
                            key = n+'/'+k
                            #print(key, variable_map[key])
                                
                            tensor = reader.get_tensor(key)
                            v.assign(tensor)

        #print([layer.weights for layer in self.extract_model.layers])

    def call(self, inputs, **kwargs):
        x = inputs
        mid_output3,mid_output2,mid_output,final_output = self.extract_model(x, **kwargs)

        return mid_output3,mid_output2,mid_output,final_output

class LeafmapModel(tf.keras.Model):
    def __init__(self, out_dim=1, conv_dim=32, renorm=False, syncbn=False, **kwarg):
        super().__init__(**kwarg)
        if syncbn:
            import horovod.tensorflow as hvd
            BatchNormalization = hvd.SyncBatchNormalization
        else:
            BatchNormalization = tf.keras.layers.BatchNormalization

        self.conv1 = []
        momentum = 0.9
        for _ in range(3):
            self.conv1.append([
                tf.keras.layers.Conv2DTranspose(conv_dim, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer="he_normal"),
                BatchNormalization(momentum=momentum, renorm=renorm),
                tf.keras.layers.Activation('swish'),
            ])
        self.conv2 = [
            tf.keras.layers.Conv2DTranspose(conv_dim, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer="he_normal"),
            BatchNormalization(momentum=momentum, renorm=renorm),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(conv_dim, kernel_size=3, padding='same', kernel_initializer="he_normal"),
            tf.keras.layers.Activation('swish'),
        ]

        self.convout = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding='same', kernel_initializer="he_normal")
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs, **kwargs):
        P2_in,P3_in,P4_in,P5_in = inputs
        x = P5_in
        for x_in, conv1 in zip([P4_in,P3_in,P2_in], self.conv1):
            for layer in conv1:
                x = layer(x, **kwargs)

            x = tf.concat([x_in, x], axis=-1)

        for layer in self.conv2:
            x = layer(x, **kwargs)
        
        x = self.convout(x, **kwargs)
        x = self.outfloat32(x, **kwargs)
        return x

class ClassModuloModel(tf.keras.Model):
    def __init__(self, out_dim=97, **kwarg):
        super().__init__(**kwarg)
        self.dense_layers = [
            tf.keras.layers.Dense(4096, kernel_initializer="he_normal"),
            tf.keras.layers.Activation('swish'),
        ]
        self.dense_out = tf.keras.layers.Dense(out_dim, kernel_initializer="he_normal")
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x, **kwargs)
        x = self.dense_out(x, **kwargs)
        x = self.outfloat32(x, **kwargs)
        return x

def freeze_bn(layer):
    if hasattr(layer, 'layers'):
        for layer in layer.layers:
            freeze_bn(layer)    
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

def CenterNetDetectionBlock(pre_weight=True, renorm=False, syncbn=False):
    backbone = BackboneModel(pre_weight=pre_weight, renorm=renorm, syncbn=syncbn, name='BackBoneNet')

    keyheatmap = LeafmapModel(out_dim=1, name='keyheatmap', renorm=renorm, syncbn=syncbn)
    sizes = LeafmapModel(out_dim=2, name='sizes', renorm=renorm, syncbn=syncbn)
    offsets = LeafmapModel(out_dim=2, name='offsets', renorm=renorm, syncbn=syncbn)
    textline = LeafmapModel(out_dim=1, name='textline', renorm=renorm, syncbn=syncbn)
    sepatator = LeafmapModel(out_dim=1, name='sepatator', renorm=renorm, syncbn=syncbn)
    feature = LeafmapModel(out_dim=feature_dim, conv_dim=512, name='feature', renorm=renorm, syncbn=syncbn)

    inputs = tf.keras.Input(shape=(height,width,3))
    backbone_out = backbone(inputs)
    maps = tf.keras.layers.Concatenate(dtype='float32')([
        keyheatmap(backbone_out), 
        sizes(backbone_out), 
        offsets(backbone_out), 
        textline(backbone_out), 
        sepatator(backbone_out),
    ])
    outputs = [maps, feature(backbone_out)]
    return tf.keras.Model(inputs, outputs, name='CenterNetBlock')

def SimpleDecoderBlock():
    embedded = tf.keras.Input(shape=(feature_dim,))

    outputs = []
    for modulo in modulo_list:
        dense = ClassModuloModel(out_dim=modulo, name='modulo%d'%modulo)(embedded)
        outputs.append(dense)

    return tf.keras.Model(embedded, outputs, name='SimpleDecoderBlock')

if __name__ == '__main__':
    model = CenterNetDetectionBlock()
    model.summary()

    decoder = SimpleDecoderBlock()
    decoder.summary()
