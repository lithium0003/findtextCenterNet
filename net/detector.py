import tensorflow as tf

import copy

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import layer_utils
from keras.applications.efficientnet_v2 import round_filters, CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER, MBConvBlock, FusedMBConvBlock, round_repeats

from keras.layers.normalization.batch_normalization import BatchNormalizationBase

width = 512
height = 512
scale = 2

feature_dim = 128

modulo_list = [1091,1093,1097]

class AverageBatchNormalization(BatchNormalizationBase):
    def __init__(self,
                axis=-1,
                momentum=0.99,
                average=1,
                epsilon=1e-3,
                center=True,
                scale=True,
                beta_initializer='zeros',
                gamma_initializer='ones',
                moving_mean_initializer='zeros',
                moving_variance_initializer='ones',
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                **kwargs):
        kwargs.pop('fused', None)
        self.average = average
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            fused=False,
            **kwargs)    

    def get_config(self):
        config = super().get_config()
        config['average'] = self.average
        return config

    def build(self, input_shape):
        super().build(input_shape)

        self.averaged_mean = tf.Variable(self.moving_mean, trainable=False, name='averaged_mean', dtype=self._param_dtype)
        self.averaged_squared_mean = tf.Variable(self.moving_variance + tf.square(self.moving_mean), trainable=False, name='averaged_squared_mean', dtype=self._param_dtype)

        self.backbuf_idx = 0
        self.backbuf_mean = [tf.Variable(self.moving_mean, trainable=False, dtype=self._param_dtype) for _ in range(self.average)]
        self.backbuf_squared_mean = [tf.Variable(self.moving_variance + tf.square(self.moving_mean), trainable=False, dtype=self._param_dtype) for _ in range(self.average)]

    def finalize_state(self):
        self.averaged_mean.assign(self.moving_mean)
        self.averaged_squared_mean.assign(self.moving_variance + tf.square(self.moving_mean))

        self.backbuf_idx = 0
        for i in range(self.average):
            self.backbuf_mean[i].assign(self.moving_mean)
            self.backbuf_squared_mean[i].assign(self.moving_variance + tf.square(self.moving_mean))

    def _calculate_mean_and_var(self, x, axes, keep_dims):
        # The dynamic range of fp16 is too limited to support the collection of
        # sufficient statistics. As a workaround we simply perform the operations
        # on 32-bit floats before converting the mean and variance back to fp16
        y = tf.cast(x, tf.float32) if x.dtype == tf.float16 else x
        local_mean = tf.reduce_mean(y, axis=axes)
        prev_mean = self.backbuf_mean[self.backbuf_idx]
        self.averaged_mean.assign_sub(prev_mean / self.average)
        self.averaged_mean.assign_add(local_mean / self.average)
        self.backbuf_mean[self.backbuf_idx].assign(local_mean)
        local_squared_mean = tf.reduce_mean(tf.square(y), axis=axes)
        prev_squared_mean = self.backbuf_squared_mean[self.backbuf_idx]
        self.averaged_squared_mean.assign_sub(prev_squared_mean / self.average)
        self.averaged_squared_mean.assign_add(local_squared_mean / self.average)
        self.backbuf_squared_mean[self.backbuf_idx].assign(local_squared_mean)
        self.backbuf_idx += 1
        if self.backbuf_idx >= self.average:
            self.backbuf_idx = 0

        mean = self.averaged_mean
        variance = self.averaged_squared_mean - tf.square(mean)

        if x.dtype == tf.float16:
            return (tf.cast(mean, tf.float16),
                    tf.cast(variance, tf.float16))
        else:
            return (mean, variance)

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
    def __init__(self, pre_weight=True, average=1, syncbn=False, **kwarg):
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
        mid_output0 = [layer for layer in outlayers if 'block7' in layer.name][-1]
        self.extract_model = tf.keras.Model(base_model.input, [mid_output3.output, mid_output2.output, mid_output1.output, mid_output0.output], name='efficientnetv2-xl')

        mbn = False
        if syncbn:
            import horovod.tensorflow as hvd
            model_config = self.extract_model.get_config()
            for layer, layer_config in zip(self.extract_model.layers, model_config['layers']):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer_config['class_name'] = 'SyncBatchNormalization'
            del self.extract_model
            self.extract_model = tf.keras.models.Model.from_config(model_config, custom_objects={'SyncBatchNormalization': hvd.SyncBatchNormalization})
        elif average > 1:
            model_config = self.extract_model.get_config()
            for layer, layer_config in zip(self.extract_model.layers, model_config['layers']):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer_config['class_name'] = 'AverageBatchNormalization'
                    layer_config['config']['average'] = average
            del self.extract_model
            self.extract_model = tf.keras.models.Model.from_config(model_config, custom_objects={'AverageBatchNormalization': AverageBatchNormalization})
            mbn = True

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
                        if mbn and k == 'averaged_mean':
                            key = n+'/'+'moving_mean'
                            tensor = reader.get_tensor(key)
                            v.assign(tensor)
                        elif mbn and k == 'averaged_squared_mean':
                            key1 = n+'/'+'moving_variance'
                            tensor1 = reader.get_tensor(key1)
                            key2 = n+'/'+'moving_mean'
                            tensor2 = reader.get_tensor(key2)
                            tensor = tensor1 + tf.square(tensor2)
                            v.assign(tensor)
                        elif mbn and k == 'Variable':
                            pass
                        else:
                            key = n+'/'+k
                            #print(key, variable_map[key])
                                
                            tensor = reader.get_tensor(key)
                            v.assign(tensor)

            self.finalize_state()

        #print([layer.weights for layer in self.extract_model.layers])

    def finalize_state(self):
        for layer in self.extract_model.layers:
            layer.finalize_state()

    def call(self, inputs, **kwargs):
        x = inputs
        mid_output3,mid_output2,mid_output,final_output = self.extract_model(x, **kwargs)

        return mid_output3,mid_output2,mid_output,final_output

class LeafmapModel(tf.keras.Model):
    def __init__(self, out_dim=1, conv_dim=8, **kwarg):
        super().__init__(**kwarg)

        self.topconv = [
            tf.keras.layers.Conv2D(conv_dim, kernel_size=1, padding='same', activation='swish', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal'), name=self.name+'_top_conv'),
        ]

        self.conv1 = []
        #conv_dims = [64,96,256,640]
        conv_dims = [8,12,32,80]
        for i in range(4):
            self.conv1.append([
                tf.keras.layers.Conv2D(conv_dims[i], kernel_size=1, padding='same', activation='swish', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal'), name=self.name+'_%s_conv'%('abcd'[i])),
                tf.keras.layers.UpSampling2D(size=2**(i+1), interpolation='bilinear', name=self.name+'_%s_up'%('abcd'[i]))
            ])
        
        self.convout = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), name=self.name+'_out_conv')
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs, **kwarg):
        P2_in,P3_in,P4_in,P5_in = inputs
        x = []
        for x_in, conv in zip([P2_in,P3_in,P4_in,P5_in], self.conv1):
            x1 = x_in
            for layer in conv:
                x1 = layer(x1, **kwarg)
            x.append(x1)
        x = tf.concat(x, axis=-1)
        for layer in self.topconv:
            x = layer(x, **kwarg)

        x = self.convout(x)
        x = self.outfloat32(x)
        return x

class ClassModuloModel(tf.keras.Model):
    def __init__(self, out_dim=97, **kwarg):
        super().__init__(**kwarg)
        self.dense_layers = [
            tf.keras.layers.Dense(1024, name=self.name+'dense_1', activation='swish', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='truncated_normal')),
        ]
        self.dense_out = tf.keras.layers.Dense(out_dim, name=self.name+'dense_out', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal'))
        self.outfloat32 = tf.keras.layers.Activation('linear', dtype='float32')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dense_out(x)
        x = self.outfloat32(x)
        return x

class CenterNetDetectionModel(tf.keras.Model):
    def __init__(self, pre_weight=True, average=1, syncbn=False, lockBase=False, **kwarg):
        super().__init__(**kwarg)
        self.backbone = BackboneModel(pre_weight=pre_weight, average=average, syncbn=syncbn, name='BackBoneNet')

        if lockBase:
            self.backbone.trainable = False

        self.keyheatmap = LeafmapModel(out_dim=1, name='keyheatmap')
        self.sizes = LeafmapModel(out_dim=2, name='sizes')
        self.offsets = LeafmapModel(out_dim=2, name='offsets')
        self.textline = LeafmapModel(out_dim=1, name='textline')
        self.sepatator = LeafmapModel(out_dim=1, name='sepatator')
        self.feature = LeafmapModel(out_dim=feature_dim, conv_dim=256, name='feature')

    def call(self, inputs):
        backbone_out = self.backbone(inputs)
        maps = tf.keras.layers.Concatenate(dtype='float32')([
            self.keyheatmap(backbone_out), 
            self.sizes(backbone_out), 
            self.offsets(backbone_out), 
            self.textline(backbone_out), 
            self.sepatator(backbone_out),
        ])
        return maps, self.feature(backbone_out)

def CenterNetDetectionBlock(pre_weight=True, average=1, syncbn=False, lockBase=False):
    return CenterNetDetectionModel(pre_weight=pre_weight, average=average, syncbn=syncbn, lockBase=lockBase, name='CenterNetBlock')

def SimpleDecoderBlock():
    embedded = tf.keras.Input(shape=(feature_dim,))

    outputs = []
    for modulo in modulo_list:
        dense = ClassModuloModel(out_dim=modulo, name='modulo%d'%modulo)(embedded)
        outputs.append(dense)

    return tf.keras.Model(embedded, outputs, name='SimpleDecoderBlock')

if __name__ == '__main__':
    model = CenterNetDetectionBlock()
    inputs = tf.keras.Input(shape=(height,width,3))
    outputs = model(inputs)
    model.summary()

    decoder = SimpleDecoderBlock()
    decoder.summary()
