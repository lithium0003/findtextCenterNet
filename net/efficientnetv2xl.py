import tensorflow as tf

import copy

from tensorflow.python.keras import backend
from keras.src.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from keras.src.applications.efficientnet_v2 import round_filters, CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER, MBConvBlock, FusedMBConvBlock, round_repeats

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
    img_input = tf.keras.layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

  x = img_input

  if include_preprocessing:
    # Apply original V1 preprocessing for Bx variants
    # if number of channels allows it
    num_channels = input_shape[bn_axis - 1]
    if model_name.split("-")[-1].startswith("b") and num_channels == 3:
      x = tf.keras.layers.Rescaling(scale=1. / 255)(x)
      x = tf.keras.layers.Normalization(
          mean=[0.485, 0.456, 0.406],
          variance=[0.229**2, 0.224**2, 0.225**2],
          axis=bn_axis,
      )(x)
    else:
      x = tf.keras.layers.Rescaling(scale=1. / 128.0, offset=-1)(x)

  # Build stem
  stem_filters = round_filters(
      filters=blocks_args[0]["input_filters"],
      width_coefficient=width_coefficient,
      min_depth=min_depth,
      depth_divisor=depth_divisor,
  )
  x = tf.keras.layers.Conv2D(
      filters=stem_filters,
      kernel_size=3,
      strides=2,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      use_bias=False,
      name="stem_conv",
  )(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="stem_bn",
  )(x)
  x = tf.keras.layers.Activation(activation, name="stem_activation")(x)

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
  x = tf.keras.layers.Conv2D(
      filters=top_filters,
      kernel_size=1,
      strides=1,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      data_format="channels_last",
      use_bias=False,
      name="top_conv",
  )(x)
  x = tf.keras.layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="top_bn",
  )(x)
  x = tf.keras.layers.Activation(activation=activation, name="top_activation")(x)

  if include_top:
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
      x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = tf.keras.layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        bias_initializer=tf.constant_initializer(0),
        name="predictions")(x)
  else:
    if pooling == "avg":
      x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
      x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = tf.keras.Model(inputs, x, name=model_name)

  return model

def EfficientNetV2XL():
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
        return EfficientNetV2(
            width_coefficient=1.0,
            depth_coefficient=1.0,
            default_size=480,
            model_name="efficientnetv2-xl",
            include_top=False,
            weights=None,
            blocks_args=blocks_args,
            )
