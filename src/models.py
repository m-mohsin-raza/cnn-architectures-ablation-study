"""Assignment 02 CNNs. All classifiers return 10 float32 logits."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _logits(x, name="logits"):
    return layers.Dense(10, dtype="float32", name=name)(x)


def build_alexnet(input_shape=(96, 96, 3)):
    """Five historical conv stages; LRN and grouped convolutions are omitted."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for name, filters, kernel, stride, pool in [
        ("conv1", 96, 11, 4, True),
        ("conv2", 256, 5, 1, True),
        ("conv3", 384, 3, 1, False),
        ("conv4", 384, 3, 1, False),
        ("conv5", 256, 3, 1, True),
    ]:
        x = layers.Conv2D(filters, kernel, strides=stride, padding="same",
                          activation="relu", name=name)(x)
        if pool:
            x = layers.MaxPooling2D(3, strides=2, padding="same",
                                    name=f"{name}_pool")(x)
    x = layers.Flatten(name="flatten")(x)
    for n in (1, 2):
        x = layers.Dense(4096, activation="relu", name=f"fc{n}")(x)
        x = layers.Dropout(0.5, name=f"dropout{n}")(x)
    return keras.Model(inputs, _logits(x), name="AlexNet_96")


def build_vgg16(input_shape=(96, 96, 3), gap_head=False):
    """Manual VGG16; gap_head is for parameter analysis only, not benchmark training."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for block, (count, width) in enumerate(zip((2, 2, 3, 3, 3),
                                                (64, 128, 256, 512, 512)), 1):
        for i in range(1, count + 1):
            x = layers.Conv2D(width, 3, padding="same", activation="relu",
                              name=f"block{block}_conv{i}")(x)
        x = layers.MaxPooling2D(2, strides=2, name=f"block{block}_pool")(x)
    if gap_head:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(4096, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        x = layers.Dense(4096, activation="relu", name="fc2")(x)
        x = layers.Dropout(0.5, name="dropout2")(x)
    return keras.Model(inputs, _logits(x),
                       name="VGG16_GAP_analytic" if gap_head else "VGG16_96")


def _inception(x, name, c1, c3r, c3, c5r, c5, pool_proj):
    b1 = layers.Conv2D(c1, 1, padding="same", activation="relu",
                       name=f"{name}_1x1")(x)
    b2 = layers.Conv2D(c3r, 1, padding="same", activation="relu",
                       name=f"{name}_3x3_reduce")(x)
    b2 = layers.Conv2D(c3, 3, padding="same", activation="relu",
                       name=f"{name}_3x3")(b2)
    b3 = layers.Conv2D(c5r, 1, padding="same", activation="relu",
                       name=f"{name}_5x5_reduce")(x)
    b3 = layers.Conv2D(c5, 5, padding="same", activation="relu",
                       name=f"{name}_5x5")(b3)
    b4 = layers.MaxPooling2D(3, strides=1, padding="same",
                             name=f"{name}_pool")(x)
    b4 = layers.Conv2D(pool_proj, 1, padding="same", activation="relu",
                       name=f"{name}_pool_proj")(b4)
    spatial = [tuple(b.shape[1:3]) for b in (b1, b2, b3, b4)]
    assert len(set(spatial)) == 1, f"{name}: branch shapes differ: {spatial}"
    return layers.Concatenate(axis=-1, name=f"{name}_concat")([b1, b2, b3, b4])


def build_googlenet(input_shape=(96, 96, 3)):
    """Inception-v1 stem and nine modules; auxiliary heads are omitted."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu",
                      name="stem_7x7")(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool1")(x)
    x = layers.Conv2D(64, 1, padding="same", activation="relu",
                      name="stem_1x1")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="relu",
                      name="stem_3x3")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool2")(x)
    specs = [
        ("3a", 64, 96, 128, 16, 32, 32),
        ("3b", 128, 128, 192, 32, 96, 64),
        ("4a", 192, 96, 208, 16, 48, 64),
        ("4b", 160, 112, 224, 24, 64, 64),
        ("4c", 128, 128, 256, 24, 64, 64),
        ("4d", 112, 144, 288, 32, 64, 64),
        ("4e", 256, 160, 320, 32, 128, 128),
        ("5a", 256, 160, 320, 32, 128, 128),
        ("5b", 384, 192, 384, 48, 128, 128),
    ]
    for spec in specs:
        name = spec[0]
        if name in ("4a", "5a"):
            x = layers.MaxPooling2D(3, strides=2, padding="same",
                                    name=f"before_{name}_pool")(x)
        x = _inception(x, f"inception_{name}", *spec[1:])
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    return keras.Model(inputs, _logits(x), name="GoogLeNet_96")


def _basic_block(x, filters, stride, name):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                      name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                 name=f"{name}_projection")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)
    assert tuple(y.shape[1:]) == tuple(shortcut.shape[1:]), \
        f"{name}: Add shape mismatch: {y.shape} versus {shortcut.shape}"
    return layers.ReLU(name=f"{name}_out")(
        layers.Add(name=f"{name}_add")([y, shortcut]))


def build_resnet18(input_shape=(96, 96, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                      name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    for stage, filters in enumerate((64, 128, 256, 512), 1):
        for block in (1, 2):
            stride = 2 if stage > 1 and block == 1 else 1
            x = _basic_block(x, filters, stride, f"stage{stage}_block{block}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    return keras.Model(inputs, _logits(x), name="ResNet18_96")


def build_ablation(variant, input_shape=(64, 64, 3)):
    """A0, A1, A2, A3-P, A3-R share widths and a 10-logit head."""
    if variant not in ("A0", "A1", "A2", "A3-P", "A3-R"):
        raise ValueError(f"Unknown variant: {variant}")
    inputs = keras.Input(shape=input_shape)
    x = inputs
    use_bn = variant != "A0"
    deep = variant in ("A3-P", "A3-R")
    residual = variant == "A3-R"
    for stage, filters in enumerate((32, 64), 1):
        shortcut = x
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not use_bn,
                          name=f"stage{stage}_conv1")(x)
        if use_bn:
            x = layers.BatchNormalization(name=f"stage{stage}_bn1")(x)
        x = layers.ReLU(name=f"stage{stage}_relu1")(x)
        if deep:
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              name=f"stage{stage}_conv2")(x)
            x = layers.BatchNormalization(name=f"stage{stage}_bn2")(x)
            if residual:
                shortcut = layers.Conv2D(filters, 1, use_bias=False,
                                         name=f"stage{stage}_projection")(shortcut)
                shortcut = layers.BatchNormalization(
                    name=f"stage{stage}_projection_bn")(shortcut)
                assert tuple(x.shape[1:]) == tuple(shortcut.shape[1:])
                x = layers.Add(name=f"stage{stage}_add")([x, shortcut])
            x = layers.ReLU(name=f"stage{stage}_relu2")(x)
        x = layers.MaxPooling2D(2, name=f"stage{stage}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    return keras.Model(inputs, _logits(x), name=variant.replace("-", "_"))


def build_transfer(input_shape=(96, 96, 3), fine_tune=False):
    """ResNet50 receives externally preprocessed 0-255-scale images."""
    inputs = keras.Input(shape=input_shape)
    backbone = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                           input_shape=input_shape)
    backbone.trainable = bool(fine_tune)
    if fine_tune:
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        for layer in backbone.layers[-20:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    return keras.Model(inputs, _logits(x),
                       name="A4_T" if fine_tune else "A4_F")


BUILDERS = {
    "AlexNet": build_alexnet,
    "VGG16": build_vgg16,
    "GoogLeNet": build_googlenet,
    "ResNet18": build_resnet18,
}
