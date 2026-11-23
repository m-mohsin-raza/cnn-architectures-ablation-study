"""Quick local check of model structure and one optimizer step.

Run from this project directory with the TensorFlow Python environment in WSL:
    python local_smoke_check.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models import BUILDERS, build_ablation


def check_model(name, builder, conv_count=None, dense_count=None, add_count=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = builder()
    logits = model(tf.zeros((2, 96, 96, 3)), training=False)
    assert tuple(logits.shape) == (2, 10), (name, logits.shape)
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    counts = {
        "conv": sum(isinstance(layer, layers.Conv2D) for layer in model.layers),
        "dense": sum(isinstance(layer, layers.Dense) for layer in model.layers),
        "add": sum(isinstance(layer, layers.Add) for layer in model.layers),
    }
    for key, expected in (("conv", conv_count), ("dense", dense_count),
                          ("add", add_count)):
        if expected is not None:
            assert counts[key] == expected, (name, key, counts[key], expected)
    first_conv = next(layer for layer in model.layers
                      if isinstance(layer, layers.Conv2D))
    kh, kw, cin, cout = [int(v) for v in first_conv.kernel.shape]
    manual = kh * kw * cin * cout + (cout if first_conv.use_bias else 0)
    assert manual == first_conv.count_params()
    print(f"{name}: PASS; logits={tuple(logits.shape)}, "
          f"params={model.count_params():,}, first conv={manual:,}, "
          f"layers={counts}")
    del model


print("TensorFlow", tf.__version__, "visible GPUs", tf.config.list_physical_devices("GPU"))
check_model("AlexNet", BUILDERS["AlexNet"], conv_count=5, dense_count=3)
check_model("VGG16", BUILDERS["VGG16"], conv_count=13, dense_count=3)
check_model("GoogLeNet", BUILDERS["GoogLeNet"], conv_count=57, dense_count=1)
check_model("ResNet18", BUILDERS["ResNet18"], conv_count=20,
            add_count=8, dense_count=1)

keras.backend.clear_session()
small = build_ablation("A0")
small.compile(optimizer=keras.optimizers.Adam(3e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
loss = small.train_on_batch(tf.random.uniform((2, 64, 64, 3)),
                            tf.constant([0, 1]))
assert float(loss) > 0 and float(loss) < 100
print(f"A0 optimizer step: PASS; finite loss={float(loss):.4f}")
