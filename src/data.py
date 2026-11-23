"""Fixed CIFAR-10 split and on-demand image preparation."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


CIFAR10_GOOGLE_MIRROR = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "cifar-10-python.tar.gz"
)
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def load_cifar_and_split(output_dir="artifacts"):
    # Keras 3.14.1 downloads from the slow Toronto host. Prime its exact cache
    # entry from the Google-hosted copy used in newer Keras code, with the same
    # official SHA-256 integrity check, before calling the standard loader.
    keras.utils.get_file(
        fname="cifar-10-batches-py-target",
        origin=CIFAR10_GOOGLE_MIRROR,
        extract=True,
        file_hash=CIFAR10_SHA256,
    )
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1).astype("int64")
    y_test = y_test.reshape(-1).astype("int64")
    all_indices = np.arange(len(y_train))
    train_idx, remainder = train_test_split(
        all_indices, train_size=12000, random_state=42, stratify=y_train)
    val_idx, _ = train_test_split(
        remainder, train_size=3000, random_state=42,
        stratify=y_train[remainder])
    assert len(train_idx) == 12000 and len(val_idx) == 3000
    assert not set(train_idx).intersection(val_idx)
    assert len(x_test) == 10000
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "split_indices.npz", train=train_idx, val=val_idx)
    return (x_train, y_train), (x_test, y_test), train_idx, val_idx


def make_dataset(images, labels, indices, size, batch_size, *, training=False,
                 augment=False, transfer=False, seed=42):
    """Resize per batch in tf.data; never keep resized copies in RAM."""
    if indices is not None:
        images, labels = images[indices], labels[indices]
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(min(len(images), 12000), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.enumerate()

    def prepare(index, sample):
        image, label = sample
        image = tf.image.resize(tf.cast(image, tf.float32), (size, size))
        if transfer:
            # ResNet50 preprocess_input requires original [0, 255] scale.
            image = keras.applications.resnet50.preprocess_input(image)
        else:
            image = image / 255.0
            if augment and training:
                random_seed = tf.stack([tf.cast(seed, tf.int32),
                                        tf.cast(index, tf.int32)])
                image = tf.image.stateless_random_flip_left_right(
                    image, seed=random_seed)
                padding = max(1, round(size * 0.08))
                image = tf.pad(image, [[padding, padding], [padding, padding],
                                       [0, 0]], mode="REFLECT")
                image = tf.image.stateless_random_crop(
                    image, [size, size, 3], seed=random_seed + [0, 1])
                image = tf.image.stateless_random_contrast(
                    image, 0.9, 1.1, seed=random_seed + [0, 2])
                image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
