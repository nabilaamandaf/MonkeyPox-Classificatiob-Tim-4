import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# KONFIGURASI PATH & PARAMETER

PROJECT_DIR = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "Data", "archive")



ORIG_FOLDS_DIR = os.path.join(
    ARCHIVE_DIR, "Original Images", "Original Images", "FOLDS"
)

# Augmented Images\Augmented Images\fold1_AUG..fold5_AUG 
AUG_FOLDS_DIR = os.path.join(
    ARCHIVE_DIR, "Augmented Images", "Augmented Images"
)

MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6
SEED = 42
USE_AUG_TRAIN = False  

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# BUILDER MODEL MOBILENETV2 (SKALA [-1, 1])

def build_mobilenet_v2(input_shape=(128, 128, 3), num_classes=6):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



#PELATIHAN PER FOLD 1- 5

for fold in range(1, 6):
    train_dir = os.path.join(ORIG_FOLDS_DIR, f"fold{fold}", "Train")
    val_dir = os.path.join(ORIG_FOLDS_DIR, f"fold{fold}", "Valid")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    if USE_AUG_TRAIN:
        aug_train_dir = os.path.join(AUG_FOLDS_DIR, f"fold{fold}_AUG", "Train")
        if os.path.isdir(aug_train_dir):
            aug_ds = tf.keras.utils.image_dataset_from_directory(
                aug_train_dir,
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode="int",
                shuffle=True,
                seed=SEED,
            )
            train_ds = train_ds.concatenate(aug_ds).shuffle(1000, seed=SEED)

    # shuffle = False agar urutan data validasi tetap konsisten
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    print(f"\n=== Melatih MobileNetV2 - Fold {fold} ===")
    model = build_mobilenet_v2(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    save_path = os.path.join(MODELS_DIR, f"mobilenetv2_fold{fold}.h5")
    model.save(save_path)
    print(f"[SAVED] Model tersimpan di: {save_path}")