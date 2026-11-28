import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================
# 1. KONFIGURASI UTAMA
# ==========================================
BASE_DIR = "dataset"          # Sesuaikan dengan folder utama dataset
AUG_DIR = "dataset_aug"       # Sesuaikan jika folder augmentasi terpisah (opsional)
OUTPUT_DIR = "outputs_5fold"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6
SEED = 42

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# ==========================================
# 2. DEFINISI ARSITEKTUR MODEL
# ==========================================
def build_simple_cnn(input_shape=(128, 128, 3), num_classes=6):
    # Normalisasi [0, 1] dilakukan lewat Rescaling(1./255)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_modified_cnn(input_shape=(128, 128, 3), num_classes=6):
    # Normalisasi [0, 1] + blok konvolusi bertingkat + Dropout
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_model(input_shape=(128, 128, 3), num_classes=6):
    # Skala [-1, 1] sesuai bobot pra-latih ImageNet
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


MODEL_BUILDERS = {
    "SimpleCNN": build_simple_cnn,
    "ModifiedCNN": build_modified_cnn,
    "MobileNetV2": build_mobilenet_model,
}


# ==========================================
# 3. HELPER MEMUAT DATA PER FOLD
# ==========================================
def load_fold_datasets(fold_idx):
    # Sesuaikan pola nama folder jika milikmu memakai 'fold1' atau 'fold_1'
    fold_train_dir = os.path.join(BASE_DIR, f"fold{fold_idx}", "Train")
    fold_val_dir = os.path.join(BASE_DIR, f"fold{fold_idx}", "Valid")
    fold_test_dir = os.path.join(BASE_DIR, f"fold{fold_idx}", "Test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        fold_train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    # Opsional: gabungkan data augmentasi khusus fold ini bila tersedia
    fold_aug_dir = os.path.join(AUG_DIR, f"fold{fold_idx}_AUG", "Train")
    if os.path.isdir(fold_aug_dir):
        aug_ds = tf.keras.utils.image_dataset_from_directory(
            fold_aug_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=True,
            seed=SEED,
        )
        train_ds = train_ds.concatenate(aug_ds).shuffle(1000, seed=SEED)

    # shuffle=False WAJIB saat validasi & test agar y_true sejajar dengan prediksi
    val_ds = tf.keras.utils.image_dataset_from_directory(
        fold_val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        fold_test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


# ==========================================
# 4. PELATIHAN & EVALUASI 5-FOLD MURNI
# ==========================================
per_fold_records = []
summary_records = []

for model_name, builder_fn in MODEL_BUILDERS.items():
    fold_test_accs = []
    fold_val_accs = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold in range(1, 6):
        train_ds, val_ds, test_ds = load_fold_datasets(fold)

        # Bangun ulang model dari awal di setiap fold (cegah kebocoran bobot antar-fold)
        model = builder_fn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=1,
        )

        # Simpan bobot model per fold
        model_path = os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold}.h5")
        model.save(model_path)

        # Evaluasi di Valid & Test pada fold yang sama
        _, val_acc = model.evaluate(val_ds, verbose=0)

        y_true_fold = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        y_prob_fold = model.predict(test_ds, verbose=0)
        y_pred_fold = np.argmax(y_prob_fold, axis=1)

        test_acc_fold = accuracy_score(y_true_fold, y_pred_fold)

        fold_val_accs.append(val_acc)
        fold_test_accs.append(test_acc_fold)
        all_y_true.append(y_true_fold)
        all_y_pred.append(y_pred_fold)
        all_y_prob.append(y_prob_fold)

        per_fold_records.append({
            "model": model_name,
            "fold": fold,
            "val_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc_fold), 4),
        })

    # Hitung metrik macro dari gabungan 5 fold test set & std test_accuracy antar-fold
    y_true_cat = np.concatenate(all_y_true, axis=0)
    y_pred_cat = np.concatenate(all_y_pred, axis=0)
    y_prob_cat = np.concatenate(all_y_prob, axis=0)

    mean_test_acc = float(np.mean(fold_test_accs))
    std_test_acc = float(np.std(fold_test_accs))

    macro_prec = float(precision_score(y_true_cat, y_pred_cat, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_cat, y_pred_cat, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0))
    macro_auc = float(roc_auc_score(y_true_cat, y_prob_cat, multi_class="ovr", average="macro"))

    summary_records.append({
        "model": model_name,
        "test_accuracy_mean": round(mean_test_acc, 4),
        "test_accuracy_std": round(std_test_acc, 4),
        "precision_macro": round(macro_prec, 4),
        "recall_macro": round(macro_rec, 4),
        "f1_macro": round(macro_f1, 4),
        "roc_auc_macro": round(macro_auc, 4),
    })

df_per_fold = pd.DataFrame(per_fold_records)
df_summary = pd.DataFrame(summary_records)

df_per_fold.to_csv(os.path.join(OUTPUT_DIR, "hasil_per_fold.csv"), index=False)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_5fold_macro.csv"), index=False)

print("=== RINGKASAN PER FOLD ===")
print(df_per_fold)
print("\n=== RINGKASAN RATA-RATA 5-FOLD (MACRO) ===")
print(df_summary)