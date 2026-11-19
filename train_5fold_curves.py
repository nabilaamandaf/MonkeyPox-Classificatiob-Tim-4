import os
import tensorflow as tf
import matplotlib.pyplot as plt

# --- KONFIGURASI PATH DATASET ---
ORIG_DIR = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS"
AUG_DIR  = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Augmented Images\Augmented Images\FOLDS_AUG"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10  # Sesuai sumbu X pada contoh gambar (Epoch 1 - 10)
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]

def build_mobilenet_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_plot_5fold():
    histories = {}
    os.makedirs("Models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    for idx, fold in enumerate(FOLDS, start=1):
        print(f"\n--- Melatih {fold.upper()} ---")
        orig_train_path = os.path.join(ORIG_DIR, fold, "Train")
        aug_train_path  = os.path.join(AUG_DIR, fold, "Train")
        val_path        = os.path.join(ORIG_DIR, fold, "Valid")

        # Muat Original Train & Valid
        train_orig = tf.keras.utils.image_dataset_from_directory(
            orig_train_path, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
        )
        num_classes = len(train_orig.class_names)

        # Gabungkan dengan Augmented Train jika foldernya tersedia
        if os.path.exists(aug_train_path):
            train_aug = tf.keras.utils.image_dataset_from_directory(
                aug_train_path, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
            )
            train_ds = train_orig.concatenate(train_aug)
        else:
            train_ds = train_orig

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

        model = build_mobilenet_model(num_classes)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
        histories[f"Fold {idx}"] = history.history

        # Simpan model tiap fold
        model.save(f"Models/mobilenetv2_{fold}.h5")

    # --- MEMBUAT GRAFIK BERDAMPINGAN (LOSS & ACCURACY) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    epochs_range = range(1, EPOCHS + 1)
    colors = ["#d97e26", "#c02938", "#7c524b", "#787878", "#17b0bf"]

    for idx, (fold_label, hist) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        # Plot Kiri: Loss
        axes[0].plot(epochs_range, hist["loss"], linestyle="--", color=c, alpha=0.5, label=f"{fold_label} Train")
        axes[0].plot(epochs_range, hist["val_loss"], linestyle="-", color=c, linewidth=2, label=f"{fold_label} Val")

        # Plot Kanan: Accuracy
        axes[1].plot(epochs_range, hist["accuracy"], linestyle="--", color=c, alpha=0.5, label=f"{fold_label} Train")
        axes[1].plot(epochs_range, hist["val_accuracy"], linestyle="-", color=c, linewidth=2, label=f"{fold_label} Val")

    axes[0].set_title("5-Fold Cross Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("5-Fold Cross Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    plt.tight_layout()
    out_file = "assets/5fold_training_curves_MobileNetV2.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\n[OK] Grafik training 5-fold tersimpan di: {out_file}")
    plt.show()

if __name__ == "__main__":
    train_and_plot_5fold()