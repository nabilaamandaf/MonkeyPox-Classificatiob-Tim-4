import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS"
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def plot_accumulated_test_cm(model_name, model_path):
    if not os.path.exists(model_path):
        print(f"[SKIP] Model tidak ditemukan: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    all_y_true = []
    all_y_pred = []
    class_names = None

    for fold in FOLDS:
        test_dir = os.path.join(BASE_DIR, fold, "Test")
        if not os.path.exists(test_dir):
            continue

        ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            shuffle=False,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
        if class_names is None:
            class_names = ds.class_names

        y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
        y_score = model.predict(ds, verbose=0)
        y_pred = np.argmax(y_score, axis=1)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred)

    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix Akumulasi Test Set (5-Fold) - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = f"assets/cm_akumulasi_5fold_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Tersimpan di: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_accumulated_test_cm("MobileNetV2", "Models/transfer_learning_model_3.h5")