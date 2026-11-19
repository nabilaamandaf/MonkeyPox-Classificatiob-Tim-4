import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

#KONFIGURASI PATH
PROJECT_ROOT = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel"
BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "archive",
    "Original Images",
    "Original Images",
    "FOLDS"
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "hasil_5fold_MobileNetV2.csv")

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
SPLITS = ["Valid", "Test"]


def load_dataset(folder_path):
    return tf.keras.utils.image_dataset_from_directory(
        folder_path,
        shuffle=False,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )


def generate_mobilenetv2_5fold_csv():
    records = []

    for split in SPLITS:
        for idx, fold in enumerate(FOLDS, start=1):
            split_dir = os.path.join(BASE_DIR, fold, split)
            if not os.path.exists(split_dir):
                print(f"[WARNING] Folder tidak ditemukan: {split_dir}")
                continue

            fold_model_path = os.path.join(MODELS_DIR, f"mobilenetv2_{fold}.h5")
            if not os.path.exists(fold_model_path):
                print(f"[WARNING] Model tidak ditemukan: {fold_model_path}")
                continue

            model = tf.keras.models.load_model(fold_model_path)
            ds = load_dataset(split_dir)
            n_classes = len(ds.class_names)

            y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
            y_score = model.predict(ds, verbose=0)
            y_pred = np.argmax(y_score, axis=1)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            try:
                roc_auc = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")
            except ValueError:
                roc_auc = np.nan

            records.append({
                "Fold": f"Fold {idx}",
                "Split": "Validation" if split == "Valid" else "Test",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": roc_auc
            })

    if not records:
        print("[ERROR] Tidak ada data evaluasi yang berhasil diproses.")
        return

    df = pd.DataFrame(records)
    summary_rows = []

    for split_label in ["Validation", "Test"]:
        sub_df = df[df["Split"] == split_label]
        if not sub_df.empty:
            summary_rows.append({
                "Fold": "Mean (5-Fold)",
                "Split": split_label,
                "Accuracy": sub_df["Accuracy"].mean(),
                "Precision": sub_df["Precision"].mean(),
                "Recall": sub_df["Recall"].mean(),
                "F1_Score": sub_df["F1_Score"].mean(),
                "ROC_AUC": sub_df["ROC_AUC"].mean()
            })

    df_final = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + df_final.to_string(index=True, float_format=lambda x: f"{x:.6f}"))

    df_final.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"\n[OK] File berhasil dibuat: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_mobilenetv2_5fold_csv()