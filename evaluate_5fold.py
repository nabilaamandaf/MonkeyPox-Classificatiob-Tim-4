import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

#KONFIGURASI PATH & PARAMETER
# Sesuaikan BASE_DIR jika lokasi folder FOLDS berbeda
BASE_DIR = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS"
MODELS_TO_EVALUATE = {
    "Simple_CNN": "Models/simple_cnn_model_1.h5",
    "Modified_CNN": "Models/modified_cnn_model_2.h5",
    "MobileNetV2": "Models/transfer_learning_model_3.h5"
}

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
FOLDS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
SPLITS = ["Valid", "Test"]  # Sesuai ketentuan: Valid & Test dari Original Images

def load_dataset(folder_path):
    ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        shuffle=False,  # Penting agar urutan label y_true cocok dengan prediksi
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    return ds

def evaluate_model_5fold(model_name, model_path):
    if not os.path.exists(model_path):
        print(f"[SKIP] Model tidak ditemukan: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"MENGEVALUASI MODEL: {model_name} ({model_path})")
    print(f"{'='*60}")

    model = tf.keras.models.load_model(model_path)
    records = []

    for split in SPLITS:
        for idx, fold in enumerate(FOLDS, start=1):
            split_dir = os.path.join(BASE_DIR, fold, split)
            if not os.path.exists(split_dir):
                print(f"[WARNING] Folder tidak ditemukan: {split_dir}")
                continue

            ds = load_dataset(split_dir)
            n_classes = len(ds.class_names)

            # Ambil label asli (y_true)
            y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)

            # Prediksi probabilitas (y_score) dan kelas (y_pred)
            y_score = model.predict(ds, verbose=0)
            y_pred = np.argmax(y_score, axis=1)

            # Hitung semua metrik evaluasi
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            # Hitung ROC-AUC Multikelas (One-vs-Rest)
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            try:
                roc_auc = roc_auc_score(y_true_bin, y_score, multi_class="ovr", average="macro")
            except ValueError:
                roc_auc = np.nan  # Jika ada kelas yang tidak muncul di split tertentu

            records.append({
                "Fold": f"Fold {idx}",
                "Split": "Validation" if split == "Valid" else "Test",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": roc_auc
            })

    df = pd.DataFrame(records)

    # Hitung baris rata-rata Mean (5-Fold) untuk Validation dan Test
    summary_rows = []
    for split_label in ["Validation", "Test"]:
        sub_df = df[df["Split"] == split_label]
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

    # Tampilkan tabel rapi di terminal (6 angka di belakang koma seperti gambar)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + df_final.to_string(index=True, float_format=lambda x: f"{x:.6f}"))

    # Simpan otomatis ke CSV agar mudah disalin ke Laporan Word & README.md
    csv_filename = f"hasil_5fold_{model_name}.csv"
    df_final.to_csv(csv_filename, index=False, float_format="%.6f")
    print(f"\n[OK] Tabel rekapitulasi tersimpan ke: {csv_filename}")

    return df_final

if __name__ == "__main__":
    # Pastikan pandas terinstal: pip install pandas
    for name, path in MODELS_TO_EVALUATE.items():
        evaluate_model_5fold(name, path)




