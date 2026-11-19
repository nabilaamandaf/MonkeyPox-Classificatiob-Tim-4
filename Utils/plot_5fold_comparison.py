import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel"

# Sesuaikan lokasi file CSV jika sudah kamu pindahkan ke Utils/ atau results/
CANDIDATE_DIRS = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "Utils"),
    os.path.join(PROJECT_ROOT, "results"),
]

def find_csv(filename):
    for d in CANDIDATE_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Tidak menemukan file: {filename}")

def get_mean_test_row(csv_path, model_label):
    df = pd.read_csv(csv_path)
    row = df[(df["Fold"] == "Mean (5-Fold)") & (df["Split"] == "Test")]
    if row.empty:
        raise ValueError(f"Baris Mean (5-Fold) / Test tidak ditemukan di {csv_path}")
    r = row.iloc[0]
    return {
        "Model": model_label,
        "accuracy": float(r["Accuracy"]),
        "precision": float(r["Precision"]),
        "recall": float(r["Recall"]),
        "f1_score": float(r["F1_Score"]),
        "roc_auc": float(r["ROC_AUC"]),
    }

def main():
    models_info = [
        ("hasil_5fold_Simple_CNN.csv", "SimpleCNN"),
        ("hasil_5fold_Modified_CNN.csv", "ModifiedCNN"),
        ("hasil_5fold_MobileNetV2.csv", "MobileNetV2"),
    ]

    data = [get_mean_test_row(find_csv(fname), label) for fname, label in models_info]
    df_plot = pd.DataFrame(data).set_index("Model")

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(metrics)))

    x = np.arange(len(df_plot.index))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - (len(metrics) - 1) / 2) * width
        vals = df_plot[metric].values
        bars = ax.bar(x + offset, vals, width, label=metric, color=color, edgecolor="none")

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_title("Perbandingan Performa Rata-Rata 5-Fold Test Set (3 Model)", fontsize=13, pad=12)
    ax.set_xlabel("Model", fontsize=10.5)
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", ncol=3, frameon=True, facecolor="white", framealpha=0.9)

    out_dir = os.path.join(PROJECT_ROOT, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perbandingan_5fold_test_3_model.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Grafik tersimpan di: {out_path}")

if __name__ == "__main__":
    main()