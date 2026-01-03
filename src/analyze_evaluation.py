from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === KONFIGURACJA ŚCIEŻEK ===

# Jeśli plik jest w src/, cofamy się o jeden poziom do katalogu głównego projektu
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

INPUT_XLSX = DATA_DIR / "evaluation_dataset.xlsx"
SUMMARY_XLSX = ANALYSIS_DIR / "evaluation_summary.xlsx"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Mapowanie modeli na nazwy kolumn z ocenami
MODEL_CONFIG = {
    "s_3": {
        "label": "small, 3 epoki",
        "variant": "small",
        "epochs": 3,
        "score_col": "ocena_s_3",
    },
    "s_6": {
        "label": "small, 6 epok",
        "variant": "small",
        "epochs": 6,
        "score_col": "ocena_s_6",
    },
    "s_9": {
        "label": "small, 9 epok",
        "variant": "small",
        "epochs": 9,
        "score_col": "ocena_s_9",
    },
    "f_3": {
        "label": "full, 3 epoki",
        "variant": "full",
        "epochs": 3,
        "score_col": "ocena_f_3",
    },
    "f_6": {
        "label": "full, 6 epok",
        "variant": "full",
        "epochs": 6,
        "score_col": "ocena_f_6",
    },
    "f_9": {
        "label": "full, 9 epok",
        "variant": "full",
        "epochs": 9,
        "score_col": "ocena_f_9",
    },
}

# Np.: BLEU_SCORES = {"s_3": 18.5, "s_6": 19.2, "f_3": 17.1, "f_6": 17.8}
BLEU_SCORES = {
    "s_3": 31.55,
    "s_6": 49.26,
    "s_9": 56.72,
    "f_3": 30.05,
    "f_6": 52.11,
    "f_9": 60.55,
}


def load_data(path: Path) -> pd.DataFrame:
    """Wczytuje główny plik z ocenami."""
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    df = pd.read_excel(path)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Buduje tabelę podsumowującą dla wszystkich modeli."""
    rows = []

    for model_key, cfg in MODEL_CONFIG.items():
        col = cfg["score_col"]

        if col not in df.columns:
            print(f"[UWAGA] Kolumna {col} nie istnieje w dataframe, pomijam model {model_key}")
            continue

        # Upewniamy się, że to są liczby
        scores = pd.to_numeric(df[col], errors="coerce").dropna()

        # Podstawowe statystyki
        mean_score = scores.mean()
        std_score = scores.std(ddof=1)

        # Rozkład ocen 0/1/2
        counts = scores.value_counts().reindex([0, 1, 2], fill_value=0)
        total = counts.sum()
        shares = counts / total if total > 0 else counts

        row = {
            "model_id": model_key,
            "model_label": cfg["label"],
            "variant": cfg["variant"],
            "epochs": cfg["epochs"],
            "mean_score": mean_score,
            "std_score": std_score,
            "count_0": int(counts.get(0, 0)),
            "count_1": int(counts.get(1, 0)),
            "count_2": int(counts.get(2, 0)),
            "share_0": float(shares.get(0, 0.0)),
            "share_1": float(shares.get(1, 0.0)),
            "share_2": float(shares.get(2, 0.0)),
            "bleu": BLEU_SCORES.get(model_key),
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sortowanie np. według variant + epochs
    summary_df = summary_df.sort_values(by=["variant", "epochs"]).reset_index(drop=True)
    return summary_df


def plot_avg_scores(summary_df: pd.DataFrame, out_path: Path):
    """Wykres słupkowy: średnia ocena manualna dla każdego modelu."""
    labels = summary_df["model_label"].tolist()
    means = summary_df["mean_score"].tolist()

    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("Średnia ocena manualna")
    plt.title("Średnia jakość tłumaczeń (ocena manualna)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rating_distribution(summary_df: pd.DataFrame, out_path: Path):
    """Wykres słupkowy (skumulowany): rozkład ocen 0/1/2 dla każdego modelu."""
    labels = summary_df["model_label"].tolist()
    x = np.arange(len(labels))

    share_0 = summary_df["share_0"].values
    share_1 = summary_df["share_1"].values
    share_2 = summary_df["share_2"].values

    plt.figure()
    # Słupki skumulowane
    plt.bar(x, share_0, label="ocena 0")
    plt.bar(x, share_1, bottom=share_0, label="ocena 1")
    plt.bar(x, share_2, bottom=share_0 + share_1, label="ocena 2")

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Udział ocen")
    plt.title("Rozkład ocen 0/1/2 dla modeli")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("Wczytywanie danych z:", INPUT_XLSX)
    df = load_data(INPUT_XLSX)

    print("Budowanie podsumowania...")
    summary_df = build_summary(df)

    print("Zapis podsumowania do Excela:", SUMMARY_XLSX)
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    # Wykresy
    avg_scores_png = ANALYSIS_DIR / "avg_scores.png"
    dist_png = ANALYSIS_DIR / "rating_distribution.png"

    print("Tworzenie wykresu średnich ocen...")
    plot_avg_scores(summary_df, avg_scores_png)
    print("Zapisano:", avg_scores_png)

    print("Tworzenie wykresu rozkładu ocen...")
    plot_rating_distribution(summary_df, dist_png)
    print("Zapisano:", dist_png)

    print("\nGotowe. Podsumowanie: evaluation_summary.xlsx, wykresy w folderze analysis/.")


if __name__ == "__main__":
    main()
