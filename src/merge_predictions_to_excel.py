# merge_predictions_to_excel.py
import json
import pandas as pd
from pathlib import Path

# === KONFIGURACJA ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRED_DIR = DATA_DIR / "predictions"

TEST_SET = DATA_DIR / "test_set_100_en_tp.json"

# Nazwy modeli -> odpowiadające pliki z predykcjami
MODEL_FILES = {
    "pred_s_3": PRED_DIR / "small_epoch3_predictions.json",
    "pred_s_6": PRED_DIR / "small_epoch6_predictions.json",
    "pred_s_9": PRED_DIR / "small_epoch9_predictions.json",
    "pred_f_3": PRED_DIR / "full_epoch3_predictions.json",
    "pred_f_6": PRED_DIR / "full_epoch6_predictions.json",
    "pred_f_9": PRED_DIR / "full_epoch9_predictions.json",
}

OUTPUT_XLSX = DATA_DIR / "evaluation_dataset.xlsx"


# === FUNKCJE ===

def load_json_sorted(path: Path):
    """Ładuje JSON i sortuje rekordy po id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data, key=lambda x: x["id"])


def main():
    print("Wczytywanie zbioru testowego:", TEST_SET)
    test_data = load_json_sorted(TEST_SET)

    # Podstawowe kolumny
    df = pd.DataFrame({
        "id": [item["id"] for item in test_data],
        "EN": [item["english"] for item in test_data],
        "TP": [item["toki_pona"] for item in test_data],
    })

    # Wczytujemy predykcje każdego modelu i dodajemy jako kolumny
    for col_name, json_path in MODEL_FILES.items():
        if not json_path.exists():
            print(f"[UWAGA] Brak predykcji: {json_path}, pomijam tę kolumnę.")
            continue

        pred_data = load_json_sorted(json_path)

        # Lista predykcji w kolejności id
        df[col_name] = [item["prediction"] for item in pred_data]

        # Kolumna ocenowa (pusta)
        df["ocena_" + col_name.split("_")[1] + "_" + col_name.split("_")[2]] = ""

    # Usuwamy id z końcowego excela
    df = df.drop(columns=["id"])

    # Zapis do Excela
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nPlik Excel zapisany do: {OUTPUT_XLSX}")
    print("Możesz teraz ręcznie uzupełnić kolumny ocena_*.")


if __name__ == "__main__":
    main()
