# plik: evaluate_bleu.py
from pathlib import Path
import json
import sacrebleu

# === KONFIGURACJA ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRED_DIR = DATA_DIR / "predictions"

# Plik z referencjami (100 zdań EN–TP)
REF_FILE = DATA_DIR / "test_set_100_en_tp.json"

# Nazwy modeli = prefiksy plików *_predictions.json
# Muszą się zgadzać z tym, co ustawiliśmy w generate_predictions_for_all_models.py
MODEL_LABELS = [
    "small_epoch3",
    "small_epoch6",
    "small_epoch9",
    "full_epoch3",
    "full_epoch6",
    "full_epoch9",
]


# === FUNKCJE POMOCNICZE ===

def load_references(ref_path: Path):
    """Wczytuje referencje TP z pliku test_set_100_en_tp.json."""
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_sorted = sorted(data, key=lambda x: x["id"])
    refs = [item["toki_pona"] for item in data_sorted]
    return refs


def load_predictions(pred_path: Path):
    """Wczytuje predykcje z pliku *_predictions.json."""
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_sorted = sorted(data, key=lambda x: x["id"])
    hyps = [item["prediction"] for item in data_sorted]
    return hyps


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("REF_FILE:", REF_FILE)

    if not REF_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku referencji: {REF_FILE}")

    ref_texts = load_references(REF_FILE)
    print(f"Liczba referencji: {len(ref_texts)}")

    for label in MODEL_LABELS:
        pred_file = PRED_DIR / f"{label}_predictions.json"
        if not pred_file.exists():
            print(f"[UWAGA] Brak pliku z predykcjami dla modelu '{label}': {pred_file}")
            continue

        hyp_texts = load_predictions(pred_file)

        if len(hyp_texts) != len(ref_texts):
            raise ValueError(
                f"Liczba predykcji ({len(hyp_texts)}) != "
                f"liczba referencji ({len(ref_texts)}) dla pliku {pred_file}"
            )

        bleu = sacrebleu.corpus_bleu(hyp_texts, [ref_texts])

        print(f"\n=== Model: {label} ===")
        print(f"BLEU = {bleu.score:.2f}")
        print(f"BP = {bleu.bp:.4f}")
        print(f"Precisions = {bleu.precisions}")
        print(f"sys_len = {bleu.sys_len}, ref_len = {bleu.ref_len}")
        print("-" * 40)


if __name__ == "__main__":
    main()
