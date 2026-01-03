# plik: generate_predictions_for_all_models.py

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

# === KONFIGURACJA ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "final"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"  # tu zapiszemy wyniki

TEST_JSON = DATA_DIR / "test_set_100_en_tp.json"

# Mapa: "etykieta_modelu" -> katalog z wytrenowanym modelem
# MODEL_DIRS: Dict[str, Path] = {
#     "small_epoch3": MODELS_DIR / "mt5_en_tp_small_final_epoch3",
#     "small_epoch6": MODELS_DIR / "mt5_en_tp_small_final_epoch6",
#     "full_epoch3":  MODELS_DIR / "mt5_en_tp_full_final_epoch3",
#     "full_epoch6":  MODELS_DIR / "mt5_en_tp_full_final_epoch6",
# }
MODEL_DIRS: Dict[str, Path] = {
    "small_epoch9": MODELS_DIR / "mt5_en_tp_small_final_epoch9",
    "full_epoch9":  MODELS_DIR / "mt5_en_tp_full_final_epoch9",
}

# Prefiks zadania – musi być taki sam jak w treningu
TASK_PREFIX = "translate English to Toki Pona: "

# Parametry generacji
MAX_LENGTH = 64
NUM_BEAMS = 4
BATCH_SIZE = 8  # ile zdań na raz generujemy


# === FUNKCJE POMOCNICZE ===

def load_test_set(path: Path):
    """Wczytuje plik JSON z 100 parami EN-TP i sortuje po id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_sorted = sorted(data, key=lambda x: x["id"])
    return data_sorted


def chunk_list(lst: List, n: int):
    """Dzieli listę na kawałki po n elementów."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_translations(
    model_dir: Path,
    test_examples: List[Dict],
    device: torch.device,
) -> List[str]:
    """Ładuje model z model_dir i generuje tłumaczenia EN→TP dla test_examples."""
    print(f"\n=== Ładowanie modelu z: {model_dir} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    english_sentences = [ex["english"] for ex in test_examples]
    predictions: List[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(chunk_list(english_sentences, BATCH_SIZE)):
            # Dodajemy prefiks zadania
            inputs = [TASK_PREFIX + s for s in batch]

            encodings = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)

            outputs = model.generate(
                **encodings,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
            )

            decoded = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            predictions.extend(decoded)

            print(
                f"  [model: {model_dir.name}] batch {batch_idx + 1} "
                f"({len(predictions)}/{len(english_sentences)})"
            )

    return predictions


def save_predictions_json(
    model_label: str,
    test_examples: List[Dict],
    predictions: List[str],
    output_dir: Path,
):
    """Zapisuje predykcje do JSON-a z id, EN, referencją TP i predykcją."""
    if len(test_examples) != len(predictions):
        raise ValueError(
            f"Liczba przykładów ({len(test_examples)}) != liczba predykcji ({len(predictions)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_label}_predictions.json"

    records = []
    for ex, pred in zip(test_examples, predictions):
        rec = {
            "id": ex["id"],
            "english": ex["english"],
            "reference_toki_pona": ex["toki_pona"],
            "prediction": pred,
        }
        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Zapisano {len(records)} predykcji do: {out_path}")


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Wczytywanie zestawu testowego z:", TEST_JSON)

    if not TEST_JSON.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku testowego: {TEST_JSON}")

    test_examples = load_test_set(TEST_JSON)
    print(f"Liczba przykładów w teście: {len(test_examples)}")

    # Ustal urządzenie (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Używane urządzenie:", device)

    # Iterujemy po wszystkich zdefiniowanych modelach
    for model_label, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            print(f"[UWAGA] Katalog modelu nie istnieje: {model_dir}")
            print("  Pomiń ten model albo popraw ścieżkę w MODEL_DIRS.\n")
            continue

        print(f"\n##### Model: {model_label} #####")
        preds = generate_translations(model_dir, test_examples, device=device)
        save_predictions_json(model_label, test_examples, preds, OUTPUT_DIR)

    print("\nGotowe. ")


if __name__ == "__main__":
    main()
