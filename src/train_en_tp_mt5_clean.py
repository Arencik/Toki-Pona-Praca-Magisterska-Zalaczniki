# plik: src/train_en_tp_mt5.py

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ==== KONFIGURACJA LOKALNA (CPU) ====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models", "clean_official_vocab_3epoch")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TSV_PATH = os.path.join(DATA_DIR, "eng_toki_pona_final_training.tsv")
TSV_PATH_FULL = os.path.join(DATA_DIR, "eng_toki_pona.tsv")
MODEL_OUTPUT_DIR = os.path.join(MODELS_ROOT, "mt5_en_tp")
FINAL_MODEL_DIR = os.path.join(MODELS_ROOT, "mt5_en_tp_final")

print("TSV_PATH:", TSV_PATH)

# ==== IMPORTY WSPÓLNE ====
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
import inspect

# ==== USTAWIENIA MODELU (wspólne) ====
COL_EN = "english"
COL_TP = "toki_pona"

BASE_MODEL_NAME = "google/mt5-small"
MAX_SOURCE_LENGTH = 64
MAX_TARGET_LENGTH = 64
TASK_PREFIX = "translate English to Toki Pona: "


# 1. Wczytanie danych i podział
def load_parallel_data(tsv_path: str) -> DatasetDict:
    print("[3/7] Wczytywanie danych z TSV...")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=[COL_EN, COL_TP])
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_total = len(df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    print(f"Całkowita liczba przykładów: {n_total}")
    print(f"  train: {len(df_train)}")
    print(f"  validation: {len(df_val)}")
    print(f"  test: {len(df_test)}")

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "validation": Dataset.from_pandas(df_val, preserve_index=False),
            "test": Dataset.from_pandas(df_test, preserve_index=False),
        }
    )
    return dataset


# 2. Słownictwo Toki Pony
def extract_toki_pona_vocab(dataset: DatasetDict, col_tp: str = COL_TP):
    print("[4/7] Ekstrakcja słownictwa Toki Pony z całego zbioru...")
    tp_vocab = set()
    for split in ["train", "validation", "test"]:
        for text in dataset[split][col_tp]:
            if isinstance(text, str):
                words = text.strip().split()
                tp_vocab.update(words)
    tp_vocab = sorted(tp_vocab)
    print(f"Liczba unikalnych słów Toki Pony: {len(tp_vocab)}")
    return tp_vocab


# 3. Model + tokenizer
def prepare_model_and_tokenizer(tp_vocab):
    print("[5/7] Ładowanie modelu bazowego i tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [w for w in tp_vocab if w not in existing_tokens]

    print(f"  Słów Toki Pony w korpusie: {len(tp_vocab)}")
    print(f"  Nowe tokeny do dodania do tokenizera: {len(new_tokens)}")

    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print("  Zaktualizowano rozmiar warstwy embeddingów modelu.")
    else:
        print("  Brak nowych tokenów do dodania.")

    return model, tokenizer


# 4. Preprocessing
def preprocess_function(examples, tokenizer):
    inputs = [TASK_PREFIX + text for text in examples[COL_EN]]
    targets = examples[COL_TP]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 5. Tworzenie TrainingArguments odpornych na wersję transformers
def create_training_args(use_fp16: bool, model_output_dir: str, log_dir: str,
                         num_train_epochs: int = 3,
                         per_device_train_batch_size: int = 8,
                         per_device_eval_batch_size: int = 8):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters
    kwargs = {}

    def add(name, value):
        if name in params:
            kwargs[name] = value
        else:
            print(f"[INFO] Parametr '{name}' nie jest wspierany w tej wersji transformers – pomijam.")

    add("output_dir", model_output_dir)
    add("learning_rate", 3e-4)
    add("per_device_train_batch_size", per_device_train_batch_size)
    add("per_device_eval_batch_size", per_device_eval_batch_size)
    add("gradient_accumulation_steps", 1)
    add("weight_decay", 0.01)
    add("save_total_limit", 2)
    add("num_train_epochs", num_train_epochs)
    add("logging_dir", log_dir)
    add("logging_steps", 100)
    add("report_to", "none")
    add("fp16", use_fp16)

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "evaluate_during_training" in params:
        kwargs["evaluate_during_training"] = True

    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    if "predict_with_generate" in params:
        kwargs["predict_with_generate"] = True

    print("\n[INFO] Tworzę Seq2SeqTrainingArguments z następującymi argumentami:")
    for k, v in kwargs.items():
        print(f"  {k} = {v}")

    return Seq2SeqTrainingArguments(**kwargs)


# 6. Główna funkcja treningu – ENV-specyficzne rzeczy (ścieżki, debug, epoki) podamy jako argumenty
def run_training(TSV_PATH, MODEL_OUTPUT_DIR, FINAL_MODEL_DIR, LOG_DIR,
                 debug_small: bool = False,
                 num_train_epochs: int = 3,
                 batch_size: int = 8):
    print("=" * 60)
    print("Start pipeline'u treningowego EN → Toki Pona (mT5-small)")
    print("=" * 60)

    dataset = load_parallel_data(TSV_PATH)
    print(dataset)

    if debug_small:
        max_train = min(1000, len(dataset["train"]))
        max_val = min(200, len(dataset["validation"]))
        max_test = min(200, len(dataset["test"]))
        dataset["train"] = dataset["train"].select(range(max_train))
        dataset["validation"] = dataset["validation"].select(range(max_val))
        dataset["test"] = dataset["test"].select(range(max_test))
        print(f"[INFO] TRYB DEBUG: train={max_train}, val={max_val}, test={max_test}")

    tp_vocab = extract_toki_pona_vocab(dataset)
    model, tokenizer = prepare_model_and_tokenizer(tp_vocab)

    print("[6/7] Tokenizacja zbiorów (train/validation/test)...")

    def _preprocess(examples):
        return preprocess_function(examples, tokenizer)

    tokenized_datasets = dataset.map(
        _preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("  Przykładowy przykład ze zbioru treningowego po tokenizacji:")
    print({k: (v[:10] if isinstance(v, list) else v)
           for k, v in tokenized_datasets["train"][0].items()})

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = torch.cuda.is_available()
    training_args = create_training_args(
        use_fp16=use_fp16,
        model_output_dir=MODEL_OUTPUT_DIR,
        log_dir=LOG_DIR,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n=== START TRENINGU ===")
    train_result = trainer.train()
    print("=== KONIEC TRENINGU ===")

    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    print("\nZapis ostatecznego modelu i tokenizera...")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Ostateczny model zapisany w: {FINAL_MODEL_DIR}")

    print("\n=== EWALUACJA NA ZBIORZE TESTOWYM ===")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("Wyniki na zbiorze testowym:", test_results)
    trainer.save_metrics("test", test_results)

    return trainer, tokenized_datasets, tokenizer, dataset

if __name__ == "__main__":
    models = {
        "s" : {
            "MODEL_OUTPUT_DIR": os.path.join(MODELS_ROOT, "mt5_en_tp_small"),
            "FINAL_MODEL_DIR": os.path.join(MODELS_ROOT, "mt5_en_tp_small_final"),
            "TSV_PATH": TSV_PATH,
        },
        "f" : {
            "MODEL_OUTPUT_DIR": os.path.join(MODELS_ROOT, "mt5_en_tp_full"),
            "FINAL_MODEL_DIR": os.path.join(MODELS_ROOT, "mt5_en_tp_full_final"),
            "TSV_PATH": TSV_PATH_FULL,
        }
    }
    for key, cfg in models.items():
        for num_epoch in [3, 6, 9]:
            print(f"\n\n##### Trening modelu '{key}' #####\n")
            trainer, tokenized_datasets, tokenizer, raw_dataset = run_training(
                TSV_PATH=cfg["TSV_PATH"],
                MODEL_OUTPUT_DIR=cfg["MODEL_OUTPUT_DIR"]+f"_epoch{num_epoch}",
                FINAL_MODEL_DIR=cfg["FINAL_MODEL_DIR"]+f"_epoch{num_epoch}",
                LOG_DIR=LOG_DIR,
                    debug_small=False,
                    num_train_epochs=num_epoch,
                    batch_size=16,
                )