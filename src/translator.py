import os
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

FINAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "mt5_en_tp_final")

# Prefix do zadania
TASK_PREFIX = "translate English to Toki Pona: "

# tokenizer i model
print("Loading tokenizer and model from:", FINAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(FINAL_MODEL_DIR)

# GPU jeśli jest, inaczej CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)
def translate_en_to_tp(
    sentences,
    max_length=64,
    num_beams=4,
    do_sample=False,
    top_p=0.9,
    temperature=1.0
):
    """
    Przyjmuje listę zdań po angielsku i zwraca listę tłumaczeń na Toki Ponę.
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    # Dodajemy prefiks T5/mT5
    inputs = [TASK_PREFIX + s for s in sentences]

    # Tokenizacja wejścia
    encodings = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Generowanie tłumaczeń
    with torch.no_grad():
        generated_ids = model.generate(
            **encodings,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )

    # Dekodowanie tokenów na tekst
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs

# Przykładowe użycie
test_sentences = [
    "Hello, how are you?",
    "I like learning Toki Pona.",
    "The weather is good today.",
    "This is my first neural machine translation model.",
]

translations = translate_en_to_tp(test_sentences)

for src, tgt in zip(test_sentences, translations):
    print("EN:", src)
    print("TP:", tgt)
    print("-" * 40)