# visualize_joint_embeddings.py
from pathlib import Path
import numpy as np
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

MODEL_DIR = MODELS_DIR / "final" / "mt5_en_tp_full_final_epoch9"

OUTPUT_PNG = ANALYSIS_DIR / "tsne_joint_en_tp_f9.png"

# słowa EN i TP
WORDS = [
    # emocje
    "good", "bad", "happy", "sad", "feel",
    "pona", "ike", "pilin",

    # osoby
    "person", "man", "woman", "child",
    "jan", "mije", "meli", "jan lili",

    # kategorie/kolory
    "color", "type",
    "kule",

    # przestrzeń
    "inside", "outside", "near", "far",
    "lon", "weka", "poka",
]


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    with torch.no_grad():
        emb_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()

    valid_words = []
    vecs = []

    for w in WORDS:
        tok_id = tokenizer.convert_tokens_to_ids(w)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            continue
        valid_words.append(w)
        vecs.append(emb_matrix[tok_id])

    vecs = np.array(vecs)

    # t-SNE wspólne
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(valid_words) - 1),
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(vecs)

    plt.figure(figsize=(12, 10))

    for i, w in enumerate(valid_words):
        color = "blue" if w.isalpha() and w.isascii() else "red"
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1], c=color)
        plt.text(emb_2d[i, 0] + 0.4, emb_2d[i, 1] + 0.4, w, fontsize=9)

    plt.title("Wspólna przestrzeń embeddingów (EN + Toki Pona) – t-SNE 2D")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print("Zapisano wykres:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
