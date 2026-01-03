# visualize_tp_embeddings_tsne.py
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import Counter

# === KONFIGURACJA ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "final"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

TP_TSV = DATA_DIR / "eng_toki_pona_final_training.tsv"
MODEL_DIR = MODELS_DIR / "mt5_en_tp_full_final_epoch9"

TOP_N = 100

OUTPUT_2D = ANALYSIS_DIR / "tp_embeddings_tsne_full_9ep_2d.png"
OUTPUT_3D = ANALYSIS_DIR / "tp_embeddings_tsne_full_9ep_3d.png"


def extract_tp_vocab_with_freq(tsv_path: Path) -> Counter:
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["toki_pona"])
    counter = Counter()
    for sent in df["toki_pona"]:
        tokens = str(sent).strip().split()
        counter.update(tokens)
    return counter


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Ładowanie modelu z:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Używane urządzenie:", device)

    print("Ekstrakcja słownictwa Toki Pona z:", TP_TSV)
    vocab_counter = extract_tp_vocab_with_freq(TP_TSV)

    most_common = vocab_counter.most_common(TOP_N * 2)
    tp_tokens = []
    token_ids = []

    print("Mapowanie tokenów na ID tokenizera...")
    for tok, freq in most_common:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            continue
        tp_tokens.append(tok)
        token_ids.append(tok_id)
        if len(tp_tokens) >= TOP_N:
            break

    print(f"Liczba tokenów TP użytych do wizualizacji: {len(tp_tokens)}")

    with torch.no_grad():
        emb_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()

    token_vecs = np.stack([emb_matrix[idx] for idx in token_ids], axis=0)
    print("Kształt macierzy embeddingów:", token_vecs.shape)

    # --- t-SNE 2D ---
    print("Uruchamianie t-SNE 2D...")
    tsne_2d = TSNE(
        n_components=2,
        perplexity=min(30, len(tp_tokens) - 1),
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_2d = tsne_2d.fit_transform(token_vecs)

    print("Rysowanie i zapis 2D:", OUTPUT_2D)
    plt.figure(figsize=(10, 8))
    x = emb_2d[:, 0]
    y = emb_2d[:, 1]
    plt.scatter(x, y)
    for i, tok in enumerate(tp_tokens):
        plt.text(x[i] + 0.1, y[i] + 0.1, tok, fontsize=6)
    plt.title("t-SNE 2D embeddingów słów Toki Pona (full, 6 epok)")
    plt.tight_layout()
    plt.savefig(OUTPUT_2D, dpi=300)
    plt.close()

    # --- t-SNE 3D ---
    print("Uruchamianie t-SNE 3D...")
    tsne_3d = TSNE(
        n_components=3,
        perplexity=min(30, len(tp_tokens) - 1),
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_3d = tsne_3d.fit_transform(token_vecs)

    print("Rysowanie i zapis 3D:", OUTPUT_3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    xs = emb_3d[:, 0]
    ys = emb_3d[:, 1]
    zs = emb_3d[:, 2]
    ax.scatter(xs, ys, zs)

    for i, tok in enumerate(tp_tokens):
        ax.text(xs[i], ys[i], zs[i], tok, fontsize=7)

    ax.set_title("t-SNE 3D embeddingów słów Toki Pona (full, 6 epok)")
    plt.tight_layout()
    plt.savefig(OUTPUT_3D, dpi=300)
    plt.close()

    print("Gotowe!")


if __name__ == "__main__":
    main()
