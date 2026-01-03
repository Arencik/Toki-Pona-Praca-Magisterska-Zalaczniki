# visualize_joint_embeddings_3d.py
from pathlib import Path
import numpy as np
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ===================== KONFIGURACJA ŚCIEŻEK =====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "final"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Najlepszy model
MODEL_DIR = MODELS_DIR / "mt5_en_tp_full_final_epoch9"

OUTPUT_PNG = ANALYSIS_DIR / "tsne_joint_en_tp_3d_f9.png"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ===================== LISTA SŁÓW (EN + TP) =====================

# format: (lang, word, group)
JOINT_WORDS = [
    # ===== emocje pozytywne =====
    ("en", "good",   "emotion_pos"),
    ("en", "happy",  "emotion_pos"),
    ("en", "smile",  "emotion_pos"),
    ("en", "peace",  "emotion_pos"),
    ("tp", "pona",   "emotion_pos"),
    ("tp", "olin",   "emotion_pos"),
    ("tp", "musi",   "emotion_pos"),
    ("tp", "suwi",   "emotion_pos"),

    # ===== emocje negatywne =====
    ("en", "bad",    "emotion_neg"),
    ("en", "sad",    "emotion_neg"),
    ("en", "angry",  "emotion_neg"),
    ("en", "fear",   "emotion_neg"),
    ("en", "hurt",   "emotion_neg"),
    ("tp", "ike",    "emotion_neg"),
    ("tp", "jaki",   "emotion_neg"),
    ("tp", "pakala", "emotion_neg"),
    ("tp", "utala",  "emotion_neg"),
    ("tp", "nasa",   "emotion_neg"),

    # ===== osoby / role =====
    ("en", "person", "person"),
    ("en", "man",    "person"),
    ("en", "woman",  "person"),
    ("en", "child",  "person"),
    ("en", "friend", "person"),
    ("en", "group",  "person"),
    ("tp", "jan",    "person"),
    ("tp", "mije",   "person"),
    ("tp", "meli",   "person"),
    ("tp", "lili",   "person"),
    ("tp", "kulupu", "person"),

    # ===== ciało / zmysły =====
    ("en", "body",   "body"),
    ("en", "head",   "body"),
    ("en", "hand",   "body"),
    ("en", "eye",    "body"),
    ("en", "ear",    "body"),
    ("en", "heart",  "body"),
    ("tp", "sijelo", "body"),
    ("tp", "luka",   "body"),
    ("tp", "lukin",  "body"),
    ("tp", "kute",   "body"),
    ("tp", "pilin",  "body"),

    # ===== poznanie / myślenie =====
    ("en", "know",     "cognition"),
    ("en", "think",    "cognition"),
    ("en", "learn",    "cognition"),
    ("en", "remember", "cognition"),
    ("tp", "sona",     "cognition"),
    ("tp", "ni",       "cognition"),
    ("tp", "kule",     "cognition"),
    ("tp", "nasin",    "cognition"),

    # ===== ruch / przestrzeń =====
    ("en", "go",      "motion_space"),
    ("en", "come",    "motion_space"),
    ("en", "walk",    "motion_space"),
    ("en", "near",    "motion_space"),
    ("en", "far",     "motion_space"),
    ("en", "inside",  "motion_space"),
    ("en", "outside", "motion_space"),
    ("tp", "tawa",    "motion_space"),
    ("tp", "kama",    "motion_space"),
    ("tp", "weka",    "motion_space"),
    ("tp", "poka",    "motion_space"),
    ("tp", "lon",     "motion_space"),
    ("tp", "anpa",    "motion_space"),
    ("tp", "sewi",    "motion_space"),

    # ===== czas =====
    ("en", "time",   "time"),
    ("en", "day",    "time"),
    ("en", "night",  "time"),
    ("en", "now",    "time"),
    ("en", "before", "time"),
    ("en", "after",  "time"),
    ("tp", "tenpo",  "time"),
    ("tp", "suno",   "time"),
    ("tp", "mun",    "time"),
    ("tp", "kama",   "time"),
    ("tp", "pini",   "time"),
    ("tp", "open",   "time"),

    # ===== ilość / rozmiar =====
    ("en", "big",    "quantity"),
    ("en", "small",  "quantity"),
    ("en", "many",   "quantity"),
    ("en", "few",    "quantity"),
    ("tp", "suli",   "quantity"),
    ("tp", "lili",   "quantity"),
    ("tp", "mute",   "quantity"),
    ("tp", "ale",    "quantity"),

    # ===== świat / natura =====
    ("en", "world",  "nature"),
    ("en", "land",   "nature"),
    ("en", "water",  "nature"),
    ("en", "fire",   "nature"),
    ("en", "sky",    "nature"),
    ("en", "sun",    "nature"),
    ("en", "moon",   "nature"),
    ("tp", "ma",     "nature"),
    ("tp", "telo",   "nature"),
    ("tp", "seli",   "nature"),
    ("tp", "kon",    "nature"),
    ("tp", "suno",   "nature"),
    ("tp", "mun",    "nature"),
    ("tp", "kasi",   "nature"),

    # ===== obiekty / miejsca =====
    ("en", "house",  "object_place"),
    ("en", "room",   "object_place"),
    ("en", "city",   "object_place"),
    ("en", "road",   "object_place"),
    ("en", "tool",   "object_place"),
    ("en", "food",   "object_place"),
    ("tp", "tomo",   "object_place"),
    ("tp", "nasin",  "object_place"),
    ("tp", "ilo",    "object_place"),
    ("tp", "moku",   "object_place"),

    # ===== społeczne / działanie =====
    ("en", "work",   "social"),
    ("en", "play",   "social"),
    ("en", "talk",   "social"),
    ("en", "help",   "social"),
    ("en", "fight",  "social"),
    ("tp", "pali",   "social"),
    ("tp", "musi",   "social"),
    ("tp", "toki",   "social"),
    ("tp", "pana",   "social"),
    ("tp", "utala",  "social"),
]

# ===================== WIZUALIZACJA 3D =====================

def main():
    print("Ładowanie modelu z:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    with torch.no_grad():
        emb_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()

    words = []
    langs = []
    groups = []
    vecs = []

    for lang, word, group in JOINT_WORDS:
        tok_id = tokenizer.convert_tokens_to_ids(word)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            # pomijamy słowa, które nie mają osobnego tokenu
            continue
        words.append(word)
        langs.append(lang)
        groups.append(group)
        vecs.append(emb_matrix[tok_id])

    vecs = np.array(vecs)
    print(f"Użytych słów: {len(words)}")

    tsne = TSNE(
        n_components=3,
        perplexity=min(30, len(words) - 1),
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_3d = tsne.fit_transform(vecs)

    # kolory dla grup
    group_colors = {
        "emotion_pos": "tab:green",
        "emotion_neg": "tab:red",
        "person":      "tab:blue",
        "body":        "tab:purple",
        "cognition":   "tab:orange",
        "motion_space":"tab:brown",
        "time":        "tab:cyan",
        "quantity":    "tab:pink",
        "nature":      "tab:olive",
        "object_place":"tab:gray",
        "social":      "gold",
    }

    # markery dla języków
    lang_markers = {
        "en": "o",
        "tp": "^",
    }

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i, (w, lang, group) in enumerate(zip(words, langs, groups)):
        color = group_colors.get(group, "black")
        marker = lang_markers.get(lang, "o")
        ax.scatter(
            emb_3d[i, 0],
            emb_3d[i, 1],
            emb_3d[i, 2],
            c=color,
            marker=marker,
            s=50,
        )
        ax.text(
            emb_3d[i, 0],
            emb_3d[i, 1],
            emb_3d[i, 2],
            w,
            fontsize=7,
        )

    ax.set_title("Wspólna przestrzeń embeddingów EN + Toki Pona (t-SNE 3D)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print("Zapisano wykres:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
