import os
import string
import pandas as pd

# KONFIGURACJA ŚCIEŻEK

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

IN_TSV_PATH = os.path.join(DATA_DIR, "eng_toki_pona.tsv")
TP_VOCAB_CSV_PATH = os.path.join(DATA_DIR, "toki_pona.csv")
UNOFFICIAL_FREQ_PATH = os.path.join(DATA_DIR, "unofficial_tokens_freq.csv")

OUT_FINAL_PATH = os.path.join(DATA_DIR, "eng_toki_pona_final_training.tsv")

COL_EN = "english"
COL_TP = "toki_pona"

# FUNKCJE POMOCNICZE

PUNCT_STRIP = string.punctuation + "„”«»\"'`“”"

def normalize_vocab_token(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)

    s = raw.strip()
    if not s:
        return ""

    base = s.split()[0]              # np. "alasa (e )" -> "alasa"
    base = base.strip(PUNCT_STRIP)
    base = base.lower()

    return base


def normalize_sentence_token(tok: str) -> str:
    if not isinstance(tok, str):
        tok = str(tok)

    clean = tok.strip(PUNCT_STRIP)
    if not clean:
        return ""

    return clean.lower()


def is_proper_name_token(tok: str) -> bool:
    if not isinstance(tok, str):
        tok = str(tok)

    clean = tok.strip(PUNCT_STRIP)
    if not clean:
        return False

    # jeśli jakakolwiek wielka litera — token to nazwa własna
    return any(ch.isupper() for ch in clean)


# 1. Wczytywanie i czyszczenie słownika oficjalnego

def load_official_tp_vocab(csv_path: str) -> set:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    # Wybieramy pierwszą kolumnę tekstową
    vocab_col = df.columns[0]
    vocab_series = df[vocab_col].dropna().astype(str)

    vocab = set()
    for raw in vocab_series:
        norm = normalize_vocab_token(raw)
        if norm:
            vocab.add(norm)

    vocab.add("o")   # awaryjnie

    print(f"[INFO] Oficjalne tokeny: {len(vocab)}")
    return vocab


# 2. Wczytywanie manualnego labelingu

def load_curated_unofficial_vocab(csv_path: str) -> set:
    df = pd.read_csv(csv_path)

    df.columns = [c.lower() for c in df.columns]

    required = {"token", "count", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Plik musi zawierać kolumny: {required}. Masz: {df.columns}")

    forbidden = {"", "unknown", "obscure"}

    mask = (
        df["count"] >= 5
        & (~df["label"].fillna("").str.lower().isin(forbidden))
    )

    selected = set(df.loc[mask, "token"].astype(str).str.lower())

    print(f"[INFO] Wybrane nieoficjalne tokeny: {len(selected)}")
    return selected


# 3. Filtracja końcowego korpusu

def sentence_acceptable(tp_sentence: str, allowed_vocab: set) -> bool:
    if not isinstance(tp_sentence, str):
        return False

    tokens = tp_sentence.split()
    if not tokens:
        return False

    for tok in tokens:
        if is_proper_name_token(tok):
            return False  # wykluczamy nazwy własne

        norm = normalize_sentence_token(tok)
        if not norm:
            continue

        if norm not in allowed_vocab:
            return False

    return True


# MAIN

def main():
    print("=== GENEROWANIE FINALNEGO KORPUSU ===")

    df = pd.read_csv(IN_TSV_PATH, sep="\t")
    print(f"[INFO] Wczytano {len(df)} wierszy z TSV")

    official_vocab = load_official_tp_vocab(TP_VOCAB_CSV_PATH)
    curated_unofficial = load_curated_unofficial_vocab(UNOFFICIAL_FREQ_PATH)

    final_vocab = official_vocab.union(curated_unofficial)
    print(f"[INFO] Finalny rozmiar słownika: {len(final_vocab)} tokenów")

    mask = df[COL_TP].apply(lambda s: sentence_acceptable(s, final_vocab))
    df_final = df[mask].reset_index(drop=True)

    print(f"[INFO] Po filtracji zostało {len(df_final)} przykładów.")

    os.makedirs(DATA_DIR, exist_ok=True)
    df_final.to_csv(OUT_FINAL_PATH, sep="\t", index=False, encoding="utf-8")

    print(f"[OK] Zapisano finalny korpus treningowy: {OUT_FINAL_PATH}")


if __name__ == "__main__":
    main()
