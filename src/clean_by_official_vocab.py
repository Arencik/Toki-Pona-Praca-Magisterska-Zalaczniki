import os
import string
import pandas as pd

# ===== KONFIGURACJA ŚCIEŻEK =====

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# wejściowy plik z parami EN–TP
IN_TSV_PATH = os.path.join(DATA_DIR, "eng_toki_pona.tsv")

# plik z oficjalnym słownikiem Toki Pony
TP_VOCAB_CSV_PATH = os.path.join(DATA_DIR, "toki_pona.csv")

# wyjściowy, przefiltrowany plik
OUT_TSV_PATH = os.path.join(DATA_DIR, "eng_toki_pona_official_vocab.tsv")

# nazwy kolumn w korpusie
COL_EN = "english"
COL_TP = "toki_pona"


def load_official_tp_vocab(csv_path: str) -> set:
    """
    Wczytuje plik CSV z oficjalnym słownikiem Toki Pony i zwraca zbiór dopuszczalnych słów.

    Zakładamy, że:
    - plik zawiera przynajmniej jedną kolumnę tekstową z tokenami,
    - domyślnie próbujemy kolumny o nazwie 'token' / 'word' / 'toki_pona',
      a jeśli ich nie ma, bierzemy po prostu pierwszą kolumnę.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie znaleziono pliku słownika Toki Pony: {csv_path}")

    df_vocab = pd.read_csv(csv_path)

    candidate_cols = ["token", "word", "toki_pona"]
    vocab_col = None
    for c in candidate_cols:
        if c in df_vocab.columns:
            vocab_col = c
            break

    if vocab_col is None:
        # jeśli żadna z powyższych nie istnieje, użyj pierwszej kolumny
        vocab_col = df_vocab.columns[0]
        print(f"[INFO] Nie znaleziono kolumn 'token'/'word'/'toki_pona'. "
              f"Używam pierwszej kolumny: '{vocab_col}'")

    vocab_series = df_vocab[vocab_col].dropna().astype(str)
    # normalizujemy do małych liter
    official_tokens = {w.strip().lower() for w in vocab_series if w.strip()}
    official_tokens.add("o")

    print(f"[INFO] Załadowano {len(official_tokens)} oficjalnych tokenów Toki Pony "
          f"z pliku: {csv_path}")
    return official_tokens


def sentence_uses_only_official_tokens(tp_sentence: str, official_vocab: set) -> bool:
    """
    Zwraca True, jeśli zdanie Toki Pona zawiera wyłącznie słowa z oficjalnego słownika.

    - Dzielimy po whitespace.
    - Usuwamy znaki interpunkcyjne na początku/końcu tokenu.
    - Porównujemy w wersji lowercase.

    Wszystkie tokeny muszą być w official_vocab. Jeśli znajdziemy choć jedno słowo spoza
    słownika, zdanie jest odrzucane.
    """
    if not isinstance(tp_sentence, str):
        return False

    tokens = tp_sentence.split()
    if not tokens:
        return False

    for tok in tokens:
        # usuwamy podstawową interpunkcję i cudzysłowy
        clean = tok.strip(string.punctuation + "„”«»\"'`“”")
        if not clean:
            continue

        # normalizujemy do małych liter
        normalized = clean.lower()

        if normalized not in official_vocab:
            # znalazło się słowo spoza słownika
            print(normalized)
            return False

    return True


def main():
    print("=" * 60)
    print("Czyszczenie korpusu EN–TP na podstawie oficjalnego słownika Toki Pony")
    print("=" * 60)
    print(f"Wejściowy plik TSV: {IN_TSV_PATH}")
    print(f"Plik słownika Toki Pony: {TP_VOCAB_CSV_PATH}")

    if not os.path.exists(IN_TSV_PATH):
        raise FileNotFoundError(f"Nie znaleziono wejściowego korpusu: {IN_TSV_PATH}")

    df = pd.read_csv(IN_TSV_PATH, sep="\t")

    if COL_EN not in df.columns or COL_TP not in df.columns:
        raise ValueError(
            f"Oczekiwane kolumny '{COL_EN}' i '{COL_TP}' nie istnieją w pliku TSV. "
            f"Masz kolumny: {list(df.columns)}"
        )

    print(f"Liczba przykładów przed czyszczeniem: {len(df)}")

    official_vocab = load_official_tp_vocab(TP_VOCAB_CSV_PATH)

    # maska: True = zdanie jest OK (same oficjalne tokeny)
    print("[INFO] Sprawdzanie, które zdania Toki Pona używają wyłącznie oficjalnego słownictwa...")
    mask_ok = df[COL_TP].apply(lambda s: sentence_uses_only_official_tokens(s, official_vocab))

    n_ok = mask_ok.sum()
    n_total = len(df)
    n_drop = n_total - n_ok

    df_clean = df[mask_ok].reset_index(drop=True)

    print(f"Liczba przykładów po czyszczeniu: {n_ok}")
    print(f"Liczba przykładów odrzuconych (z nieoficjalnymi słowami): {n_drop}")

    # zapisujemy oczyszczony korpus
    os.makedirs(DATA_DIR, exist_ok=True)
    df_clean.to_csv(OUT_TSV_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"\nZapisano oczyszczony plik TSV do: {OUT_TSV_PATH}")

    # przykładowe odrzucone zdania – do inspekcji
    print("\nPrzykładowe odrzucone zdania (max 5):")
    removed_examples = df[~mask_ok].head(5)
    for i, row in removed_examples.iterrows():
        print("EN:", row[COL_EN])
        print("TP:", row[COL_TP])
        print("---")


if __name__ == "__main__":
    main()
