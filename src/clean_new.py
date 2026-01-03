import os
import string
import pandas as pd
from collections import Counter

# ===== KONFIGURACJA ŚCIEŻEK =====

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# wejściowy plik z parami EN–TP
IN_TSV_PATH = os.path.join(DATA_DIR, "eng_toki_pona.tsv")

# plik z oficjalnym słownikiem Toki Pony (zanieczyszczony)
TP_VOCAB_CSV_PATH = os.path.join(DATA_DIR, "toki_pona.csv")

# wyjściowy, przefiltrowany plik – tylko oficjalne słowa
OUT_TSV_OFFICIAL_PATH = os.path.join(DATA_DIR, "eng_toki_pona_official_vocab1.tsv")

# wyjściowy plik – zdania z nieoficjalnymi słowami, ale bez nazw własnych
OUT_TSV_UNOFFICIAL_PATH = os.path.join(DATA_DIR, "eng_toki_pona_unofficial_no_names.tsv")

# słownik częstości nieoficjalnych tokenów
OUT_UNOFFICIAL_FREQ_CSV = os.path.join(DATA_DIR, "unofficial_tokens_freq.csv")

# nazwy kolumn w korpusie
COL_EN = "english"
COL_TP = "toki_pona"


# ===== POMOCNICZE NORMALIZACJE =====

EXTRA_PUNCT = "„”«»\"'`“”"
PUNCT_STRIP = string.punctuation + EXTRA_PUNCT


def normalize_vocab_token(raw: str) -> str:
    """
    Normalizacja pojedynczej pozycji ze słownika Toki Pony:

    - strip() spacje,
    - bierzemy pierwszy "wyraz" przed whitespace (np. 'kama (e )' -> 'kama'),
    - z tego słowa zdejmujemy interpunkcję z początku/końca,
    - zamieniamy na lowercase.

    Zwraca pusty string, jeśli po czyszczeniu nic nie zostało.
    """
    if not isinstance(raw, str):
        raw = str(raw)

    s = raw.strip()
    if not s:
        return ""

    # pierwszy fragment przed spacją: 'alasa (e )' -> 'alasa'
    base = s.split()[0]

    # zdejmij interpunkcję
    base = base.strip(PUNCT_STRIP)

    base = base.lower()
    return base


def normalize_sentence_token(raw_tok: str) -> str:
    """
    Normalizacja tokenu ze zdania:

    - zdejmujemy interpunkcję z początku/końca,
    - lowercase.

    Zwraca pusty string, jeśli po czyszczeniu nic nie zostało.
    """
    if not isinstance(raw_tok, str):
        raw_tok = str(raw_tok)

    clean = raw_tok.strip(PUNCT_STRIP)
    if not clean:
        return ""
    return clean.lower()


def is_proper_name_token(raw_tok: str) -> bool:
    """
    Heurystyka: nazwą własną uznajemy token zawierający jakąkolwiek wielką literę
    po zdjęciu interpunkcji z początku/końca.

    'Italija' -> True
    'Lisa,'   -> True
    'jan'     -> False
    """
    if not isinstance(raw_tok, str):
        raw_tok = str(raw_tok)

    clean = raw_tok.strip(PUNCT_STRIP)
    if not clean:
        return False

    return any(ch.isupper() for ch in clean)


# ===== 1. ŁADOWANIE I CZYSZCZENIE SŁOWNIKA =====

def load_official_tp_vocab(csv_path: str) -> set:
    """
    Wczytuje plik CSV z oficjalnym słownikiem Toki Pony i zwraca zbiór dopuszczalnych słów.

    Ten wariant *czyści* plik:
    - wybiera pierwszą kolumnę z tokenami,
    - z każdej komórki robi normalizację (normalize_vocab_token),
    - wyrzuca pustki i duplikaty.

    Dla pliku z pu/ku powinno dać ~120+ unikalnych słów.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie znaleziono pliku słownika Toki Pony: {csv_path}")

    df_vocab = pd.read_csv(csv_path)

    # próbujemy nazwę kolumny: 'token' / 'word' / 'toki_pona' / inaczej pierwsza
    candidate_cols = ["token", "word", "toki_pona"]
    vocab_col = None
    for c in candidate_cols:
        if c in df_vocab.columns:
            vocab_col = c
            break

    if vocab_col is None:
        vocab_col = df_vocab.columns[0]
        print(f"[INFO] Nie znaleziono kolumn 'token'/'word'/'toki_pona'. "
              f"Używam pierwszej kolumny: '{vocab_col}'")

    vocab_series = df_vocab[vocab_col].dropna().astype(str)

    official_tokens = set()
    for raw in vocab_series:
        norm = normalize_vocab_token(raw)
        if norm:
            official_tokens.add(norm)

    # na wszelki wypadek dodajemy 'o', jeśli coś by je wycięło
    official_tokens.add("o")

    print(f"[INFO] Załadowano {len(official_tokens)} oficjalnych tokenów Toki Pony "
          f"z pliku: {csv_path}")
    return official_tokens


# ===== 2. Funkcja do KORPUSU OFICJALNEGO =====

def sentence_uses_only_official_tokens(tp_sentence: str, official_vocab: set) -> bool:
    """
    Zwraca True, jeśli zdanie Toki Pona zawiera wyłącznie słowa z oficjalnego słownika.

    - Dzielimy po whitespace.
    - Usuwamy interpunkcję na początku/końcu tokenu.
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
        norm = normalize_sentence_token(tok)
        if not norm:
            continue

        if norm not in official_vocab:
            return False

    return True


# ===== 3. zdania z NIEOFICJALNYMI słowami, BEZ nazw własnych =====

def extract_unofficial_sentences_and_counts(
    df: pd.DataFrame,
    official_vocab: set,
    col_tp: str = COL_TP
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Z wejściowego DataFrame (z kolumną Toki Pona) wyciąga:

    - df_unofficial: wiersze, w których:
        * występuje co najmniej jedno NIEOFICJALNE słowo (nie ma go w official_vocab),
        * nie występuje żadna nazwa własna (heurystyka: token z wielką literą).
    - counts: Series z częstością nieoficjalnych tokenów, posortowana malejąco.

    Nazwy własne są wykrywane przez is_proper_name_token().
    Normalizacja tokenów w zdaniach jak w sentence_uses_only_official_tokens().
    """
    rows = []
    counts = Counter()

    for _, row in df.iterrows():
        sent = row[col_tp]
        if not isinstance(sent, str):
            continue

        tokens = sent.split()
        if not tokens:
            continue

        has_name = False
        unofficial_tokens_in_sent = []

        for raw_tok in tokens:
            # najpierw sprawdzamy, czy to nazwa własna
            if is_proper_name_token(raw_tok):
                has_name = True
                break

            norm = normalize_sentence_token(raw_tok)
            if not norm:
                continue

            if norm not in official_vocab:
                unofficial_tokens_in_sent.append(norm)

        # jeśli mamy nazwę własną albo nie ma nieoficjalnych tokenów – odrzucamy
        if has_name or not unofficial_tokens_in_sent:
            continue

        # ten wiersz kwalifikuje się do korpusu "nieoficjalnego"
        rows.append(row)

        # zliczamy nieoficjalne tokeny
        for ut in unofficial_tokens_in_sent:
            counts[ut] += 1

    df_unofficial = pd.DataFrame(rows).reset_index(drop=True)
    counts_series = pd.Series(counts).sort_values(ascending=False)

    print(f"[INFO] Zdania z nieoficjalnymi słowami (bez nazw własnych): {len(df_unofficial)}")
    print(f"[INFO] Liczba różnych nieoficjalnych tokenów: {len(counts_series)}")

    return df_unofficial, counts_series


# ===== 4. PROSTY main =====

def main():
    print("=" * 60)
    print("Czyszczenie korpusu EN–TP na podstawie oficjalnego słownika Toki Pony")
    print("=" * 60)
    print(f"Wejściowy plik TSV:        {IN_TSV_PATH}")
    print(f"Plik słownika Toki Pony:   {TP_VOCAB_CSV_PATH}")

    if not os.path.exists(IN_TSV_PATH):
        raise FileNotFoundError(f"Nie znaleziono wejściowego korpusu: {IN_TSV_PATH}")

    df = pd.read_csv(IN_TSV_PATH, sep="\t")

    if COL_EN not in df.columns or COL_TP not in df.columns:
        raise ValueError(
            f"Oczekiwane kolumny '{COL_EN}' i '{COL_TP}' nie istnieją w pliku TSV. "
            f"Masz kolumny: {list(df.columns)}"
        )

    print(f"Liczba przykładów w korpusie: {len(df)}")

    official_vocab = load_official_tp_vocab(TP_VOCAB_CSV_PATH)

    # --- 1) Korpus tylko z oficjalnymi słowami ---
    print("[INFO] Budowanie korpusu z samymi oficjalnymi tokenami...")
    mask_ok = df[COL_TP].apply(lambda s: sentence_uses_only_official_tokens(s, official_vocab))
    df_official = df[mask_ok].reset_index(drop=True)

    n_ok = len(df_official)
    n_total = len(df)
    n_drop = n_total - n_ok

    print(f"Liczba przykładów po czyszczeniu (oficjalne): {n_ok}")
    print(f"Liczba przykładów odrzuconych (mają coś nieoficjalnego / nazwy własne): {n_drop}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df_official.to_csv(OUT_TSV_OFFICIAL_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"[OK] Zapisano oficjalny korpus do: {OUT_TSV_OFFICIAL_PATH}")

    # --- 2) Korpus z nieoficjalnymi słowami, ale bez nazw własnych ---
    print("\n[INFO] Szukanie zdań z nieoficjalnymi słowami (bez nazw własnych)...")
    df_unofficial, unofficial_counts = extract_unofficial_sentences_and_counts(df, official_vocab, COL_TP)

    df_unofficial.to_csv(OUT_TSV_UNOFFICIAL_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"[OK] Zapisano korpus nieoficjalny do: {OUT_TSV_UNOFFICIAL_PATH}")

    unofficial_counts.to_csv(OUT_UNOFFICIAL_FREQ_CSV, header=["count"])
    print(f"[OK] Zapisano częstości nieoficjalnych tokenów do: {OUT_UNOFFICIAL_FREQ_CSV}")

    # top 20:
    print("\nTop 20 nieoficjalnych tokenów:")
    print(unofficial_counts.head(20))


if __name__ == "__main__":
    main()
