# Toki-Pona-Praca-Magisterska-Zalaczniki

Załączniki do Pracy Magisterskiej "Budowa i ewaluacja systemu tłumaczącego z języka angielskiego na niskozasobowy język sztuczny z wykorzystaniem uczenia transferowego i dedykowanej wektoryzacji, na przykładzie języka Toki Pona"

## Struktura repozytorium

### 📁 `data/`
Katalog zawierający zbiory danych treningowych, testowych oraz słowniki używane w projekcie.

**Pliki:**
- **`eng_toki_pona.tsv`** - Główny korpus równoległy angielski-Toki Pona w formacie TSV.
- **`eng_toki_pona_final_training.tsv`** - Finalny zbiór treningowy po oczyszczeniu i filtracji.
- **`eng_toki_pona_no_names.tsv`** - Korpus bez nazw własnych.
- **`eng_toki_pona_official_vocab.tsv`** - Korpus zawierający tylko oficjalne słownictwo Toki Pona.
- **`eng_toki_pona_official_vocab1.tsv`** - Alternatywna wersja korpusu z oficjalnym słownictwem.
- **`eng_toki_pona_unofficial_no_names.tsv`** - Korpus z nieoficjalnym słownictwem, bez nazw własnych.
- **`toki_pona.csv`** - Dane w formacie CSV z frazami w języku Toki Pona.
- **`unofficial_tokens_freq.csv`** - Częstotliwości występowania nieoficjalnych tokenów.
- **`test_set_100_en_tp.json`** - Zbiór testowy składający się ze 100 par zdań angielski-Toki Pona w formacie JSON.
- **`evaluation_dataset.xlsx`** - Plik Excel z tłumaczeniami dokonanymi przed każdy z modeli, a także zawierający manualną ocenę stworzonych tłumaczeń, gdzie przyjęto następującą skalę ocen: 0 - tłumaczenie całkowicie niepoprawne, 1 - tłumaczenie bliskie poprawności, jednak posiadające pewne błędy, 2 - tłumaczenie w pełni poprawne, identyczne z danymi treningowymi, a także tłumaczenie zawierające minimalną halucynację, która po usunięciu nie wpływa na znaczenie reszty zdania.

#### 📁 `data/predictions/`
Katalog zawierający predykcje wygenerowane przez wytrenowane modele.

**Pliki:**
- **`full_epoch3_predictions.json`** - Predykcje modelu pełnego (full) po 3 epokach treningu.
- **`full_epoch6_predictions.json`** - Predykcje modelu pełnego po 6 epokach treningu.
- **`full_epoch9_predictions.json`** - Predykcje modelu pełnego po 9 epokach treningu.
- **`small_epoch3_predictions.json`** - Predykcje modelu małego (small) po 3 epokach treningu.
- **`small_epoch6_predictions.json`** - Predykcje modelu małego po 6 epokach treningu.
- **`small_epoch9_predictions.json`** - Predykcje modelu małego po 9 epokach treningu.

### 📁 `models/` - **W związku z ograniczeniem dotyczącym rozmiaru plików, modele nie zostały dodane do repozytorium. W folderach modeli znajdują się jednak pliki parametryzacyjne**
Katalog zawierający wytrenowane modele MT5 (Multilingual T5) dla tłumaczenia z angielskiego na Toki Pona.

#### Modele pełne (full):
- **`mt5_en_tp_full_final_epoch3/`** - Model pełny po 3 epokach treningu.
- **`mt5_en_tp_full_final_epoch6/`** - Model pełny po 6 epokach treningu.
- **`mt5_en_tp_full_final_epoch9/`** - Model pełny po 9 epokach treningu.

#### Modele małe (small):
- **`mt5_en_tp_small_final_epoch3/`** - Model mały po 3 epokach treningu.
- **`mt5_en_tp_small_final_epoch6/`** - Model mały po 6 epokach treningu.
- **`mt5_en_tp_small_final_epoch9/`** - Model mały po 9 epokach treningu.

**Zawartość każdego katalogu modelu:**
- `config.json` - Konfiguracja modelu.
- `model.safetensors` - Wagi modelu w formacie SafeTensors. - niedostępne w repozytorium
- `tokenizer_config.json` - Konfiguracja tokenizera.
- `spiece.model` - Model SentencePiece do tokenizacji. - niedostępne w repozytorium
- `special_tokens_map.json` - Mapowanie tokenów specjalnych.
- `added_tokens.json` - Dodatkowe tokeny specyficzne dla zadania.
- `generation_config.json` - Konfiguracja parametrów generowania tekstu.

### 📁 `src/`
Katalog zawierający kod źródłowy projektu - skrypty Python do treningu, ewaluacji i wizualizacji modeli.

**Pliki:**

#### Przygotowanie danych:
- **`prepare_final_tp_corpus.py`** - Skrypt do przygotowania finalnego korpusu Toki Pona, filtracji i czyszczenia danych.
- **`clean_by_official_vocab.py`** - Czyszczenie korpusu z wykorzystaniem oficjalnego słownictwa Toki Pona.
- **`clean_new.py`** - Alternatywny skrypt do czyszczenia i przetwarzania danych.

#### Trening i predykcja:
- **`train_en_tp_mt5_clean.py`** - Główny skrypt treningowy dla modelu MT5 na zadaniu tłumaczenia angielski-Toki Pona.
- **`generate_predictions_for_all_models.py`** - Skrypt do generowania predykcji dla wszystkich wytrenowanych modeli na zbiorze testowym.
- **`translator.py`** - Moduł zawierający klasę translatora do ładowania i używania wytrenowanych modeli.

#### Ewaluacja:
- **`evaluate_bleu.py`** - Skrypt do obliczania metryki BLEU dla oceny jakości tłumaczeń.
- **`analyze_evaluation.py`** - Analiza wyników ewaluacji, porównanie różnych modeli i epok treningu.
- **`merge_predictions_to_excel.py`** - Łączenie predykcji z różnych modeli i eksport do formatu Excel dla łatwiejszej analizy.

#### Wizualizacja:
- **`visualize_tp_embeddings_tsne.py`** - Wizualizacja embeddingów Toki Pona przy użyciu algorytmu t-SNE.
- **`visualize_joint_embeddings.py`** - Wizualizacja wspólnych embeddingów dla języka angielskiego i Toki Pona.
- **`visualize_joint_embeddings_3d.py`** - Wizualizacja wspólnych embeddingów w przestrzeni 3D.

### 📁 `analysis/`
Katalog przeznaczony na wyniki analiz, wykresy i raporty generowane podczas ewaluacji modeli.

## Wymagania

Projekt wykorzystuje następujące główne biblioteki:
- `transformers` - do pracy z modelami MT5
- `torch` - framework do uczenia głębokiego
- `datasets` - do zarządzania zbiorami danych
- `sacrebleu` - do obliczania metryki BLEU
- `scikit-learn` - do wizualizacji (t-SNE)
- `matplotlib`, `seaborn` - do tworzenia wykresów

## Opis projektu

Projekt koncentruje się na budowie systemu tłumaczenia maszynowego z języka angielskiego na Toki Pona - minimalny język sztuczny składający się z około 120-137 słów. Ze względu na niskozasobowy charakter języka, wykorzystano podejście oparte na uczeniu transferowym z użyciem przedtrenowanego modelu mT5 (Multilingual T5).

Przebadano wpływ różnych rozmiarów modeli (small vs full) oraz liczby epok treningu (3, 6, 9) na jakość tłumaczeń.

## Uwagi

Należy dostosować ścieżki do ponownego użycia kodu, a także odpowiednio skonfigurować lokalne środowisko wirtualne, w celu zachowania poprawności działania kodu. Wersje bibliotek zostały zapisane w pliku requirements.txt

