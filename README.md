# Sprawozdanie z projektu: Deep Learning Inference Optimization

**Autor:** Stanisław Wałga
**Repozytorium:** `dl-inference-optimization` (GitHub)
**Sprzęt testowy:** NVIDIA GeForce RTX 2060 (CUDA 13.1, sterownik), PyTorch 2.13.0+cu130
**Model referencyjny:** ResNet18 (torchvision, wagi IMAGENET1K_V1, 11 689 512 parametrów)

---

## 1. Cel projektu

Projekt odtwarza — w mniejszej skali — trzy zadania typowe dla inżynierii wydajności modeli AI w środowisku produkcyjnym:

1. **Pomiar i poprawa szybkości inferencji** na akceleratorach GPU.
2. **Profilowanie obciążeń DL** w celu identyfikacji wąskich gardeł.
3. **Wdrożenie modelu jako usługi produkcyjnej** (REST API) z uwzględnieniem realnych wzorców ruchu.

Model referencyjny (ResNet18) pełni rolę obciążenia testowego — celem nie jest jakość klasyfikacji, tylko charakterystyka wydajnościowa i metodologia jej badania, przenaszalna na inne architektury.

---

## 2. Metodologia pomiaru

Zastosowano dwie metody pomiaru czasu, celowo porównane ze sobą:

| Metoda | Zastosowanie | Uwagi |
|---|---|---|
| `time.time()` | CPU, pomiar orientacyjny | Zawodny na GPU ze względu na asynchroniczne wykonanie |
| `torch.cuda.Event` + `torch.cuda.synchronize()` | GPU, pomiar precyzyjny | Gwarantuje odczyt po faktycznym zakończeniu obliczeń na karcie |

Każdy pomiar poprzedzony był fazą **rozgrzewki** (10 wywołań pominiętych w statystyce) — pierwsze wywołanie modelu jest systematycznie wolniejsze (alokacja pamięci, inicjalizacja kerneli CUDA) i zniekształcałoby wynik średni. Każdy właściwy pomiar to średnia z 100 powtórzeń.

Weryfikacja krzyżowa obu metod na tym samym obciążeniu dała zbieżne wyniki (4.25–4.67 ms), co potwierdza wiarygodność pomiarów prostszą metodą tam, gdzie precyzyjna nie była jeszcze zaimplementowana.

---

## 3. Wyniki: baseline CPU vs GPU

| Rozmiar batcha | CPU (ms) | GPU (ms) | Przyspieszenie |
|---|---|---|---|
| 1 | 38,86–39,72 | 4,25–4,67 | ~9,1× |
| 32 | 1011,46 | 25,19–25,70 | ~40,2× |

**Obserwacja:** przewaga GPU rośnie nieliniowo wraz z rozmiarem batcha — CPU skaluje się niemal liniowo z liczbą obrazów, GPU wykorzystuje równoległość sprzętową i rośnie znacznie wolniej. To pierwszy sygnał, że GPU tej klasy jest niedociążone przy pojedynczych zapytaniach.

---

## 4. Profilowanie (`torch.profiler`)

### CPU — rozkład czasu wykonania

| Operacja | Udział w czasie całkowitym |
|---|---|
| `aten::mkldnn_convolution` | 81–85% |
| `aten::max_pool2d_with_indices` | 6–9% |
| `aten::batch_norm` (i pochodne) | 4–5% |
| pozostałe | <2% |

Konwolucje jednoznacznie dominują — dowolna optymalizacja CPU powinna koncentrować się wyłącznie na tej operacji.

### GPU — rozkład czasu wykonania

| Operacja | Udział w czasie całkowitym |
|---|---|
| `aten::cudnn_convolution` | 25–35% |
| `aten::cudnn_batch_norm` | 22–26% |
| narzut zarządzania (`empty`, `view`, `empty_like`) | ~10–15% |

**Wniosek:** na GPU rozkład jest znacznie bardziej równomierny. Konwolucje przestają dominować (cuDNN wykonuje je bardzo wydajnie), a proporcjonalny udział normalizacji wsadowej oraz narzutu zarządzania pamięcią rośnie. Oznacza to, że optymalizacja pod GPU wymaga innego podejścia niż pod CPU — nie wystarczy przyspieszyć jednej operacji.

---

## 5. Testy technik optymalizacji

Przetestowano cztery standardowe techniki optymalizacji inferencji, każdą względem tego samego baseline'u (`torch.cuda.Event`, fp32, bez optymalizacji).

### Przy batch_size = 1

| Technika | Wynik | Zmiana względem baseline |
|---|---|---|
| `torch.compile()` | 5,06–5,19 ms | 0,86–0,95× (**wolniej**) |
| `.half()` (fp16 ręczne) | 6,25–6,44 ms | 0,72× (**wolniej**) |
| `torch.autocast` (mixed precision) | 5,70–5,82 ms | 0,78–0,80× (**wolniej**) |
| `torch.backends.cudnn.benchmark` | 4,59 ms | 0,99× (bez zmiany) |

**Żadna z czterech technik nie przyniosła poprawy.** Wynik ten, choć sprzeczny z intuicją opartą na dokumentacji tych narzędzi, jest w pełni wytłumaczalny (patrz sekcja 6).

### Przy batch_size = 32

| Technika | Wynik | Zmiana względem baseline |
|---|---|---|
| `torch.compile()` | — | 0,79× (nadal wolniej) |
| `.half()` (fp16 ręczne) | — | **1,99×** (szybciej) |
| `torch.autocast` (mixed precision) | — | **2,01×** (szybciej) |
| `torch.backends.cudnn.benchmark` | — | 1,03× (marginalnie) |

**Kluczowy wynik projektu:** fp16 i mixed precision, bezużyteczne przy pojedynczych zapytaniach, dają blisko dwukrotne przyspieszenie przy większym obciążeniu równoległym.

---

## 6. Analiza przyczynowa

Trzy czynniki tłumaczą powyższe wyniki:

1. **Niedociążenie GPU przy batch=1.** Pojedyncze zapytanie nie generuje wystarczającej liczby równoległych operacji, by zrekompensować narzut zarządzania precyzją (konwersja fp32↔fp16) czy kompilacją grafu (`torch.compile`). Narzut administracyjny przewyższa zysk obliczeniowy.

2. **Generacja architektury GPU.** RTX 2060 (Turing, 2018) ma rdzenie Tensor pierwszej generacji, o istotnie mniejszej przepustowości niż układy Ampere/Ada/Hopper, na których dokumentowane są największe zyski z mixed precision. Komunikat profilera `Not enough SMs to use max_autotune_gemm mode` bezpośrednio potwierdza ograniczenie sprzętowe względem trybów agresywnej optymalizacji `torch.compile`.

3. **Skala modelu.** ResNet18 (11,7 mln parametrów) jest już blisko optymalnie obsługiwany przez natywny cuDNN w fp32. Techniki optymalizacyjne ujawniają swoją wartość silniej przy większych, głębszych architekturach, gdzie koszt administracyjny jest relatywnie mniejszy wobec objętości obliczeń.

**Wniosek metodologiczny:** efektywność technik optymalizacji inferencji nie jest właściwością uniwersalną — zależy łącznie od sprzętu, rozmiaru modelu i rzeczywistego wzorca obciążenia. Decyzje optymalizacyjne wymagają pomiaru w warunkach zbliżonych do produkcyjnych, nie ekstrapolacji z dokumentacji ani wyników referencyjnych publikowanych na innym sprzęcie.

---

## 7. Wdrożenie produkcyjne: API z dynamicznym batchowaniem

### Problem projektowy

Wynik z sekcji 5 rodzi praktyczny konflikt: typowe zapytanie API obsługuje pojedynczy obraz (batch=1) — dokładnie scenariusz, w którym fp16/autocast nie dają korzyści. Zbudowanie API bez uwzględnienia tego faktu skutkowałoby architekturą, która nigdy nie realizuje zmierzonego potencjału przyspieszenia.

### Rozwiązanie: dynamic batching

Zaimplementowano wzorzec stosowany w produkcyjnych serwerach inferencji (koncepcyjnie zbliżony do Triton Inference Server / TorchServe):

```
Zapytania klientów (asynchroniczne, pojedyncze)
        │
        ▼
   queue.Queue()  ──►  wątek w tle (daemon)
        │                    │
        │        zbiera zapytania przez okno czasowe
        │        (max 50 ms lub do MAX_BATCH_SIZE=32)
        │                    │
        │        łączy w jeden tensor wsadowy
        │        inferencja w torch.autocast (fp16)
        │                    │
        │        rozdziela wyniki do poszczególnych
        │        zapytań (threading.Condition)
        ▼                    ▼
   odpowiedź 1  ◄──────  wynik dla zapytania 1
   odpowiedź 2  ◄──────  wynik dla zapytania 2
   ...
```

**Elementy implementacji:**
- `queue.Queue` — bezpieczna wątkowo kolejka przyjmująca zapytania
- wątek `daemon` działający niezależnie od cyklu żądanie–odpowiedź Flaska
- `threading.Condition` — synchronizacja bez aktywnego odpytywania (busy-waiting)
- `app.run(threaded=True)` — obsługa wielu równoczesnych połączeń klienckich

### Wynik testu obciążeniowego

Test: 10 równoczesnych zapytań (osobne wątki klienckie, wysłane w tym samym momencie).

| Metryka | Wartość |
|---|---|
| Liczba zapytań zebranych w jeden batch | 10 / 10 (100%) |
| Czas inferencji całego batcha | 268,68 ms |
| Czas przetwarzania per obraz w batchu | ~26,9 ms |
| Całkowity czas odpowiedzi (round-trip) per zapytanie | 346–375 ms |

**Interpretacja:** wszystkie równoczesne zapytania zostały poprawnie połączone w jeden batch i przetworzone wspólnie z wykorzystaniem `torch.autocast`, realizując zmierzone w sekcji 5 przyspieszenie. Różnica między czasem samej inferencji (268 ms) a pełnym round-trip (~355 ms) wynika z czasu oczekiwania w oknie batchowania oraz narzutu komunikacji HTTP — jest to świadomy kompromis: **wzrost przepustowości systemu (throughput) kosztem opóźnienia pojedynczego zapytania (latency)**, typowy dla architektur batchujących w produkcji.

---

## 8. Ograniczenia projektu

- Testy przeprowadzono na pojedynczym GPU klasy konsumenckiej (RTX 2060); wyniki ilościowe nie ekstrapolują się wprost na karty klasy data center (A100, H100).
- Okno batchowania (50 ms) i maksymalny rozmiar batcha (32) dobrano empirycznie na potrzeby demonstracji; produkcyjne strojenie wymagałoby analizy rozkładu ruchu (traffic pattern) i akceptowalnego SLA opóźnienia.
- `torch.compile()` nie został poddany głębszej diagnostyce przyczyn braku poprawy (np. analizie wygenerowanego kodu Triton) — potencjalny kierunek dalszej pracy.
- Test API wykorzystuje syntetyczne dane wejściowe (losowy tensor) — środowisko produkcyjne wymagałoby walidacji z rzeczywistymi obrazami i obsługą błędnych/niepoprawnych danych wejściowych.

---

## 9. Wnioski końcowe

1. GPU zapewnia 9–40-krotne przyspieszenie względem CPU dla tego obciążenia, przy czym przewaga rośnie z rozmiarem batcha.
2. Standardowe techniki optymalizacji inferencji (mixed precision, `torch.compile`, autotuning cuDNN) **nie działają uniwersalnie** — ich skuteczność zależy od rozmiaru batcha, architektury GPU i skali modelu, i wymaga empirycznej weryfikacji w warunkach zbliżonych do docelowych.
3. Przy odpowiednio dobranym obciążeniu (batch=32) mixed precision (fp16/autocast) dała powtarzalne, blisko dwukrotne przyspieszenie.
4. Świadomość rozbieżności między charakterystyką pojedynczego zapytania a charakterystyką zapytań wsadowych pozwoliła zaprojektować architekturę wdrożeniową (dynamic batching), która realizuje zmierzony potencjał optymalizacji również przy typowym, pojedynczym ruchu API — bez tego mechanizmu zysk z fp16 pozostałby czysto teoretyczny.
