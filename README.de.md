# GGUF-Lupe

[English](README.md) · **Deutsch**

Ein einsteigerfreundlicher Blick in jede `.gguf`-Modelldatei: was drinsteht und wie das Modell rechnet.

![GGUF-Reader](docs/reader.png)

## Ausprobieren

**Online:** <https://themoxl.github.io/gguf-lupe/>. Es ist dieselbe Seite: deine Datei bleibt in deinem Browser.

**Oder lokal:**

1. `gguf-lupe.html` im Browser öffnen.
2. Eine `.gguf`-Datei ins Fenster ziehen oder auf **Datei öffnen** klicken.

- Nichts wird hochgeladen. Die Seite liest nur, was sie gerade anzeigt; die Dateigröße spielt also keine Rolle.
- Keine Installation, kein Server, keine Grafikkarte. Funktioniert offline.
- Ohne Datei zeigt die Seite ein eingebautes Beispiel: Qwen3.8-27B (Header, ausgewählte Zeilen, komplette Landkarte).
- Deutsch oder Englisch, je nach Browsersprache. Umschalten mit **DE/EN** oben in der Leiste.
- Erklärungen stecken hinter den **ⓘ**-Knöpfen.

## Was du siehst

**GGUF-Reader** · *Was steht in der Datei?*

- **Überblick**
- **Tokens**: Wörterbuch, Merge-Regeln, Spezial-Tokens, einen Text Schritt für Schritt zerlegen
- **Embedding-Tabelle**
- **Gewichte**: jede einzelne Gewichtstabelle, als Zahlen
- **Tensoren**: das Tensorverzeichnis
- **Metadaten**: alle Schlüssel und Werte

**Architektur** · *Wie rechnet das Modell?*

- **Datenfluss**: der Weg durch die Blöcke
- **Durchlauf**: eine Frage, jeder Rechenschritt (live mit dem Server oder als aufgezeichnetes Beispiel)
- **Bedeutung**: ähnliche Wörter, Wort-Rechnen (King − man + woman), Landkarte aller Tokens
- **Rechenweg**: dieselbe Rechnung im Quelltext von llama.cpp
- **Rundgang**: dein Satz Schritt für Schritt durch das Modell, im Vollbild

## Optional: der Lupe-Server

Ein kleines Python-Programm auf deinem eigenen Rechner, das auf der Grafikkarte oder der CPU rechnet. Dazu kommen:

- **Durchlauf live**: eine Frage stellen und jeden Rechenschritt für jedes Token verfolgen
- **Komplette Landkarte** für jedes Modell (einmal berechnet, danach gespeichert)
- **Dimensions-Schichten** auf der Karte: jede Spalte der Embedding-Tabelle oder die Hauptrichtungen
- **Ähnliche Wörter und Wort-Rechnen** über alle Tokens, in Sekundenbruchteilen
- **Modellliste**: die `.gguf`-Dateien in deinen Modell-Ordnern, mit einem Klick geöffnet

### Was du brauchst

- Python 3.9 oder neuer mit `numpy` und `gguf`: `pip install numpy gguf`
- Optional [PyTorch](https://pytorch.org), für NVIDIA-Grafikkarten (CUDA), Apple-Chips (MPS) oder eine schnellere CPU

Mit conda:

```
conda create -n gguf-lupe python=3.12 numpy
conda activate gguf-lupe
pip install gguf
```

Eine Grafikkarte ist optional: Die CPU liefert dasselbe Ergebnis, nur langsamer (komplette Landkarte von Qwen3.8-27B: ca. 2 min auf einer RTX 5090, ca. 24 min auf 16 CPU-Kernen).

### Starten

- **Windows:** `start-lupe-server.cmd` doppelklicken (findet auch eine conda-Umgebung namens `gguf-lupe`)
- **macOS / Linux:** `./start-lupe-server.sh`

Der Browser öffnet `http://127.0.0.1:8765/`. Oben wählst du GPU oder CPU sowie dein Modell und öffnest es. Bis dahin zeigt die Seite das eingebaute Beispiel.

```
python src/lupe_server.py [--port 8765] [--device auto|gpu|cuda|mps|cpu] [--models ORDNER ...] [--no-browser]
```

`--models` ergänzt weitere Ordner. Automatisch durchsucht werden `~/.lmstudio/models`, `~/.cache/lm-studio/models`, `~/.cache/huggingface/hub`, `~/models` und `~/Downloads`. Die Startdateien reichen Optionen durch. Anderes Python: `LUPE_PYTHON` oder `lupe-python.txt` (siehe Startdateien).

### Durchlauf und llama.cpp

Beim ersten Mal bietet der Tab **Durchlauf** Knöpfe an, die die offizielle llama.cpp-Version **b11388** (dieselbe, deren Quelltext der Tab **Rechenweg** zeigt) von GitHub nach `~/.cache/gguf-lupe/llama.cpp-b11388/` laden:

| Variante | Für | Größe |
|---|---|---|
| CUDA | NVIDIA-Grafikkarten | ca. 580 MB |
| Vulkan | Grafikkarten von NVIDIA, AMD und Intel | ca. 33 MB |
| CPU | Rechner ohne Grafikkarte | ca. 19 MB |

Von Hand: das Archiv vom [Release b11388](https://github.com/ggml-org/llama.cpp/releases/tag/b11388) in einen Ordner `llama.cpp` neben `gguf-lupe.html` oder in den Cache-Ordner oben entpacken. Für CUDA kommt das Archiv `cudart-…` in denselben Ordner.

Das Modell nimmt immer das wahrscheinlichste Token (dieselbe Frage, dieselbe Antwort; Frage und Antwort zusammen höchstens 100 Tokens). Dann wird der ganze Text noch einmal gerechnet und jeder Schritt des echten Rechengraphen von llama.cpp mitgeschrieben. Du siehst:

- den Weg jedes Tokens durch alle Schichten (3-D-Stapel) und die Logit-Linse (die Vorhersage nach jeder Schicht)
- die Aufmerksamkeit je Kopf und das nächste Token mit Wahrscheinlichkeiten
- jeden Rechenschritt eines Blocks; ein Klick auf eine Zahl rechnet sie nach: Eingaben × Gewichtszeile aus der Datei

![Durchlauf](docs/forward-pass.png)

## Welche Dateien?

Jede `.gguf`-Datei, egal wie groß. Getestet mit Sprachmodellen (Qwen3.8 27B, Gemma 4 12B) und Bild-Encodern (mmproj-Dateien). Grenzen:

- **Gewichte als Zahlen:** alle gängigen Formate (F32, F16, BF16, Q4_0 bis Q8_0, Q2_K bis Q6_K, IQ4_NL, IQ4_XS, MXFP4, NVFP4 …). IQ1, IQ2, IQ3, TQ1_0, Q1_0: nur Form und Größe, noch ohne Zahlen.
- **Text Schritt für Schritt zerlegen:** BPE-Tokenizer (Qwen, Llama 3, Mistral Nemo, GPT-artige …) und Gemma 4. Bei anderen Tokenizern nur das Wörterbuch.
- **Durchlauf:** jede Architektur, die llama.cpp b11388 kennt (rund 150). Bild-Encoder und andere Dateien ohne Wörterbuch lassen sich lesen, aber nicht befragen.
- **Aufgeteilte Modelle** (`…-00001-of-00003.gguf`): Jeder Teil lässt sich einzeln lesen. Für den Durchlauf den ersten Teil öffnen.

## Aufgezeichnete Beispiele

Echte, mit llama.cpp aufgezeichnete Durchläufe, oben im Tab **Durchlauf**. Zum Ansehen genügt die Seite, ohne Server und Grafikkarte. Mitgeliefert: Durchläufe auf Deutsch und Englisch. Der Ordner `examples/` gehört neben `gguf-lupe.html`.

**Eigene Beispiele:** Mit dem Server eine Frage durchrechnen und auf **Als Beispiel speichern** klicken. Die Datei (etwa 5 bis 12 MB) landet in `examples/` und erscheint in der Liste. Oder bei laufendem Server (das Modell muss in einem seiner Modell-Ordner liegen):

```
python src/make_example.py "Frage" --model PFAD-ZUR-DATEI.gguf
```

## Datenschutz und Speicherorte

- Der Server ist nur auf deinem Rechner erreichbar (`127.0.0.1`) und liest nur `.gguf`-Dateien in seinen Modell-Ordnern. Nichts wird hochgeladen oder kopiert.
- Die Seite lädt nichts aus dem Internet (auch die Schriften sind eingebaut). Der Server geht nur online, um llama.cpp herunterzuladen, wenn du darauf klickst.

| Was | Wo |
|---|---|
| Berechnete Landkarten | `~/.cache/gguf-lupe/*.lupe-map.bin`, eine je Modelldatei; jederzeit löschbar |
| llama.cpp | `~/.cache/gguf-lupe/llama.cpp-b11388/` (oder `llama.cpp/` neben `gguf-lupe.html`) |
| Letzte 30 Modelle, Rechengerät, Sprache | Browserspeicher (`localStorage`): `lupe-model-history`, `lupe-device`, `lupe-lang` |
| Aufgezeichnete Durchläufe | nur im Arbeitsspeicher (die letzten 4); auf die Festplatte nur als gespeichertes Beispiel |

`~` ist dein Benutzerordner (Windows: `C:\Users\<Name>`).

## Selbst bauen

`gguf-lupe.html` wird aus `src/` zusammengesetzt. Dafür brauchst du Node.js, keine Pakete:

```
cd src
node build.cjs ../gguf-lupe.html
```

## Herkunft und Lizenzen

- Eigener Code: MIT, siehe [LICENSE](LICENSE).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, © 2023-2026 The ggml authors): Der Tab **Rechenweg** enthält unveränderte Quelltext-Dateien der Version b11388; der Lizenztext ist in der Seite einsehbar. Tokenizer, Entpacken der Gewichte und Rechenweg folgen derselben Version. Die llama.cpp-Programme für den Durchlauf sind offizielle Releases, die erst zur Laufzeit von GitHub geladen werden; sie sind nicht Teil dieses Repos.
- Das eingebaute Beispiel und die aufgezeichneten Durchläufe enthalten Daten, die aus Qwen3.8-27B und Gemma 4 12B abgeleitet sind (beide Apache-2.0, GGUF-Dateien von lmstudio-community).
- Schriften: Archivo und JetBrains Mono (SIL Open Font License 1.1), in die Seite eingebettet.
- Alle Hinweise zu Fremdkomponenten: [NOTICE.md](NOTICE.md).
