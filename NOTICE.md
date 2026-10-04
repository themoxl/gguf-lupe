# Third-party notices

GGUF-Lupe's own code is MIT-licensed (see `LICENSE`). This repository also contains material from others:

## llama.cpp (MIT)

- The page `gguf-lupe.html` (view “Code path” / „Rechenweg“) embeds unmodified excerpts of the llama.cpp source code, version **b11388** (`src/ll.b64`).
- The optional Lupe server downloads official llama.cpp release binaries (b11388) from <https://github.com/ggml-org/llama.cpp/releases> into `~/.cache/gguf-lupe` on first use. These binaries are not part of this repository.

Copyright (c) 2023-2026 The ggml authors. License: [`LICENSES/MIT-llama.cpp.txt`](LICENSES/MIT-llama.cpp.txt).

## Model data (Apache-2.0)

Two kinds of files contain data taken from, or computed with, model files:

- the built-in demo: `src/demo.json`, embedded in `gguf-lupe.html`
- the recorded examples: `examples/*.js`

The models:

| Model file | Source | License | Used for |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF), base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0 | demo: GGUF header (metadata, vocabulary), selected weight rows, precomputed neighbors and token map; example run |
| `gemma-4-12B-it-QAT-Q4_0.gguf` | [lmstudio-community/gemma-4-12B-it-QAT-GGUF](https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF) | Apache-2.0 | example run |

How the data was changed:
- The demo data was extracted from the files and partly recomputed: dequantized rows, nearest neighbors, 2-D projection.
- The example runs are intermediate results recorded with llama.cpp while the model answered one question, stored as float16 and gzip-compressed.

License: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Fonts

The page embeds the latin and latin-ext subsets of **Archivo** (Copyright 2020 The Archivo Project Authors) and **JetBrains Mono** (Copyright 2020 The JetBrains Mono Project Authors), as served by Google Fonts (`src/fonts/`). Both are licensed under the SIL Open Font License 1.1: [`LICENSES/OFL-Archivo.txt`](LICENSES/OFL-Archivo.txt), [`LICENSES/OFL-JetBrainsMono.txt`](LICENSES/OFL-JetBrainsMono.txt).
