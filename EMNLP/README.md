# IOCE-Lexical Anonymous Artifact

This artifact combines the key implementation code and a compact dataset sample for the IOCE-Lexical framework. It is intended to show the implementation structure, core method components, and data format for review and inspection.
## Software Contents

The `software/` directory contains the key source code and configuration template:

- `software/src/`: core code for lexical memory construction, input-side training, output-side confidence-head training, CALM decoding, IOCE evaluation, and shared utilities.
- `software/configs/ioce_emnlp.yaml`: parameterized configuration template.
- `software/requirements.txt`: Python dependency list.

Optional software entrypoints:

```bash
python software/src/run_ioce_pipeline.py --config software/configs/ioce_emnlp.yaml
python software/src/build_memory.py --config software/configs/ioce_emnlp.yaml
python software/src/train_input_side.py --config software/configs/ioce_emnlp.yaml
python software/src/train_ioce_head.py --config software/configs/ioce_emnlp.yaml
python software/src/eval_ioce.py --config software/configs/ioce_emnlp.yaml
```

## Data Contents

The `data/` directory contains a 1000-row unsplit sample for an extremely low-resource Manchu-to-Chinese machine translation task:

- `data/data.tsv`: 1000 sentence pairs in a single unsplit TSV file.
- `data/data_statement.md`: dataset documentation.
- `data/preprocessing.md`: selection and packaging procedure.
- `data/license_or_access_note.md`: access and usage note.

The TSV file has one header row:

- `input`: Manchu sentence in Latin transliteration.
- `output`: Chinese translation.

Dataset size and split policy:

- Total rows: 1000.
- No train, development, or test split is provided in this artifact.

