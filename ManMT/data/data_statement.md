# Data Statement

## Task

- Task type: sentence-level machine translation.
- Direction: Manchu in Latin transliteration to Chinese.

## Language and Domain

- Source language: Manchu, represented as romanized or transliterated text.
- Target language: Chinese.
- Domain: historical-document related language, including names, titles, institutions, and narrative passages.

## Data Size
- No train, development, or test split is included.

## Construction Policy

- The sample was selected from the beginning of the original training TSV file.
- The original row order was preserved.
- Only the `input` and `output` fields are included.

## Intended Use

- Research and educational use for low-resource machine translation, lexical evidence modeling, and artifact inspection.

## Known Limitations

- The folder is a compact sample, not a complete benchmark release.
- The language coverage is domain-focused and should not be treated as general-purpose modern usage.
- Romanization conventions may differ across external resources.
