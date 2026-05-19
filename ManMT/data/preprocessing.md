# Preprocessing and Packaging

## Row Selection

- Read `train.tsv` as UTF-8 text.
- Preserved the header `input<TAB>output`.
- Preserved the original row order.

## Output Format

- Exported a single UTF-8 TSV file named `data.tsv`.
- Did not create train, development, or test splits.
- Did not apply additional normalization, tokenization, or text rewriting.

## Integrity Checks

- `data.tsv` contains exactly one header row and 1000 data rows.
- The header is exactly `input<TAB>output`.
- No split files are included in this folder.
