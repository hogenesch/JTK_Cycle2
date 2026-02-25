# Data

This repo uses GEO dataset **GSE11923** (mouse liver, 48 hourly points) as a reproducible benchmark.

## Why this dataset
- Dense 48-point time course useful for method development
- Supports downsampling experiments (48 -> 24 -> 12 points)

## Source
- GEO accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE11923
- Platform annotation: GPL1261 (Mouse430_2)

## Reproduce local download

```bash
mkdir -p ~/Downloads/GSE11923
cd ~/Downloads/GSE11923
curl -L "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE11923&format=file" -o GSE11923_RAW.tar
curl -L "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11923/matrix/GSE11923_series_matrix.txt.gz" -o GSE11923_series_matrix.txt.gz
curl -L "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz" -o GPL1261.annot.gz
```

Then run:

```bash
python scripts/geo_gse11923_preview.py --workdir ~/Downloads/GSE11923 --topn 1000
```

## Files included in-repo

- `GSE11923_expression_CT_top1000var.tsv`
  - lightweight processed subset (top 1000 by variance)
  - included for quick demo/testing and downsampling examples

## Not included in git

- Raw CEL tarballs (`GSE11923_RAW.tar`) and extracted CEL files (large)
