# PDETime
Rethinking Long-Term Multivariate Time Series Forecasting from the perspective of partial differential equations

## Reproduction entry points

- [Paper-derived runner and recommended commands](docs/experiments/02_paper_reproduction.md): paper-disclosed settings with explicit `train.py` fallbacks; seven datasets and horizons 96/192/336/720.
- [`scripts/reproduce_paper.py`](scripts/reproduce_paper.py): parameterized training, validation-only search and checkpoint evaluation.
- [`scripts/reproduce_paper.sh`](scripts/reproduce_paper.sh): smoke / fixed / search commands; use `--dry-run` to inspect.
- [Earlier A5000 engineering baseline](docs/experiments/01_a5000_reproduction.md): preserved for comparison.

```bash
bash scripts/reproduce_paper.sh smoke --dry-run
# After placing the CSVs under data/:
bash scripts/reproduce_paper.sh smoke
DATASETS=ETTm1 HORIZONS=96 REPEATS=1 bash scripts/reproduce_paper.sh fixed
```

Status: CPU mechanism tests and small ETTm1 end-to-end runs passed; full GPU/paper-table reproduction is pending. The upstream implementation has not been verified as author-official.
