# PDETime
Rethinking Long-Term Multivariate Time Series Forecasting from the perspective of partial differential equations

## Source-preserving reproduction (2026-10-21)

The current entry is [`train_reproduce.py`](train_reproduce.py). It imports the original PDETime/PDETimeLoss and preserves the source sharing, forecast and optimizer coverage; original `train.py` and `src/PDETime/` are unchanged. There is no model/layer or experiment-directory refactor.

- [Audit, parameter comparison, recommended commands and validation](docs/experiments/03_tslib_entry.md)
- [ETTm1 independent script](scripts/long_term_forecast/ETT_script/PDETime_ETTm1.sh): four explicit H=96/192/336/720 commands.
- [Independent dataset scripts](scripts/long_term_forecast): four ETT files plus Electricity, Traffic and Weather files.

```bash
python train_reproduce.py --help
python train_reproduce.py --data ETTm1 --seq_len 480 --pred_len 96 --dry_run
# After placing the CSV under data/ETT-small/:
bash scripts/long_term_forecast/ETT_script/PDETime_ETTm1.sh
```

Paper-disclosed data settings are explicit; unknown settings fall back to `train.py`. Model/loss differences between the paper and source are audited rather than silently rewritten. CPU source-equivalence and ETTm1 smoke checks passed; full GPU/paper-table reproduction is pending. The upstream repository has not been verified as author-official.

## Historical experiments

- [A5000 engineering baseline](docs/experiments/01_a5000_reproduction.md): registered aggregation parameters, which changes the original optimizer coverage.
- [Earlier paper-derived reconstruction](docs/experiments/02_paper_reproduction.md): uses a separately reconstructed PaperPDETime, modified objective and a batch launcher. Preserved for traceability; it is not the source-preserving entry above.
