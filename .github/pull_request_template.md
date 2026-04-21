## Linked Issue

Closes #<!-- issue number -->

## Summary of Change

<!-- 2-4 sentences: what does this PR do and why? -->

## Type of Change

- [ ] `feat` — new pipeline or feature
- [ ] `fix` — bug fix
- [ ] `refactor` — code restructuring without behaviour change
- [ ] `test` — tests only
- [ ] `docs` — documentation only
- [ ] `chore` — tooling, deps, config

## Pipeline(s) Affected

<!-- Tick every Kedro pipeline this PR touches -->
- [ ] `data_quality`
- [ ] `data_cleaning`
- [ ] `data_feat_engineering`
- [ ] `data_split`
- [ ] `model_train`
- [ ] `model_selection`
- [ ] `model_predict`
- [ ] `data_drifts`
- [ ] `api`
- [ ] `streamlit_app`
- [ ] `ci` / `config`

## Tests Added or Updated

<!-- List the test files/functions added -->
- `tests/pipelines/test_<name>.py::test_<function>`

## Time Series Safety

If this PR touches feature engineering or model training, confirm:
- [ ] No `train_test_split` with `shuffle=True`
- [ ] Rolling/lag features computed inside the training window (not on full series)
- [ ] Ran `/check-leakage` on affected nodes — no issues found

## How to Test Locally

```bash
# commands to reproduce the run
kedro run --pipeline <name>
# or
pytest tests/pipelines/test_<name>.py -v
```

## Screenshots / Run Logs

<!-- Paste MLflow run URL, terminal output, or screenshot of the Streamlit view if relevant -->

## Reviewer Checklist

- [ ] Code follows project conventions in `CONTRIBUTING.md`
- [ ] All CI checks are green
- [ ] No secrets, raw data, or `mlruns/` committed
- [ ] PR description is complete
