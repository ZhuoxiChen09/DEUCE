# Demo bundle reproducibility

This record addresses [issue #2](https://github.com/ZhuoxiChen09/DEUCE/issues/2).
It qualifies the README's reproducibility statement: a committed bundle must
validate unchanged across supported checkouts; rebuilding numerical results is
tested within a fixed environment, not promised byte-for-byte across all hosts
or dependency versions.

## Demonstrated failure

The clean Python 3.11 checkout of `d6fb5f3` reproduced 137 passing tests and two
checksum failures. The [Ubuntu CI run](https://github.com/ZhuoxiChen09/DEUCE/actions/runs/42981657185)
failed the same two tests. No Git attributes controlled the bundle's newlines.

Three JSON manifest digests matched the **CRLF expansion of the committed LF
Git blobs**, not the blobs themselves. Below are SHA-256 prefixes; the complete
corrected digests are in the manifest.

| File under `artifacts/demo/` | Old expected = CRLF bytes | Git blob = LF checkout bytes |
| --- | --- | --- |
| `tennis_model_core.pkl.features.json` | `1f44effcf194` | `fb28f23da21d` |
| `walk_forward_report.json` | `09f0fe9de7f6` | `6f5b72572d32` |
| `demo_metrics.json` | `809e8f4256ab` | `ced9fc0cae62` |

An unmodified Windows build reproduced CRLF in all three JSON outputs and
validated locally. Its sidecar exactly matched the old expected digest; its
other numerical outputs differed slightly from the historical build. Python's
default text writer translated LF to CRLF before hashing. Git text cleaning
then stored LF bytes, invalidating the manifest. The dataset, OOS CSV, and
binary model already matched their committed hashes. Only the three proven
CRLF hashes were repaired; the model, dataset, and metric/report files were
not regenerated for this fix.

## Byte contract

- Demo JSON and CSV files are UTF-8 without a BOM, with LF line endings.
  JSON retains the existing two-space indentation and no final newline; CSV
  writers terminate records with LF. No arbitrary JSON reformatting occurs.
- Writers explicitly suppress platform newline translation. Scoped
  `.gitattributes` rules keep these bytes unchanged in Git and checkouts,
  including with `core.autocrlf=true`, `input`, or `false`.
- The serialized model is binary and excluded from Git text conversion.
- SHA-256 continues to hash **raw file bytes**, without normalization at
  validation time. Even a whitespace-only JSON edit or LF-to-CRLF conversion
  must fail validation. An altered pickle must fail too.
- `build_metadata.json` is explicitly volatile and outside the checksum map.
  It contains the timestamp and host information. The manifest does not hash
  itself. All six existing dataset/model/output checksum entries remain.

## Repeated builds and limits

With identical inputs, configuration, random seeds, and numerical environment,
the regression test requires byte-identical semantic text outputs, schemas,
metrics, and text checksums across two builds. It also compares the loaded
models' calibrated probabilities on feature vectors for all 1,232 demo matches.
Each build's own model checksum must validate, but serialized pickle bytes and
therefore their hashes are not required to match between builds. The manifest
comparison excludes only that model digest; volatile metadata is checked
separately.

The requirements contain version ranges. They do not lock Python patch
versions, every dependency, BLAS, or CPU behavior. A rebuild using NumPy 1.26.4,
SciPy 1.17.1, and scikit-learn 1.5.2 differed from the committed historical
metrics by roughly `1e-15` in Brier score/log loss, with identical accuracy and
AUC. Exact numerical or pickle equivalence across different environments is
not guaranteed. This fix does not change the dependency policy or replace
historical artifacts to conceal that difference.

## Verification

Local verification on November 12, 2026 used Windows, Python 3.11.9, and a fresh
virtual environment installed from the **committed** `requirements.txt`.
The primary checkout's modified requirements were not used.

- Baseline: 137 passed, two checksum failures; `demo validate` failed.
- Fixed full suite: 146 passed. Existing dependency deprecation warnings remain.
- Fixed committed bundle: `python -m deuce.cli demo validate` passed.
- Two separate CLI `demo build` processes in a disposable worktree each passed
  `demo validate`. All nonvolatile output files were byte-identical between
  those builds, including the pickle and manifest; timestamps differed.
- `git diff --check` passed. The original checkout's uncommitted files were
  verified against their pre-task SHA-256 hashes and excluded from the fix.
- New tests: real builds, index/blob and fresh-checkout byte comparisons for all
  three `autocrlf` modes, repeated-build comparison, and checksum rejection of
  CRLF conversion, JSON whitespace changes, and a binary byte append.
- The three round-trip cases fail against the old Windows writers even when
  given the new attributes, establishing that attributes alone are insufficient.
- CI now runs the full suite and explicit committed-bundle validation on both
  Ubuntu and Windows with Python 3.11. The linked PR's checks record the actual
  runner results. macOS is not tested.

The first Windows CI run exposed a separate encoding constraint in the existing
Streamlit help smoke test: the runner's legacy output encoding cannot encode
`U+2191` (up arrow). Forcing `cp1252` locally reproduced the same error; UTF-8
output succeeded. CI sets `PYTHONUTF8=1` for both the test process and its child
processes, retaining the full smoke test. For local reproduction on Windows
with a legacy encoding, set `$env:PYTHONUTF8 = "1"` in PowerShell before running
the commands below.

To verify manually, first test the committed bundle before any build. Then run
builds in a separate disposable checkout so they cannot mask committed-bundle
failures or overwrite working artifacts:

```bash
python -m pytest tests -q
python -m deuce.cli demo validate
git worktree add --detach ../DEUCE-demo-check HEAD
cd ../DEUCE-demo-check
python -m deuce.cli demo build
python -m deuce.cli demo validate
python -m deuce.cli demo build
python -m deuce.cli demo validate
```

Use the same Python 3.11 environment in both directories. The repeated-build
regression test performs the semantic comparison; successful validation alone
only establishes each bundle's internal consistency. These checks do not
address evaluation/serving parity or dataset redistribution clearance.
