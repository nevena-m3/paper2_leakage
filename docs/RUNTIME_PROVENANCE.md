# Runtime provenance

## Historical Goal-3 Stage-E execution

The frozen Stage-E runtime audit records the environment actually used for the controlled waveform experiment on 2026-09-02:

- Python: `3.14.2`
- executable: `C:\Users\musikicn\Desktop\Nevena_project\Paper_2_Leakage\Code\.venv\Scripts\python.exe`
- `silero-vad==6.2.1`
- `onnxruntime==1.29.0`
- `gammatone==1.0.3`
- `torch==2.13.0`
- `ffmpeg`: `C:\ffmpeg\bin\ffmpeg.EXE`
- `ffprobe`: `C:\ffmpeg\bin\ffprobe.EXE`
- automatic installation of the three governed Python measurement dependencies was enabled by the Stage-E notebook.

The Stage-E notebook also ran a live Silero ONNX model-load smoke test before cohort execution. The final Stage-E response contains 14,792/14,792 expected rows and zero measurement-unavailable rows.

## Documented Python compatibility divergence

The pinned Paper-1 measurement repository at commit

`cb31fb6886df1b2b2fedba4ffbbf8624bd56d7e8`

declares `requires-python = ">=3.11,<3.13"` in `pyproject.toml`. The historical Stage-E run nevertheless executed under Python 3.14.2 because Stage E imported the pinned source implementation directly and its runtime gate checked the critical Python packages and external executables, but did not programmatically reject the Python interpreter version.

This is a reproducibility/provenance divergence and must not be rewritten as though the historical Stage-E run used Python 3.11 or 3.12.

It does not alter the already-frozen numerical outputs. The run completed its live dependency/smoke gates, all 14,792 measurement rows, the target-Q ordering/direction gates, the participant bootstrap package, and the downstream model/figure seals. However, a strict clean-room reproduction under the package's declared supported Python range would require re-executing the controlled measurement pipeline under Python 3.11 or 3.12 and comparing the resulting frozen outputs.

## Reconstruction environment

For future supported reconstruction, use `environment-goal3-supported.yml`, which uses Python 3.12 and the critical pinned Paper-1 measurement dependencies. This is a supported reconstruction recipe, not a claim that it was the historical Stage-E environment.

`requirements-goal3-stageE-historical-critical.txt` records the exact critical Python package versions preserved by the Stage-E runtime audit. It is not a complete transitive historical lockfile because a full `pip freeze` was not preserved at Stage-E execution time.

## General Paper-2 environment

`environment.yml` remains the general project environment specification. Notebook kernel metadata should not be used as proof of the interpreter that actually generated a result; machine-readable runtime/output manifests take precedence where available.
