# Contributing

This is a research-oriented repository for Apple Silicon/CoreML TTS experiments. Contributions are most useful when they improve reproducibility, clarify benchmark results, or reduce friction for running the conversion and parity checks on another machine.

## Good Contributions

- Reproduction notes for different Apple Silicon hardware.
- Small fixes to setup, scripts, or documentation.
- Benchmark results with hardware, macOS version, model path assumptions, and command output.
- Narrow bug fixes with the smallest relevant smoke check.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make smoke
```

Model-dependent tests require local Fish S2 Pro and ANEMLL assets. Use `FISH_MODEL_DIR`, `ANEMLL_REPO`, `FISH_SPEECH_REPO`, and `ANEMLL_CKPT_DIR` instead of committing machine-specific paths.

## Pull Requests

Keep PRs focused. Include the command you ran, the hardware used for benchmark changes, and whether the result is measured or estimated.
