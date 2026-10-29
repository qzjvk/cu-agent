# Computer Use Clanker — GTX 1070

Local computer-use agent: **[LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)**
(1.92B vision-language model, Hugging Face, transformers, **fp16**), running
**only on the GTX 1070** (8GB VRAM, CUDA, sm_61). Takes screenshots, decides the
next GUI action, executes it with xdotool, repeats until done.

Model footprint: ~3.7GB VRAM (fp16) — the whole model fits on the 1070, no
offloading, no Ollama, no server. Typical step time: ~4-5s at 1280x720.

## Setup (one time)

```bash
cd /home/gs/Applications/cu-agent
python3 -m venv --system-site-packages .venv
.venv/bin/pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install "transformers>=5.21" accelerate
```

Notes:

- **torch must be the cu126 build** (`2.14.1+cu126`). Newer `cu13x` wheels drop
  Pascal (sm_61) kernels and fail with "no kernel image is available".
- The model (3.8GB) downloads on first run into `~/.cache/huggingface`.
- fp16 (not bf16) is used on purpose: Pascal's fast path is fp16.

## Quick start

```bash
# describe the current screen (sanity check)
.venv/bin/python cu_agent.py --describe

# run a task
.venv/bin/python cu_agent.py "open firefox and go to example.com"

# with a hint (helps a lot on cluttered desktops)
.venv/bin/python cu_agent.py --hint "XFCE: click the app-menu button top-left, type app name." \
    "open a terminal and type ls"
```

## Options

| flag | meaning |
|---|---|
| `--describe` | one-shot: describe the current screen |
| `task` (positional) | the task to perform |
| `--model` | default `LiquidAI/LFM3-VL-2B` |
| `--gpu` | CUDA index to force (default: auto-detect the GTX 1070 by name). Note: CUDA index order is NOT nvidia-smi order — the auto-detect handles that. |
| `--max-steps` | default 10 |
| `--max-tokens` | cap on model output per step (default 256, 1024 with `--think`) |
| `--hint` | extra environment hint appended to the task |
| `--dry-run` | show the actions it would take without touching mouse/keyboard |
| `--think` | let the model reason before each action (enable_thinking in the chat template — much slower on the 1070, use sparingly) |
| `--debug` | save every step's screenshot to `./debug_shots/` (git-ignored) |
| `--no-auto-enter` | disable the safety net that auto-presses Enter when the model finishes with a command typed but not executed in a terminal |

## Actions the model can take

`click`, `double_click`, `right_click`, `type`, `key` (e.g. `Return`, `ctrl+a`),
`scroll`, `shell` (runs a bash command, 30s timeout — how it launches apps),
`wait`, `done`.

> **Note:** the `shell` action gives the model real command execution on your
> box. It's what makes app-launching reliable, but keep that in mind for
> sensitive machines.

## Notes

- The agent pins CUDA to the GTX 1070 by device name before importing torch, so
  the busy RTX 3060s are never touched. Override with `--gpu N` if you ever
  want to run it on another card (any CUDA GPU with ≥6GB free works, but the
  1070 is the target).
- Screenshots are downscaled to 1280x720 before inference (coordinates are
  scaled back up) — ~910 image tokens, the main speed lever.
- Generation is greedy (`do_sample=False`) for deterministic actions, and the
  chat template is called with `enable_thinking=False` so the model answers
  with the bare JSON object instead of reasoning at length.
- LFM3-VL-2B scores 83.1 on GUI understanding — strong at finding and clicking
  UI elements, which is exactly what this agent needs.
- Tips for reliability:
  - Give specific tasks ("open xterm", not "open a terminal").
  - Use `--hint` to tell it how to do things on your desktop.
  - Fewer open windows = easier for the model.