# Computer Use Clanker — GTX 1070

Local computer-use agent: **Qwen2.5-VL-7B (Q4_K_M)** via Ollama, fully on the
GTX 1070 (8GB VRAM, CUDA). Takes screenshots, decides the next GUI action,
executes it with xdotool, repeats until done.

## Quick start

```bash
# 1. make sure ollama is running (skip if already up)
curl -s http://localhost:11434/api/version || nohup ollama serve > /tmp/opencode/ollama.log 2>&1 &

# 2. describe the current screen (sanity check)
python3 /home/gs/cu-agent/cu_agent.py --describe

# 3. run a task
python3 /home/gs/cu-agent/cu_agent.py "open firefox and go to example.com"

# with a hint (helps a lot on cluttered desktops)
python3 /home/gs/cu-agent/cu_agent.py --hint "GNOME desktop: press Super, type app name, Enter." \
    "open a terminal and type ls"
```

## Options

| flag | meaning |
|---|---|
| `--describe` | one-shot: describe the current screen |
| `--task` (positional) | the task to perform |
| `--model` | default `qwen2.5vl:7b` |
| `--max-steps` | default 10 |
| `--hint` | extra environment hint appended to the task |
| `--dry-run` | show the actions it would take without touching mouse/keyboard |
| `--think` | show the model's reasoning trace before each action (adds ~1-2s/step) |
| `--no-auto-enter` | disable the safety net that auto-presses Enter when the model finishes with a command typed but not executed in a terminal |

## Actions the model can take

`click`, `double_click`, `right_click`, `type`, `key` (e.g. `Return`, `ctrl+a`),
`scroll`, `shell` (runs a bash command, 30s timeout — how it launches apps),
`wait`, `done`.

> **Note:** the `shell` action gives the model real command execution on your
> box. It's what makes app-launching reliable, but keep that in mind for
> sensitive machines.

## Notes

- Model: `qwen2.5vl:7b` (Q4_K_M, ~6GB) — loads 100% on the 1070's VRAM.
- Screenshots are downscaled to 1280x720 before sending (coordinates are
  scaled back up) — this is the main speed lever: ~6-8s/step vs ~25s at 1080p.
- The `shell` action launches apps reliably; new windows are auto-focused.
- The agent moves your real mouse and types — keep an eye on it.
- Tips for reliability:
  - Give specific tasks ("open xterm", not "open a terminal").
  - Use `--hint` to tell it how to do things on your desktop.
  - Fewer open windows = easier for the model.
- Swap models: `ollama pull qwen2.5vl:3b` for a faster, weaker model, or
  `qwen2.5vl:7b-q8_0` (9.4GB, won't fully fit in 8GB VRAM — avoid).