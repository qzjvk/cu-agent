#!/usr/bin/env python3
"""
cu_agent.py — minimal computer-use agent loop for a local Qwen2.5-VL model via Ollama.

Flow: screenshot -> model decides next action (JSON) -> execute with xdotool -> repeat.

Usage:
  python3 cu_agent.py "open the terminal and type ls"     # run task loop
  python3 cu_agent.py --describe                          # one-shot: describe the screen
  python3 cu_agent.py --max-steps 15 "task"               # limit steps (default 10)
  python3 cu_agent.py --model qwen2.5vl:7b "task"         # pick model
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
SCREENSHOT = "/tmp/opencode/cu_shot.png"
SHOT_W, SHOT_H = 1280, 720  # downscale target (big speedup vs 1080p)

SYSTEM_PROMPT = """You are a computer-use agent controlling a Linux desktop.
You are given a screenshot of the current screen. Decide the single next action.

Respond with ONLY a single JSON object, no markdown, no explanation, matching one of:
{"action": "click", "x": <int>, "y": <int>}
{"action": "double_click", "x": <int>, "y": <int>}
{"action": "right_click", "x": <int>, "y": <int>}
{"action": "type", "text": "<string>"}
{"action": "key", "combo": "<e.g. 'Return', 'ctrl+a', 'alt+Tab'>"}
{"action": "scroll", "direction": "up"|"down", "amount": <int>}
{"action": "shell", "cmd": "<bash command>"}
{"action": "wait", "seconds": <int>}
{"action": "done", "answer": "<final result or 'ok'>"}

Rules:
- This is an XFCE desktop. There is NO Super-key overview. The app menu is the button at the top-left of the panel.
- To LAUNCH an app, prefer the shell action, e.g. {"action": "shell", "cmd": "xterm &"}. GUI apps must end with &.
- If the target window is ALREADY open on screen, do NOT launch it again — click into it instead.
- If the task names a SPECIFIC app (e.g. "xterm"), you MUST launch that exact app with the shell action. Do NOT substitute a different, similar-looking window that happens to be open.
- Use clicks only to interact with windows that are already open.
- Coordinates are pixel positions in the screenshot (top-left origin).
- Click the center of the target element.
- The Enter key is spelled "Return" (not "Enter"). Other names: "Escape", "Tab", "BackSpace".
- Use "key" for Enter/Escape/shortcuts, "type" for plain text.
- If the task involves typing a command into a terminal, you MUST press Enter (Return) to execute it before emitting done. A command that is typed but not yet executed is NOT done.
- Before typing, make sure the target window is focused: click inside it first if unsure.
- Shell commands have a 30s timeout.
- CRITICAL: Once the task's goal is achieved, you MUST immediately emit {"action": "done", "answer": "..."}. 
  If you see the RESULT of your previous action on screen (e.g. the command you typed has run and its output is visible), the task is COMPLETE — emit done.
- NEVER repeat an action you have already performed. If you are about to repeat yourself, the task is almost certainly done — emit done instead.
- If the task is impossible, use "done" with an explanation.
- Never output anything except the JSON object."""


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr.strip():
        print(f"        !! command failed: {cmd}\n           {r.stderr.strip()}")
    return r


KEY_ALIASES = {
    "enter": "Return", "return": "Return",
    "esc": "Escape", "escape": "Escape",
    "backspace": "BackSpace", "delete": "Delete", "del": "Delete",
    "space": "space", "tab": "Tab",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End",
    "pageup": "Page_Up", "pagedown": "Page_Down",
    "insert": "Insert", "capslock": "Caps_Lock",
}


def norm_combo(combo):
    """Normalize a key combo like 'ctrl+Enter' to xdotool keysyms 'ctrl+Return'."""
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    if not parts:
        return "Return"
    mods, key = parts[:-1], parts[-1]
    key = KEY_ALIASES.get(key.lower(), key)
    return "+".join(mods + [key])


def active_window():
    r = run("xdotool getactivewindow getwindowname 2>/dev/null")
    return r.stdout.strip() or "?"


TERMINAL_HINTS = ("terminal", "xterm", "konsole", "alacritty", "kitty", "tilix",
                  "lxterminal", "mate-terminal", "gnome-terminal", "xfce4-terminal",
                  "urxvt", "putty", "console", "bash", "zsh", "fish")

# WM_CLASS is far more reliable than the title: xterm titles itself with the cwd
# (e.g. "~/cu-agent"), which matches nothing in TERMINAL_HINTS.
TERMINAL_CLASSES = ("xterm", "gnome-terminal", "konsole", "alacritty", "kitty",
                    "tilix", "xfce4-terminal", "lxterminal", "mate-terminal",
                    "urxvt", "rxvt", "terminator", "qterminal", "putty", "tabby",
                    "termite", "cool-retro-term", "blackbox", "contour", "wezterm",
                    "guake", "yakuake", "foot", "st")


def is_terminal(name):
    n = (name or "").lower()
    return any(h in n for h in TERMINAL_HINTS)


def active_window_id():
    r = run("xdotool getactivewindow 2>/dev/null")
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def window_name(wid):
    r = run(f"xdotool getwindowname {wid} 2>/dev/null")
    return r.stdout.strip()


def window_class(wid):
    r = run(f"xprop -id {wid} WM_CLASS 2>/dev/null")
    m = re.search(r"WM_CLASS\(STRING\) = (.*)", r.stdout)
    return m.group(1).lower() if m else ""


def is_terminal_window(wid):
    """True if this window is a terminal: WM_CLASS first, title as fallback."""
    if not wid:
        return False
    if any(c in window_class(wid) for c in TERMINAL_CLASSES):
        return True
    return is_terminal(window_name(wid))


def screenshot():
    """Capture the screen, downscale to SHOT_W x SHOT_H, return (b64, scale_x, scale_y)."""
    raw = "/tmp/opencode/cu_shot_raw.png"
    r = run(f"gnome-screenshot -f {raw} 2>/dev/null || import -window root {raw}")
    if r.returncode != 0 or not os.path.exists(raw):
        raise RuntimeError(f"screenshot failed: {r.stderr}")
    r = run(f"convert {raw} -resize {SHOT_W}x{SHOT_H} {SCREENSHOT}")
    if r.returncode != 0:
        raise RuntimeError(f"resize failed: {r.stderr}")
    # actual screen size
    r = run("xdotool getdisplaygeometry")
    sw, sh = (int(v) for v in r.stdout.split())
    with open(SCREENSHOT, "rb") as f:
        return base64.b64encode(f.read()).decode(), sw / SHOT_W, sh / SHOT_H


def ask(model, image_b64, user_text, history=None, system=SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text, "images": [image_b64]})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def parse_action(text):
    """Extract the first valid JSON object from the model output."""
    actions = parse_actions(text)
    return actions[0] if actions else None


def parse_actions(text):
    """Extract ALL valid JSON objects from the model output (it may plan several)."""
    actions = []
    start = text.find("{")
    while start != -1:
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict) and "action" in obj:
                actions.append(obj)
                start = text.find("{", start + end)
                continue
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return actions


def window_ids():
    r = run("wmctrl -l 2>/dev/null")
    return {line.split()[0] for line in r.stdout.splitlines() if line.strip()}


def execute(action, dry_run=False, scale=(1.0, 1.0)):
    a = action.get("action")
    if a in ("click", "double_click", "right_click"):
        x = int(round(int(action["x"]) * scale[0]))
        y = int(round(int(action["y"]) * scale[1]))
        btn = {"click": "1", "double_click": "1", "right_click": "3"}[a]
        clicks = "click --repeat 2" if a == "double_click" else "click"
        if not dry_run:
            run(f"xdotool mousemove {x} {y} {clicks} --clearmodifiers button{btn}")
        return f"{a} at ({x},{y})"
    if a == "type":
        text = action.get("text", "")
        trailing_newlines = len(text) - len(text.rstrip("\n"))
        text = text.rstrip("\n")
        safe = text.replace("'", "'\\''")
        focus = active_window()
        if not dry_run:
            run(f"xdotool type --clearmodifiers --delay 30 '{safe}'")
            for _ in range(trailing_newlines):
                run("xdotool key --clearmodifiers Return")
        return f"type {text[:40]!r} (+{trailing_newlines} Enter) into {focus!r}"
    if a == "key":
        combo = norm_combo(action.get("combo", "Return"))
        focus = active_window()
        if not dry_run:
            run(f"xdotool key --clearmodifiers {combo}")
        return f"key {combo} (focus: {focus!r})"
    if a == "shell":
        cmd = action.get("cmd", "").strip()
        if not cmd:
            return "shell: empty command"
        if dry_run:
            return f"shell (dry-run): {cmd!r}"
        out_file = "/tmp/opencode/cu_shell_out"
        is_bg = "&" in cmd
        p = subprocess.Popen(["/bin/bash", "-c", f"{cmd} > {out_file} 2>&1"],
                             start_new_session=True)
        if is_bg:
            before = window_ids()
            time.sleep(2.5)
            new = window_ids() - before
            focus_msg = ""
            if new:
                wid = sorted(new)[0]
                run(f"xdotool windowactivate --sync {wid} 2>/dev/null")
                focus_msg = f", focused new window {wid}"
            return f"shell: {cmd!r} launched{focus_msg}"
        try:
            p.wait(timeout=30)
            out = ""
            try:
                out = open(out_file).read().strip().replace("\n", " ")[:150]
            except OSError:
                pass
            return f"shell: {cmd!r} rc={p.returncode} {out}"
        except subprocess.TimeoutExpired:
            p.kill()
            return f"shell: {cmd!r} TIMED OUT after 30s"
    if a == "scroll":
        direction = "down" if action.get("direction", "down") == "down" else "up"
        amount = int(action.get("amount", 3))
        btn = "5" if direction == "down" else "4"
        if not dry_run:
            run(f"xdotool click --repeat {amount} {btn}")
        return f"scroll {direction} x{amount}"
    if a == "wait":
        if not dry_run:
            time.sleep(float(action.get("seconds", 1)))
        return f"wait {action.get('seconds', 1)}s"
    if a == "done":
        return None
    return f"unknown action {a}"


def action_signature(action):
    """A hashable identity for an action, used to detect repetition."""
    a = action.get("action")
    if a in ("click", "double_click", "right_click"):
        return (a, action.get("x"), action.get("y"))
    if a == "type":
        return (a, action.get("text"))
    if a == "key":
        return (a, action.get("combo"))
    if a == "shell":
        return (a, action.get("cmd"))
    if a == "scroll":
        return (a, action.get("direction"), action.get("amount"))
    return (a,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default=None)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--hint", default="", help="extra environment hint appended to the task")
    ap.add_argument("--dry-run", action="store_true", help="show actions without executing them")
    ap.add_argument("--think", action="store_true",
                    help="ask the model for a short reasoning line before each action (slower)")
    ap.add_argument("--no-auto-enter", action="store_true",
                    help="do not auto-press Enter when the model finishes with a command typed but not executed in a terminal")
    args = ap.parse_args()

    if args.describe:
        img, _, _ = screenshot()
        out = ask(args.model, img, "Describe what is on this screen in detail.",
                  system="You are a helpful assistant describing screenshots.")
        print(out)
        return

    if not args.task:
        ap.print_help()
        sys.exit(1)

    task = args.task
    if args.hint:
        task = f"{task}\n\nEnvironment hint: {args.hint}"

    print(f"Task: {task}")
    print(f"Model: {args.model} @ {OLLAMA}")
    print("-" * 60)

    history = []
    launch_note = ""
    repeat_note = ""
    executed = {}  # action signature -> times seen
    pending_enter = None  # id of the terminal window where a command was typed but not yet executed
    system = SYSTEM_PROMPT
    if args.think:
        system = SYSTEM_PROMPT.replace(
            "Never output anything except the JSON object.",
            "First write 1-3 short lines of reasoning starting with 'Thought: ' "
            "(what you see, what you will do and why). Then output the JSON object on its own line.")
    for step in range(1, args.max_steps + 1):
        img, sx, sy = screenshot()
        user_text = task
        if launch_note:
            user_text = task + "\n\nIMPORTANT STATE: " + launch_note
        if repeat_note:
            user_text = task + "\n\nWARNING: " + repeat_note
        launch_note = ""  # consumed this step; re-set below if a launch happens
        repeat_note = ""
        t0 = time.time()
        raw = ask(args.model, img, user_text, history, system=system)
        dt = time.time() - t0
        actions = parse_actions(raw)
        if not actions:
            print(f"[step {step}] could not parse action, raw output:\n{raw}")
            break
        if args.think:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            thought = raw[:m.start()].strip() if m else ""
            if thought:
                print(f"[step {step}] ({dt:.1f}s) thought: {thought}")
            else:
                print(f"[step {step}] ({dt:.1f}s)")
        for action in actions:
            a = action.get("action")
            print(f"          {json.dumps(action)}")
            if a == "done":
                if pending_enter and not args.dry_run and not args.no_auto_enter:
                    # The model typed a command into a terminal but never pressed Enter.
                    # Focus that terminal and press Enter so the command actually runs,
                    # even if focus drifted elsewhere in the meantime.
                    r = run(f"xdotool windowactivate --sync {pending_enter} 2>/dev/null")
                    if r.returncode == 0:
                        print(f"        -> command typed but not executed in {window_name(pending_enter)!r} — auto-pressing Enter")
                        run("xdotool key --clearmodifiers Return")
                        time.sleep(1.0)
                    else:
                        print(f"        -> auto-Enter skipped: terminal window {pending_enter} is gone")
                print(f"DONE: {action.get('answer')}")
                return
            sig = action_signature(action)
            executed[sig] = executed.get(sig, 0) + 1
            if executed[sig] >= 3:
                print(f"        !! action repeated {executed[sig]}x — stopping (task complete or stuck)")
                return
            if executed[sig] >= 2:
                print(f"        !! repeating an action already taken — skipping, telling model to stop")
                repeat_note = ("You are repeating an action you already performed. The task is very likely "
                               "COMPLETE. Do NOT repeat it. Emit {\"action\": \"done\", \"answer\": \"...\"} now.")
                history.append({"role": "assistant", "content": json.dumps(action)})
                continue
            result = execute(action, dry_run=args.dry_run, scale=(sx, sy))
            if result:
                print(f"        -> {result}")
            if not args.dry_run:
                if a == "type":
                    wid = active_window_id()
                    if is_terminal_window(wid):
                        pending_enter = wid
                elif a == "key":
                    pending_enter = None
            history.append({"role": "assistant", "content": json.dumps(action)})
            if a == "shell" and result and "launched" in result:
                launch_note = ("You just launched an app and its window is now open and focused. "
                               "The launch/open step is COMPLETE. Do NOT launch it again. "
                               "Proceed to the next step of the task (e.g. type into the focused window).")
            time.sleep(0.6)  # let the UI settle between actions
        time.sleep(0.5)

    print("Max steps reached.")


if __name__ == "__main__":
    main()