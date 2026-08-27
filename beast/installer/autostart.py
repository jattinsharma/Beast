"""Auto-start with Windows via Task Scheduler (schtasks.exe).

Why Task Scheduler over Startup folder / registry Run key:
- Survives UAC and works without user login interaction quirks.
- Easy enable/disable by a single /Change or /Delete call - the tray toggle
  is one subprocess call, no registry parsing.
- Standard, documented, visible in taskschd.msc for debugging.

Trigger choice (verified live on this machine 2026-08-23):
- ONLOGON / ONSTART triggers require administrator rights here ("Access is
  denied" for a standard user) - NOT usable.
- DAILY trigger at 00:00 with the task DISABLED-by-default does not work as
  auto-start either. The working standard-user pattern is: create the task
  with a DAILY schedule at boot time and rely on Windows' "Run task as soon
  as possible after a scheduled start is missed" behavior... which schtasks
  cannot set. So instead we use the DAILY trigger purely as a carrier for
  the task definition and ALSO place a .lnk in the user's Startup folder -
  Startup folder needs no admin rights and fires at logon reliably.
  The scheduled task exists so taskschd.msc shows it and /Run can launch it
  on demand; the Startup shortcut is what actually starts Beast at login.

The task runs pythonw.exe (no console window) with app/main.py.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("beast.autostart")

TASK_NAME = "BeastAssistant"


def _venv_pythonw() -> str:
    """Path to pythonw.exe in the project venv (falls back to system)."""
    venv_pw = Path(__file__).parent.parent / "venv" / "Scripts" / "pythonw.exe"
    if venv_pw.exists():
        return str(venv_pw)
    cand = Path(sys.executable).with_name("pythonw.exe")
    if cand.exists():
        return str(cand)
    return "pythonw.exe"


def _main_py() -> str:
    return str(Path(__file__).parent.parent / "app" / "main.py")


def _startup_shortcut() -> Path:
    """Path of the .lnk in the user's Startup folder."""
    startup = (Path(os.environ["APPDATA"])
               / "Microsoft" / "Windows" / "Start Menu" / "Programs"
               / "Startup")
    return startup / "BeastAssistant.lnk"


def _make_shortcut(target: str, args: str, workdir: str, dest: Path):
    """Create a .lnk via COM (no admin needed)."""
    import ctypes
    # Use PowerShell to build the shortcut - reliable, no extra deps.
    ps = (
        f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
        f"'{dest}');$s.TargetPath='{target}';$s.Arguments='{args}';"
        f"$s.WorkingDirectory='{workdir}';$s.Save()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, text=True, timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.returncode == 0


def _run(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def task_exists() -> bool:
    code, _ = _run(["schtasks", "/Query", "/TN", TASK_NAME])
    return code == 0


def is_enabled() -> bool:
    """True if BOTH the scheduled task exists AND the Startup shortcut
    exists (the shortcut is what actually launches Beast at logon)."""
    return task_exists() and _startup_shortcut().exists()


def enable() -> bool:
    """Create the scheduled task + Startup shortcut. Returns True on success."""
    pythonw = _venv_pythonw()
    main_py = _main_py()
    workdir = str(Path(main_py).parent)

    # 1. Scheduled task (visible in taskschd.msc, enables run_now testing).
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{pythonw}" "{main_py}"',
        "/SC", "DAILY", "/ST", "00:00",   # carrier only; see module docstring
        "/RL", "LIMITED",
        "/F",
    ]
    code, out = _run(cmd)
    if code != 0:
        logger.error("Failed to create scheduled task (%d): %s", code, out)
        return False

    # Disable the daily trigger so it never actually fires on schedule;
    # we only want the task as an on-demand launcher + visibility.
    _run(["schtasks", "/Change", "/TN", TASK_NAME, "/DISABLE"])

    # 2. Startup-folder shortcut - this is what starts Beast at logon.
    lnk = _startup_shortcut()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    ok = _make_shortcut(pythonw, f'"{main_py}"', workdir, lnk)
    if not ok:
        logger.error("Failed to create Startup shortcut: %s", lnk)
        return False

    logger.info("Auto-start ENABLED: task=%s, shortcut=%s", TASK_NAME, lnk)
    return True


def disable() -> bool:
    """Remove both the task and the Startup shortcut."""
    ok = True
    if task_exists():
        code, out = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
        if code != 0:
            logger.error("Failed to delete task (%d): %s", code, out)
            ok = False
        else:
            logger.info("Scheduled task deleted: %s", TASK_NAME)
    lnk = _startup_shortcut()
    if lnk.exists():
        try:
            lnk.unlink()
            logger.info("Startup shortcut removed: %s", lnk)
        except OSError as e:
            logger.error("Failed to remove shortcut: %s", e)
            ok = False
    return ok


def toggle() -> bool:
    """Flip auto-start. Returns the new state (True = enabled)."""
    if is_enabled():
        disable()
        return False
    enable()
    return True


def run_now() -> bool:
    """Launch Beast via the exact mechanism Windows uses (the shortcut
    target), detached from any console - closest reliable equivalent to a
    real reboot test."""
    pythonw = _venv_pythonw()
    main_py = _main_py()
    try:
        subprocess.Popen(
            [pythonw, main_py],
            cwd=str(Path(main_py).parent),
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
        logger.info("Launched Beast silently via pythonw (detached)")
        return True
    except Exception as e:
        logger.error("Silent launch failed: %s", e)
        return False
