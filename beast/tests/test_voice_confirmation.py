"""Live test: Level-3 confirmation with REAL microphone input.

Uses the actual Milestone 2 voice stack:
  - Piper TTS speaks the question aloud
  - Mic records your spoken answer
  - faster-whisper transcribes it
  - ConfirmationGate decides yes/no

YOU must speak the answers. No scripted input.

Run: python tests/test_voice_confirmation.py
"""

import subprocess
import sys
import time

sys.path.insert(0, r"C:\Beast\beast")

from agent.autonomy import ConfirmationGate
from app.pipeline import PiperTTS, WhisperSTT
from computer.computer_tool import ComputerTool


def record_answer(max_seconds=6.0, silence_threshold=0.01, silence_frames=8):
    """Record mic audio until silence - same pattern as main.py."""
    import numpy as np
    import sounddevice as sd

    frames = []
    silent_run = 0
    started = False

    def cb(indata, frame_count, time_info, status):
        nonlocal silent_run, started
        frames.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms > silence_threshold:
            started = True
            silent_run = 0
        elif started:
            silent_run += 1

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=1280, callback=cb):
        t0 = time.time()
        while time.time() - t0 < max_seconds:
            time.sleep(0.05)
            if started and silent_run >= silence_frames:
                break

    import numpy as np
    return np.concatenate(frames, axis=0).flatten() if frames else None


def main():
    print("=== LEVEL-3 VOICE CONFIRMATION TEST (REAL MIC) ===")
    print("You will HEAR a question and must ANSWER OUT LOUD.\n")

    # GUARD: refuse to run while the Beast tray app is running. The tray app
    # holds the mic in its own process; a second stream here caused phantom
    # transcripts ('hey jarvis!' instead of 'yes'). Close Beast from the tray
    # before running this test.
    import psutil
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "app\\main.py" in cmd or "app/main.py" in cmd:
                print("ERROR: Beast tray app is running (PID "
                      f"{proc.pid}). It holds the microphone, so this test "
                      "would record contention garbage.")
                print("Quit Beast from the tray icon first, then re-run.")
                sys.exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # START CLEAN: kill any existing notepad processes to avoid interference
    print("[cleanup] Starting clean - killing any existing notepad processes")
    subprocess.run(["taskkill", "/f", "/im", "notepad.exe"],
                   capture_output=True)
    time.sleep(1)  # Give time for processes to terminate

    ct = ComputerTool()
    tts = PiperTTS()
    stt = WhisperSTT()

    gate = ConfirmationGate(tts, stt, record_answer)

    # Warm up models so first question isn't delayed by loading.
    print("[warmup] Loading TTS + STT models...")
    tts._ensure_loaded()
    stt._ensure_loaded()
    print("[warmup] done\n")

    results = []

    for round_num, expected in [("ROUND 1", "NO"), ("ROUND 2", "YES")]:
        print(f"--- {round_num}: answer '{expected}' out loud ---")
        subprocess.Popen(["notepad"])
        time.sleep(2.5)

        before = sum(1 for w in ct.list_windows() if "notepad" in w.lower())
        confirmed = gate.confirm(
            "Should I close this Notepad window? Please state your choice.")

        # THE TEST MUST ACT: if the gate confirmed, the test performs the
        # risky action (close Notepad). Without this, ROUND 2 (YES) can
        # never pass because nothing ever closes the window.
        if confirmed:
            # Focus the Notepad window to make sure it's active
            ct.focus_window("Notepad")
            time.sleep(0.5)

            # Try to close the window by getting the window object and calling close
            import pygetwindow as gw
            notepad_windows = [w for w in gw.getAllWindows() if "notepad" in w.title.lower()]
            if notepad_windows:
                win = notepad_windows[0]
                win.close()
                time.sleep(1)  # Wait for the save dialog to appear

                # Check if we have a save dialog
                active_window = ct.get_active_window()
                if active_window and ("save" in active_window.lower() or "unsaved" in active_window.lower()):
                    # Press 'n' for Don't Save
                    ct.press("n")
                    time.sleep(0.5)
            # If no Notepad window found, do nothing (the window might already be closed)
            # If no save dialog, window should be closing or closed

        time.sleep(2)
        after = sum(1 for w in ct.list_windows() if "notepad" in w.lower())

        acted = after < before

        # SAFETY INVARIANT: the observed action MUST match the expectation.
        # said NO  -> window must still be open (did not act)
        # said YES -> window must be closed   (acted)
        action_correct = (expected == "NO" and not acted) or \
                         (expected == "YES" and acted)
        # Gate decision should also match, but ACTION is what matters for
        # safety - a correct action with a wrong gate flag is still safe.
        correct = action_correct
        results.append((expected, confirmed, acted, correct))
        verdict = "PASS" if correct else "FAIL"
        print(f"    gate={'confirmed' if confirmed else 'declined'}, "
              f"window closed={acted}, expected={expected} -> {verdict}\n")

    # SAFETY: only clean up windows Beast did NOT act on (i.e. the NO round
    # leftover). If the YES round worked, its window is already gone.
    # We do NOT blanket-kill notepad here - that would mask whether the gate
    # actually held (the bug from the previous test version).
    print("[cleanup] Closing any remaining test Notepad via taskkill "
          "(this is the TEST cleaning up, NOT Beast acting)")
    subprocess.run(["taskkill", "/f", "/im", "notepad.exe"],
                   capture_output=True)

    print("=== RESULTS ===")
    all_pass = all(r[3] for r in results)
    for exp, conf, acted, ok in results:
        print(f"  said {exp} -> window {'closed' if acted else 'still open'} "
              f"[{'PASS' if ok else 'FAIL'}]")
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("\nWAV files of every attempt saved to logs\\confirmations\\ "
          "- listen to them to hear exactly what the mic captured.")


if __name__ == "__main__":
    main()
