"""Live end-to-end test: full SEE->THINK->ACT->VERIFY loop.

Test 1: "open Notepad and type hello" - the milestone's primary verification
        case. Must complete via escalation ladder (known tool / UIA), with
        VERIFY confirming the text actually appeared.
Test 2: estop mid-loop - must halt the agent loop, not just one ComputerTool
        call.

Run this and watch the screen: Notepad should open and 'hello' appear.
"""

import subprocess
import sys
import threading
import time

sys.path.insert(0, r"C:\Beast\beast")

from agent.loop import AgentLoop
from computer.computer_tool import ComputerTool
from computer.safety import estop, reset_estop


def main():
    print("=== AGENT LOOP END-TO-END LIVE TEST ===\n")

    ct = ComputerTool()
    # No TTS/STT for this scripted test - gate disabled (L3 actions will be
    # tested separately). Loop still runs L1 actions and verification.
    loop = AgentLoop(ct)

    # Close any existing notepad first for a clean start.
    subprocess.run(["taskkill", "/f", "/im", "notepad.exe"],
                   capture_output=True)
    time.sleep(1)

    # ---- Test 1: open Notepad and type hello ----
    print("[TEST 1] goal='open Notepad and type hello'")
    t0 = time.time()
    result = loop.run("open Notepad and type hello")
    dt = time.time() - t0
    print(f"\n    Loop result: {result!r}")
    print(f"    Elapsed: {dt:.1f}s, iterations: {len(loop.history)}")

    # Independent confirmation: OCR the screen and look for 'hello'.
    time.sleep(1.5)
    lines = loop.see.ocr_lines()
    all_text = " ".join(l["text"] for l in lines).lower()
    found = "hello" in all_text
    print(f"    OCR check: 'hello' visible on screen -> {found}")
    for l in lines[:8]:
        print(f"      ocr: {l['text']!r}")

    if found:
        print("    TEST 1 PASS\n")
    else:
        print("    TEST 1 FAIL - hello not visible\n")

    # ---- Test 2: estop mid-loop ----
    print("[TEST 2] estop fires mid-loop")
    reset_estop()

    def fire_estop():
        time.sleep(4)  # let the loop get into its first action
        estop()

    threading.Thread(target=fire_estop, daemon=True).start()
    result = loop.run("open Calculator and click buttons randomly")
    print(f"    Loop result: {result!r}")
    halted = "halted" in result.lower() or "emergency" in result.lower()
    print(f"    TEST 2 {'PASS' if halted else 'FAIL'}\n")
    reset_estop()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
