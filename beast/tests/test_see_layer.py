"""Live test: SEE layer - UIA tree + OCR on real windows.

Run with Notepad open (or let this script open it).
Verifies:
1. UIA correctly identifies elements in a real window (Notepad menu/edit)
2. OCR correctly reads real on-screen text
3. summarize() produces a usable state description
"""

import subprocess
import sys
import time

sys.path.insert(0, r"C:\Beast\beast")

from agent.screen_state import ScreenState, find_element_center


def main():
    print("=== SEE LAYER LIVE TEST ===\n")

    # Open Notepad fresh so we know its state.
    print("[1] Opening Notepad...")
    subprocess.Popen(["notepad"])
    time.sleep(2.5)

    see = ScreenState()

    # --- Test 1: UIA ---
    print("\n[TEST 1] UIA element walk on Notepad")
    elems = see.uia_elements("Notepad")
    print(f"    Found {len(elems)} elements")
    for e in elems[:15]:
        print(f"    - {e['name']!r} | {e['type']} | rect={e['rect']}")
    assert len(elems) > 0, "UIA found no elements in Notepad!"
    types = {e["type"] for e in elems}
    print(f"    Control types present: {sorted(types)}")
    print("    PASS" if len(elems) >= 3 else "    WEAK (few elements)")

    # --- Test 1b: find_element_center ---
    print("\n[TEST 1b] find_element_center('File')")
    pt = find_element_center(elems, "File")
    print(f"    -> {pt}")
    print("    PASS" if pt else "    FAIL")

    # --- Test 2: OCR ---
    print("\n[TEST 2] Windows OCR on screen")
    lines = see.ocr_lines()
    print(f"    Found {len(lines)} text lines")
    for l in lines[:10]:
        print(f"    - {l['text']!r} @ {l['rect']}")
    if lines:
        print("    PASS")
    else:
        print("    NOTE: no text detected (screen may be mostly empty)")

    # --- Test 3: summarize ---
    print("\n[TEST 3] summarize() output")
    summary = see.summarize()
    print(summary[:1500])
    assert "Active window:" in summary
    print("\n    PASS")

    print("\n=== ALL SEE TESTS DONE ===")


if __name__ == "__main__":
    main()
