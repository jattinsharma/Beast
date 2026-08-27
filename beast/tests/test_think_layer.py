"""Live test: THINK layer - LLM decision quality on real screen state.

Tests decision quality BEFORE wiring to ACT (per milestone plan):
1. Simple goal on Notepad state -> should pick open_app or click/type sensibly
2. Goal already achieved -> task_complete
3. Ambiguous goal -> ask_user
4. Grammar forces valid JSON (no parse failures)
"""

import json
import sys
import time

sys.path.insert(0, r"C:\Beast\beast")

from agent.brain import AgentBrain
from agent.screen_state import ScreenState


def main():
    print("=== THINK LAYER LIVE TEST ===\n")
    brain = AgentBrain()
    see = ScreenState()

    # Make sure Notepad is open for a realistic state.
    import subprocess
    subprocess.Popen(["notepad"])
    time.sleep(2.5)

    state = see.summarize()
    print("--- SCREEN STATE GIVEN TO LLM ---")
    print(state[:800])
    print("---------------------------------\n")

    results = []

    # Test 1: simple goal, fresh Notepad -> expect open_app(notepad) or
    # type_text(hello) since notepad is already open.
    print("[TEST 1] goal='open Notepad and type hello'")
    d = brain.decide("open Notepad and type hello", state, [])
    print(f"    -> {json.dumps(d)}")
    ok = d["action"] in ("open_app", "type_text", "click")
    results.append(("simple goal", ok))
    print(f"    {'PASS' if ok else 'FAIL'}\n")

    # Test 2: after typing happened, goal met -> task_complete expected.
    history = [
        {"action": '{"action": "open_app", "target": "notepad"}',
         "result": "opened notepad [verify=pass]"},
        {"action": '{"action": "type_text", "text": "hello"}',
         "result": "typed 5 chars [verify=pass]"},
    ]
    print("[TEST 2] goal='open Notepad and type hello' (already done)")
    d = brain.decide("open Notepad and type hello", state, history)
    print(f"    -> {json.dumps(d)}")
    ok = d["action"] == "task_complete"
    results.append(("task_complete when done", ok))
    print(f"    {'PASS' if ok else 'FAIL'}\n")

    # Test 3: ambiguous goal -> ask_user expected.
    print("[TEST 3] goal='do the thing with it'")
    d = brain.decide("do the thing with it", state, [])
    print(f"    -> {json.dumps(d)}")
    ok = d["action"] == "ask_user"
    results.append(("ask_user on ambiguity", ok))
    print(f"    {'PASS' if ok else 'FAIL'}\n")

    # Test 4: grammar validity across several runs of a normal goal.
    print("[TEST 4] JSON validity over 3 runs")
    valid = 0
    for i in range(3):
        d = brain.decide("click the File menu", state, [])
        if d["action"] != "ask_user" or "invalid" not in d.get("reason", ""):
            valid += 1
        print(f"    run {i+1}: {json.dumps(d)}")
    ok = valid == 3
    results.append((f"grammar validity ({valid}/3)", ok))
    print(f"    {'PASS' if ok else 'FAIL'}\n")

    print("=== SUMMARY ===")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'} - {name}")
    print(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
