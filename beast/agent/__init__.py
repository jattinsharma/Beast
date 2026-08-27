"""BEAST Agent Loop (Milestone 4): SEE -> THINK -> ACT -> VERIFY.

Modules:
- screen_state: SEE layer - UIA tree + Windows OCR (escalation ladder)
- brain:        THINK layer - Qwen3 grammar-constrained JSON action output
- executor:     ACT layer - one structured action per iteration via ComputerTool
- verifier:     VERIFY layer - expected-effect checks after each action
- autonomy:     Risk tiers L1-L4 with Level-3 voice confirmation
- loop:         The full agent loop, estop-aware and fully logged

Memory integration:
- The AgentLoop accepts an optional memory_manager (MemoryManager)
- Memory context is injected into the LLM prompt each iteration
- Memory actions (remember/recall/forget) are available as agent actions
"""
