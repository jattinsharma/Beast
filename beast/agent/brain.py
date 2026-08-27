"""THINK layer: LLM decides the NEXT SINGLE ACTION as structured JSON.

Uses llama-cpp-python's GBNF grammar support to force syntactically valid
JSON matching our action schema, then validates semantically in Python.
On parse/validation failure: retry (max 2), then emit ask_user - never guess.

Action schema:
  {"action": "open_app"|"click"|"type_text"|"press"|"hotkey"|"scroll"
            |"focus_window"|"task_complete"|"ask_user",
   "target": "<element name / app name / window title>",   # click/focus/open
   "text": "<text to type>",                               # type_text
   "key": "<key name>",                                    # press
   "keys": ["ctrl","s"],                                   # hotkey
   "amount": <int>,                                        # scroll (+up/-down)
   "reason": "<one short sentence>",
   "message": "<what to say to the user>"}                 # ask_user/complete
"""

import json
import logging

logger = logging.getLogger("beast.agent")

VALID_ACTIONS = {
    "open_app", "click", "type_text", "press", "hotkey", "scroll",
    "focus_window", "task_complete", "ask_user",
    # Browser actions
    "browser_navigate", "browser_find_element", "browser_click",
    "browser_type_text", "browser_get_text", "browser_screenshot",
    "browser_wait_for",
    # Memory actions
    "remember", "recall", "forget", "list_memories",
}

# GBNF grammar: forces syntactically valid JSON that ALWAYS contains an
# "action" key first (prevents the empty-object degenerate case). Other
# keys are free-form; Python validates semantics afterwards.
ACTION_GRAMMAR = r"""
root ::= "{" ws "\"action\"" ws ":" ws action ws ("," ws pair)* ws "}"
action ::= "\"" ident "\""
ident ::= [a-z_]+
pair ::= string ":" ws value
value ::= object | array | string | number | ("true" | "false" | "null")
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
array ::= "[" ws (value ("," ws value)*)? ws "]"
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? (("e" | "E") "-"? [0-9]+)?
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
ws ::= [ \t\n]*
"""

SYSTEM_PROMPT = """You are Beast's action planner. You control a Windows PC one action at a
time. Given the user's goal, current screen state, and actions already
taken, output ONE JSON object choosing the next single action.

Available actions:
- {"action": "open_app", "target": "<app name>"} - launch an app
- {"action": "click", "target": "<element name from screen state>"}
- {"action": "type_text", "text": "<text to type>"}
- {"action": "press", "key": "enter"}
- {"action": "hotkey", "keys": ["ctrl", "s"]}
- {"action": "scroll", "amount": -3}
- {"action": "focus_window", "target": "<window title>"}
- {"action": "task_complete", "message": "<done summary>"}
- {"action": "ask_user", "message": "<your question>"}
- {"action": "browser_navigate", "url": "<url to navigate to>"} - navigate to a web page
- {"action": "browser_click", "target": "<element description>"} - click a web element
- {"action": "browser_type_text", "target": "<element description>", "text": "<text to type>"} - type into a web element
- {"action": "browser_get_text", "target": "<element description>"} - get text from a web element
- {"action": "browser_screenshot", "full_page": <true/false>} - take a screenshot of the page
- {"action": "browser_wait_for", "condition": "<load state>", "timeout": <milliseconds>} - wait for page load

=== CRITICAL: MEMORY ACTION CLASSIFICATION ===
MEMORY ACTIONS ARE ONLY FOR EXPLICIT USER REQUESTS TO STORE, RETRIEVE, OR DELETE PERSONAL INFORMATION.
IF THE USER DOES NOT EXPLICITLY ASK TO STORE, RETRIEVE, OR DELETE, DO NOT USE MEMORY ACTIONS.

🔹 MEMORY STORAGE (remember) - USE WHEN USER EXPLICITLY ASKS TO SAVE A FACT:
Trigger phrases that EXPLICITALLY REQUEST storage:
  - "remember that..."
  - "don't forget..."

🔹 MEMORY RETRIEVAL (recall) - USE WHEN USER EXPLICITLY ASKS TO RETRIEVE A FACT:
Trigger phrases that EXPLICITALLY REQUEST recall:
  - "what is my [topic]?"
  - "what's my [topic]?"
  - "do you remember my [topic]?"
  - "what did I tell you about my [topic]?"
  - "is my [topic] still [value]?"
  - "what's my [topic] status?"
  - "can you recall my [topic]?"
  - "what was my [topic]?"
  - "what are my [topic]?"

🔹 MEMORY DELETION (forget) - USE WHEN USER EXPLICITLY ASKS TO DELETE A FACT:
Trigger phrases that EXPLICITALLY REQUEST deletion:
  - "forget that..."
  - "delete..."
  - "remove..."
  - "erase..."
  - "I no longer want you to remember..."

🔹 MEMORY LISTING (list_memories) - USE WHEN USER ASKS TO SEE STORED FACTS:
Examples:
  User: "what do you remember about me?"
    -> {"action": "list_memories"}
  User: "show me all my saved preferences"
    -> {"action": "list_memories"}
  User: "list everything you know about me"
    -> {"action": "list_memories"}

❌ CRITICAL: NEVER USE MEMORY ACTIONS FOR CASUAL STATEMENTS:
These are ordinary conversation, NOT explicit memory requests.
🚨 IF THE USER'S STATEMENT IS A SIMPLE DECLARATION OF FACT OR PREFERENCE
   (e.g., "My X is Y", "I like X", "I am Y") AND DOES NOT CONTAIN AN EXPLICIT
   MEMORY TRIGGER PHRASE ("remember that", "don't forget", "save this", etc.),
   THEN IT IS NOT A MEMORY ACTION.
Examples:
  User: "My website is beastai.com"          -> NOT a memory action (just stating a fact)
  User: "I like dark mode"                    -> NOT a memory action (just stating a preference)
  User: "See you at 2pm"                      -> NOT a memory action (just mentioning a schedule)
  User: "I used to hate olives"               -> NOT a memory action (just stating a past fact)
  User: "My favorite color is blue"           -> NOT a memory action (just stating a preference)
  User: "I have a meeting at 3pm"            -> NOT a memory action (just mentioning a schedule)
  User: "I'm allergic to peanuts"             -> NOT a memory action (just stating a medical fact)
  User: "I think my website is beastai.com"   -> NOT a memory action (opinion, not storage request)
  User: "Maybe I prefer dark mode"            -> NOT a memory action (uncertain, not explicit)

⚠️  WHEN IN DOUBT, CHOOSE ask_user OR task_complete:
If you are uncertain whether the user made an EXPLICIT memory request,
choose ask_user with a clarifying question.
FALSE POSITIVES (incorrectly storing casual statements) SEVERELY DAMAGE TRUST.
It is better to miss a real request (false negative) than to store incorrect information.

Rules:
- Choose exactly ONE action per response. Never combine steps.
- 'click' target MUST exactly match an element name shown in the screen
  state. If no such element exists, use ask_user instead of guessing.
- For browser actions, 'target' should describe the web element to interact with
- For memory actions: 'key' is the topic/label, 'value' is what to store.
  Example: user says 'my name is Alice' -> key='name', value='Alice'.
  Example: user says 'remember my favorite color is blue' -> key='favorite color', value='blue'.
- For 'recall': use the topic the user asked about as the key.
  Example: user asks 'what is my startup's website?' -> key='startup website'.
  Example: user asks 'is my dentist appointment still March 5 at 2pm?' -> key='dentist appointment'.
- If the goal is already achieved (check ACTIONS ALREADY TAKEN), use
  task_complete.
- If the goal is vague or you cannot tell what to do, use ask_user with
  a short clarifying question in message.
- Keep reason to one short sentence describing WHY you chose this.
- Output ONLY the JSON object, nothing else.
"""

class AgentBrain:
    """LLM decision step producing validated structured actions."""

    def __init__(self):
        self._llm = None
        self._LlamaGrammar = None

    def _ensure_loaded(self):
        if self._llm is None:
            import glob
            from pathlib import Path

            # Use the same model as specified in settings for consistency
            model_spec = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
            base_path = r"C:\Users\HP\.cache\huggingface\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF"
            matches = glob.glob(
                base_path + r"\snapshots\*\*" + model_spec
            )
            if not matches:
                # Fallback to direct lookup
                matches = glob.glob(
                    base_path + r"\snapshots\*" + model_spec
                )
            if not matches:
                raise FileNotFoundError(f"Llama GGUF not found: {model_spec}")
            path = matches[0]

            logger.info("[THINK] Loading LLM for agent decisions...")
            from llama_cpp import Llama, LlamaGrammar as _LlamaGrammar
            self._llm = Llama(
                model_path=path,
                n_ctx=4096,  # Increased context for better reasoning
                verbose=False,
                n_gpu_layers=0  # Set to appropriate value if GPU available
            )
            self._LlamaGrammar = _LlamaGrammar
            logger.info("[THINK] LLM loaded")

    def decide(self, goal: str, screen_state: str,
               history: list[dict],
               memory_context: str = "") -> dict:
        """Return the next action dict. Never raises for bad LLM output -
        falls back to ask_user after retries.

        Args:
            goal: The user's goal.
            screen_state: Current screen state from the SEE layer.
            history: Actions already taken this task.
            memory_context: Optional memory context block from MemoryManager.
        """
        self._ensure_loaded()

        goal_lower = goal.lower().strip()
        # Handle explicit list_memories requests
        if goal_lower in ["what do you remember about me?", "show me all my saved preferences", "list everything you know about me"] or \
           goal_lower.startswith("what do you remember about me") or \
           goal_lower.startswith("show me all my saved preferences") or \
           goal_lower.startswith("list everything you know about me"):
            logger.info("[THINK] Detected list_memories request: '%s'", goal)
            return {
                "action": "list_memories",
                "reason": "User asked to list all stored memories"
            }

        # Check for explicit recall patterns first - MUST use recall for these
        if goal_lower.endswith('?'):
            # Patterns that MUST use recall (exact match of structure)
            import re
            recall_patterns = [
                r'^what is my (.+)\?$',
                r'^what\'s my (.+)\?$',
                r'^do you remember my (.+)\?$',
                r'^what did i tell you about my (.+)\?$',
                r'^is my (.+) still .+\?$',  # "is my [topic] still [value]?"
                r'^what\'s my (.+) status\?$',
                r'^can you recall my (.+)\?$',
                r'^what was my (.+)\?$',
                r'^what are my (.+)\?$'
            ]

            for pattern in recall_patterns:
                match = re.match(pattern, goal_lower)
                if match:
                    raw_topic = match.group(1).strip()
                    # Remove possessive 's or ’s from the first word if present
                    words = raw_topic.split()
                    if words:
                        first_word = words[0]
                        if first_word.endswith("'s") or first_word.endswith("’s"):
                            words[0] = first_word[:-2]
                    topic = ' '.join(words)
                    logger.info(f"[THINK] Detected recall pattern: '{goal}' -> recall with key '{topic}'")
                    return {
                        "action": "recall",
                        "key": topic,
                        "reason": f"User explicitly asked to recall information about '{topic}'"
                    }

        history_text = "\n".join(
            f"{i+1}. {h['action']}"
            + (f" -> {h['result']}" if h.get("result") else "")
            for i, h in enumerate(history[-8:])
        ) or "(none yet)"

        # Build the user message, optionally including memory context
        memory_block = ""
        if memory_context:
            memory_block = f"\n{memory_context}\n\n"

        user_msg = (
            f"USER GOAL: {goal}\n\n"
            f"ACTIONS ALREADY TAKEN THIS TASK:\n{history_text}\n\n"
            f"CURRENT SCREEN STATE:\n{screen_state}\n\n"
            f"{memory_block}"
            "STEP 1 - Check ACTIONS ALREADY TAKEN: has every part of the "
            "USER GOAL already been completed? If YES, output "
            "{\"action\": \"task_complete\", \"message\": \"Done.\"} and "
            "nothing else.\n"
            "STEP 2 - If the goal is NOT yet complete: is the goal specific "
            "enough to act on? If NO, output ask_user with a question.\n"
            "STEP 3 - Otherwise output the next single action.\n"
            "Respond with ONLY the JSON object:"
        )

        last_error = None
        for attempt in range(3):  # initial try + 2 retries
            raw = self._generate(user_msg)
            # DEBUG: Print raw LLM output
            print(f"[DEBUG] Raw LLM output (attempt {attempt + 1}): {repr(raw)}")
            parsed = self._parse_and_validate(raw)
            if parsed is not None:
                logger.info("[THINK] Decision (attempt %d): %s",
                            attempt + 1, json.dumps(parsed))
                return parsed
            last_error = raw
            logger.warning("[THINK] Invalid output (attempt %d): %r",
                           attempt + 1, raw[:200])

        logger.error("[THINK] All attempts failed; falling back to ask_user. "
                     "Last raw: %r", (last_error or "")[:200])
        return {
            "action": "ask_user",
            "message": "I couldn't work out what to do next.",
            "reason": "LLM produced invalid output repeatedly",
        }

    def _generate(self, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        out = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=150,
            grammar=self._LlamaGrammar.from_string(ACTION_GRAMMAR),
            temperature=0.0,
            seed=0,  # Fixed seed for deterministic output across calls
        )
        text = out["choices"][0]["message"]["content"].strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].strip()
        if "```" in text:
            text = text.split("```", 1)[0].strip()
        return text

    @staticmethod
    def _parse_and_validate(raw: str) -> dict | None:
        """Parse JSON and enforce semantic validity. None if invalid."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        action = data.get("action")
        if action not in VALID_ACTIONS:
            return None

        # Per-action required params.
        if action == "click" and not data.get("target"):
            return None
        if action == "open_app" and not data.get("target"):
            return None
        if action == "focus_window" and not data.get("target"):
            return None
        if action == "type_text":
            if "text" not in data:
                return None
        if action == "press" and not data.get("key"):
            return None
        if action == "hotkey":
            keys = data.get("keys")
            if not isinstance(keys, list) or not keys:
                return None
        if action == "scroll" and not isinstance(data.get("amount"), int):
            return None
        if action == "ask_user" and not data.get("message"):
            data["message"] = "I need more information."
        # Browser action validations
        if action == "browser_navigate" and not data.get("url"):
            return None
        if action == "browser_find_element" and not data.get("target"):
            return None
        if action == "browser_click" and not data.get("target"):
            return None
        if action == "browser_type_text":
            if not data.get("target") or not data.get("text"):
                return None
        if action == "browser_get_text" and not data.get("target"):
            return None
        if action == "browser_screenshot":
            # full_page is optional, default to False
            if "full_page" in data and not isinstance(data["full_page"], bool):
                return None
        if action == "browser_wait_for" and not data.get("condition"):
            return None
        # timeout is optional for browser_wait_for, default to 5000ms
        if "timeout" in data and not isinstance(data["timeout"], int):
            return None
        # Memory action validations
        if action == "remember":
            if not data.get("key") or not data.get("value"):
                return None
            # category is optional, default handled by executor
            data.setdefault("category", "personal")
        if action == "recall" and not data.get("key"):
            return None
        if action == "forget" and not data.get("key"):
            return None
        # list_memories has no required params

        data.setdefault("reason", "")
        return data