# ----------------------------------------------------------
#  Settings command helper for Sapphire CLI
# ----------------------------------------------------------
import json, shlex, textwrap
from typing import Dict, Any, Tuple
from core.param_store import ParamStore  
from core.sapphire_core import MemoryNode 
import datetime


STORE_PATH = "./presets/phasers_params.json"
STORE = ParamStore(STORE_PATH)
 
HELP_TEXT = textwrap.dedent("""
    config                  → show live parameters
    config list             → list saved presets
    config load  NAME       → load preset into live params
    config saveas NAME      → save current live params as new preset
    config set KEY VAL      → change a single param (not auto-saved)
    config update NAME KEY VAL   → patch a stored preset
    config delete NAME      → remove a preset
    cloud                   → render active UMB wordcloud    
    load                    → Load a UMB preset
    umb                     → Display current UMB
    clean                   → Starts new UMB / 2 paramaters required
    reload                  → Reload Default UMB
    config help             → show this help
""").strip()

# Descriptions for each parameter (extend / edit as you wish)
DESCR = {
    "temp"               : "softmax temperature (creativity)",
    "top_p"              : "nucleus-sampling cutoff",
    "top_k"              : "k-best vocabulary cap (0 = disabled)",
    "repetition_penalty" : "discourage recently used tokens",
    "max_forward_tokens" : "max tokens per forward pass",
    "max_reply_sentences": "hard stop after N sentences",
    "top_n"              : "memory stack depth",
    "weight"             : "memory bias-scale coefficient",
    "tau"                : "rank-based decay for memory scores",
    "lam"                : "global decay on memory logits",
    "n_sieve"            : "LLM passes before SBERT rank choose",
    "inference_mem"      : "Switches memory source: / 0-prompt only memory scan / 1-prompt -with- inference scan & output append",
    "sieve_rank_mem"     : "Switches ranking source: / 0-prompt only / 1-with prompt memory / 2-prompt & inference memory",
    "sigma"              : "floor of exponential memory decay function",
    "prompt_constr"      : "prompt sequence constructor (refer to docs)",
    "top_t"              : "length of memory tail",
}


def handle_settings_command(
    user_line: str,
    live_params: Dict[str, Any]
) -> Tuple[Dict[str,Any], str]:
    """
    Parse and execute a 'settings …' command.
    Returns (possibly updated live_params, message_to_user)
    """
    try:
        tokens = shlex.split(user_line)
    except ValueError as e:
        return live_params, f"[config] parse error: {e}"

    if len(tokens) == 1:                # plain 'config' → show live
        return live_params, json.dumps(live_params, indent=2)

    cmd = tokens[1].lower()

    # ---------- list ----------
    if cmd == "list":
        names = STORE.list()
        return live_params, "presets: " + ", ".join(names) if names else "no presets"

    # ---------- load ----------
    if cmd == "load" and len(tokens) == 3:
        name = tokens[2]
        try:
            live_params = STORE.get(name)
            return live_params, f"[config] loaded preset '{name}'"
        except KeyError:
            return live_params, f"[config] no preset named '{name}'"

    #------------ clean default UMB
       # moved to chat
    # ---------- saveas ----------
    if cmd == "saveas" and len(tokens) == 3:
        name = tokens[2]
        try:
            STORE.add(name, live_params)
            STORE.save()
            return live_params, f"[config] saved current params as '{name}'"
        except KeyError:
            return live_params, f"[config] preset '{name}' already exists"

    # ---------- set (live only) ----------
    if cmd == "set" and len(tokens) == 4:
        key, val = tokens[2], tokens[3]
        if key not in STORE.REQUIRED_KEYS:
            return live_params, f"[config] unknown key '{key}'"
        live_params[key] = _coerce(val)
        return live_params, f"[config] live param '{key}' set to {val} (not saved)"

    # ---------- update stored preset ----------
    if cmd == "update" and len(tokens) == 5:
        name, key, val = tokens[2], tokens[3], tokens[4]
        try:
            STORE.update(name, **{key: _coerce(val)})
            STORE.save()
            return live_params, f"[config] preset '{name}' updated ({key}={val})"
        except (KeyError, ValueError) as e:
            return live_params, f"[config] {e}"

    # ---------- delete ----------
    if cmd == "delete" and len(tokens) == 3:
        name = tokens[2]
        try:
            STORE.delete(name)
            STORE.save()
            return live_params, f"[config] deleted preset '{name}'"
        except KeyError:
            return live_params, f"[config] no preset named '{name}'"

    # ---------- help ----------
    if cmd == "help":
        return live_params, HELP_TEXT

    if cmd == "keys":
        lines = []
        for k in STORE.REQUIRED_KEYS:
            cur = live_params.get(k, "∅")
            lines.append(f"{k:20} = {cur:<8}  — {DESCR.get(k, '')}")
        return live_params, "\n".join(lines)

    return "[error]", "[config] error in command - data not updated."

# ------------------------- util -----------------------------
def _coerce(val:str):
    """Try to cast val to int or float when suitable, else str."""
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            continue
    # bool literals?
    if val.lower() in ("true","false"):
        return val.lower() == "true"
    return val
