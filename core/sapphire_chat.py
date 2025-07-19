from __future__ import annotations

print("\n 📀 booting GPT-2-mini 💎 Sapphire Alpha v0.13.3 \n")

MAX_FORWARD_TOKENS = 75
import warnings
warnings.filterwarnings("ignore") 


from collections.abc import Mapping
import warnings, os, re, json, glob, argparse, shutil, math, time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple
import torch, torch.nn.functional as F
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, default_data_collator
)
from sentence_transformers import SentenceTransformer, util
#import language_tool_python as lt
import nltk 
nltk.download('punkt') # you can comment out dwnload after first shot gets i

from  cli.settings_manager import  handle_settings_command
### Hardware check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###
from core.param_store import ParamStore
# ────────────────────────────────────────────────────────────────────────
# 1.   PROMPT CLASSIFIER
# ────────────────────────────────────────────────────────────────────────
class PromptClassifier:
    CAUSAL_RE  = re.compile(r"\bwhy\b|\bhow (?:do|does)\b|\bbecause\b", re.I)
    CONDIT_RE  = re.compile(r"\bif .* then\b", re.I)
    ARITH_RE   = re.compile(r"\d+\s*[\+\-\*/]\s*\d+", re.I)

    def classify(self, prompt: str) -> str:
        if self.ARITH_RE.search(prompt):  return "arithmetic"
        if self.CONDIT_RE.search(prompt): return "conditional"
        if self.CAUSAL_RE.search(prompt): return "causal"
        return "chat"

# ────────────────────────────────────────────────────────────────────────
# 2.   NHCE MEMORY + SOFT‑LOGIT Sampler             
# ────────────────────────────────────────────────────────────────────────
from core.sapphire_core import NHCE_Engine, ManualSampler,  GPT2CustomTrainer   
from core.sapphire_core import MemoryLoader, MemoryNode


# ----------------------------

# ────────────────────────────────────────────────────────────────────────
# 4.   CLI  – CHAT with REASONING and presence
# ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode")

    t = sub.add_parser("train", help="fine‑tune on .txt")
    t.add_argument("txt")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--out", default="./ckpt")

    sub.add_parser("chat", help="interactive chat")

    args = ap.parse_args()
    trainer = GPT2CustomTrainer()

    if args.mode == "train":
        trainer.finetune_txt(args.txt, args.epochs, args.out)
        return



    # -------- chat ----------
    trainer.maybe_load_latest(args.out if hasattr(args, "out") else "./ckpt")
    nhce  = NHCE_Engine(trainer.model, trainer.tok)
    gen   = ManualSampler(trainer.model, trainer.tok, nhce, _embedder = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE))
    clf   = PromptClassifier()
    loader = MemoryLoader()
    
    live_params = {
                    "temp":0.567,
                    "top_n":int(10),
                    "top_p":0.72,
                    "top_k":42,
                    "repetition_penalty":1.35,
                    "max_forward_tokens":55,
                    "max_reply_sentences":3,
                    "weight":.42,
                    "tau":0.246,
                    "lam":0.65,
                    "n_sieve":7,
                    "inference_mem":1,
                    "sieve_rank_mem":2,
                    "sigma":.222,
                    "prompt_constr": str("memory;prompt;memory;tail;prompt;memory;prompt;"),
                    "top_t": int(7),
    }

            
    def update_model_with_live_params(lp, gen):
        gen.top_n = lp.get("top_n", 9)
        gen.temp = lp.get("top_p", 0.7)
        gen.top_k = lp.get("top_k", 20)
        gen.pen = lp.get("repetition_penalty", 1.2)
        gen.max_tokens = lp.get("max_forward_tokens", 60)
        gen.max_reply_sentences = lp.get("max_reply_sentences", 3)
        gen.b_scale = lp.get("weight", 0.5)
        gen.tau = lp.get("tau", 0.2)
        gen.lam = lp.get("lam", 0.6)
        gen.n_sieve = lp.get("n_sieve", 3)
        gen.inference_mem = lp.get("inference_mem", 1)
        gen.sieve_rank_mem = lp.get("sieve_rank_mem", 1)
        gen.sigma = lp.get("sigma", 0.2)
        gen.prompt_constr = lp.get("prompt_constr", "memory;tail;prompt")
        gen.top_t = lp.get("top_t", 6)
            

    
    print("\n───────────────────────────────────────────────────")
    print("  💎 SAPPHIRE | GPT-2-mini + Reasoning micro🕉 core  ")
    print("─────────────────────────────────────────────────────\n")
    
    print(" 🖥  rendering device: ", DEVICE)
    print(" 🆘  type 'config help' for instructions\n\n")
    
    
#=============================================================================
###############  WORD CLOUD GENERATOR
#=============================================================================
    _STOPWORDS = {
        "a","an","and","are","as","at","be","but","by","for","from","has","have",
        "he","her","his","i","in","is","it","its","me","my","of","on","or","our",
        "s","she","that","the","their","them","they","this","to","was","we","were",
        "what","with","you","your", "am", "do", "who", "so"
    }

    _TEXT_ATTRS = ("inp", "output")        

    def _field(mem, attr, default=None):
        if isinstance(mem, Mapping):
            return mem.get(attr, default)
        return getattr(mem, attr, default)


    # ▼  pick the *first* usable text attribute; if none, skip the memory ---
    def _memory_text(mem):
        for attr in _TEXT_ATTRS:
            val = _field(mem, attr)
            if isinstance(val, str) and val.strip():
                return val
        return None  

    def _tokenize(text: str):
        # alpha words only, lower-cased
        return re.findall(r"[a-zA-Z']{2,}", text.lower())


    def _word_counts_from_umb(umb, weight_fn):
        counts = Counter()
        for mem in umb:
            text = _memory_text(mem)
            if not text:
                continue              # skip nodes without usable text

            weight = float(weight_fn(mem)) or 0.0
            if weight == 0.0:
                continue

            for tok in _tokenize(text):
                if tok in _STOPWORDS:     # ▼  filter stop-words
                    continue
                counts[tok] += weight
        return counts

    def _render_ascii_cloud(word_counts: Counter[str], top_n: int = 35) -> str:
        term_width = shutil.get_terminal_size((80, 20)).columns
        most_common = word_counts.most_common(top_n)
        if not most_common:
            return "[UMB empty – nothing to cloud]"

        max_freq = most_common[0][1]
        lines: list[str] = []
        for word, freq in most_common:
            reps = max(1, round((freq / max_freq) * 10))
            blob = (" " + word) * reps
            lines.append(blob.strip().center(term_width))
        return "\n".join(lines)


    # ----------------------------
    # public API
    # ----------------------------

    
        
    def handle_cloud_command(
        unified_memory_bank: Iterable[Any] | None,
        *,
        weight_fn: Callable[[Any], float] | None = None,
        top_n: int = 35,
    ) -> None:
        if not unified_memory_bank:
            print("[UMB is empty – nothing to display]")
            return

        # Default: salience attr or key → else 1.0
        if weight_fn is None:
            weight_fn = lambda m: _field(m, "salience", 1.0)

        counts = _word_counts_from_umb(unified_memory_bank, weight_fn)
        cloud = _render_ascii_cloud(counts, top_n)
        print("\n" + cloud + "\n")

    
# ─────── MAIN CLI loop
    
    
    print(" 📓 chat history\n---")
    for chatlog in nhce.tail_memories(n=4):
        print(chatlog)
    
    while True:
        print('---')
        usr = input(" 🧠 > ")
        
        if len(usr) == 0: #nextusr
            continue
        
        if usr.lower() == "exit": break
        
        if usr.lower().strip() == "cloud":
            handle_cloud_command(nhce.memory)
            continue  # Skip standard generation

        if usr.lower().strip() == "tail":
            print("---")
            for chatlog in nhce.tail_memories(n=4):
                print(chatlog)
            print("-----")
            continue  # Skip standard generation
            
        if usr.lower().strip() == "load":

            loader.choose_memory_file()
            nhce.memory_file = loader.memory_file #update cross system
            nhce.memory = loader.load_memory()
            print(" 📓 chat history \n")
            for chatlog in nhce.tail_memories(n=3):
                print(chatlog)
            continue  # Skip standard generation            
        
        if usr.lower().strip() == "umb":
            print(">> 💾 ", nhce.memory_file)
            continue  # Skip standard generation        
        
        
        if usr.lower().strip()[0:5] == "clean": 
            
            anchors = usr.split()
            
            if len(anchors) == 3:
                prompt_anchor = anchors[1]
                llm_anchor = anchors[2]
                    
                print("🔢 initializing UMB", flush=True )
                # seed with root‑identity prompt
                now = datetime.utcnow().isoformat()
                root = MemoryNode(now, prompt_anchor, llm_anchor, "identity", 0.95, 1.0, 1.0)
                nhce.memory = []  #wiped
                nhce.memory.append(root)
                nhce.memory_file = "./memory/emergence_UMB.json"
                with open(nhce.memory_file, "w") as fh:
                    json.dump([root.__dict__ for m in nhce.memory], fh, indent=2)
                
                print("[config] UMB initialized") 
                continue # skip gen
            print("[config] ❓ your did not provide init root pair of 'prompt inference'.\n[config] please append to 'clean' command ") 
            continue #nextusr
            
        if usr.lower().strip() == "reload":
            with open("./memory/emergence_UMB.json") as fh:
                raw = json.load(fh)
            nhce.memory = [MemoryNode(**m) for m in raw]
            print("\n>> 🚅 ", "UMB reloaded")
            continue  # Skip standard regeneration
            
            
        if usr.lower().startswith("config"):
                live_params, msg = handle_settings_command(usr, live_params)
                print(msg)            # or route to console log
                update_model_with_live_params(live_params, gen)
                continue 

        else:
            reply = gen.generate(usr, write_memory=True)

        print("\n 🖥  > ", nhce.enforce_sentence_boundaries(reply).replace(usr, "", 1).strip(), "\n--")

if __name__ == "__main__":
    main()
