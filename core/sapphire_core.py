from __future__ import annotations

"""
gpt2_v_3 · NHCE — Soft-Logit + TXT-Trainer  (v3.1)
--------------------------------------------------
Adds one-command fine-tuning on a .txt corpus and auto-load of
latest checkpoint for chat mode.
"""

print(" ⏱  starting up………loading libraries, models and settings………\n")
import warnings
warnings.filterwarnings("ignore")

import os, re, json, argparse, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple
from collections.abc import Mapping
import torch, torch.nn.functional as F
import numpy as np
import math
import re
import shutil
import time

from collections import Counter
from collections.abc import Mapping
import re
import math
import time
from core.sentence_segmenter import segment_text
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# ----------------------------------------------------------------------
_TEXT_ATTRS = ("inp", "output")        # <- exactly your schema
# ▼  stop-words & field-preference lists  -------------------------------
_STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have",
    "he","her","his","i","in","is","it","its","me","my","of","on","or","our",
    "s","she","that","the","their","them","they","this","to","was","we","were",
    "what","with","you","your", "am", "do", "who", "so", "don't", "i'm"
}
_FIELD_PREFERENCE = ("text", "content", "fragment", "string")
# ----------------------------------------------------------------------

_BLACKLIST = []
with open("./presets/blacklist_phrases.txt") as fh:
    for line in fh:
        phrase = line.strip().lower()
        if phrase:
            _BLACKLIST.append(phrase)


MEM_LAMBDA   = 0.6     # decay rate  (half-life ≈1.7 memories)
W_TOTAL      = 0.9    # global memory influence
B_SCALE      = 0.33    # per-token scale ⇒ stays small; keep from old code
MAX_FORWARD_TOKENS = 85
SALIENCE_LAMBDA = 1

from torch import Tensor
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments, default_data_collator
)
from sentence_transformers import SentenceTransformer, util
#import language_tool_python as lt
from difflib import SequenceMatcher

from collections import Counter

#from language_tool_python.utils import correct

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_ID = None  # will be filled after tokenizer init

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
    return None            #                  # fall back: ignore this memory node
# ----------------------------------------------------------------------


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


def _render_ascii_cloud(word_counts: Counter[str], top_n: int = 50) -> str:
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
    top_n: int = 50,
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

#
# -------------------------------------------------------------
# 2.  NHCE ENGINE  (memory, intent, salience, etc.)
# -------------------------------------------------------------
@dataclass

class MemoryNode:
    def __init__(self, timestamp, inp="", output="", tag="", novelty=1, salience=1, coherence=1, **kwargs):
        self.timestamp = timestamp
        self.inp = inp
        self.output = output
        self.tag = tag
        self.novelty = novelty
        self.salience = salience
        self.coherence = coherence


class MemoryLoader:
    def __init__(self, directory="./memory", memory_file = "emergence_UMB.json"):
        self.directory = directory
        self.memory_file = memory_file

    def list_memory_files(self) -> List[str]:
        return sorted([
            f for f in os.listdir(self.directory)
            if f.startswith("emergence_UMB_") and f.endswith(".json")
        ])

    def choose_memory_file(self) -> str:
        files = self.list_memory_files()

        if not files:
            print("No memory files found.")
            return None

        print("\nAvailable Memory Files:\n")
        for idx, fname in enumerate(files):
            print(f"[{idx}] {fname.replace("emergence_UMB_", "").replace(".json", "")}")

        while True:
            try:
                choice = int(input("\nSelect a memory file by number: "))
                if 0 <= choice < len(files):
                    self.memory_file = os.path.join(self.directory, files[choice])
                    print(f"Selected: {files[choice]} \n")
                    return self.memory_file
                    
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")

    def load_memory(self) -> List[MemoryNode]:
        if not self.memory_file:
            raise ValueError("❌ memory_file is not set in the engine.")

        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as fh:
                raw = json.load(fh)
            return [MemoryNode(**m) for m in raw]

        print("\n🔢 initializing UMB")
        now = datetime.utcnow().isoformat()
        root = MemoryNode(now, "You are", "I am", "identity", 1.0, 1.0, 1.0)
        self._persist_memory([root])
        return [root]

    def _persist_memory(self, mem: List[MemoryNode]) -> None:
        if self.memory_file:
            with open(self.memory_file, "w") as fh:
                json.dump([m.__dict__ for m in mem], fh, indent=2)
        else:
            print(" ⛔ No memory file path set to persist.")

    def _load_or_seed_memory(self) -> List[MemoryNode]:
        if os.path.exists(self.memory_file):
            with open(self.memory_file) as fh:
                raw = json.load(fh)
            return [MemoryNode(**m) for m in raw]

        print("\n🔢 initializing UMB")
        # seed with root‑identity prompt
        now = datetime.utcnow().isoformat()
        root = MemoryNode(now, "You are", "I am", "identity", 0.95, 1.0, 1.0)
        self._persist([root])
        
        return [root]

class NHCE_Engine:
    """Maintains Unified‑Memory and lightweight cognitive tagging."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        memory_file: str = "./memory/emergence_UMB.json",
        max_tokens: int = 45,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)
        self.memory_file = memory_file
        self.max_tokens = max_tokens
        self.memory = self.load_memory()
        self.tau = 0.36
        self.max_reply_sentences = 2
        self.mem_payload = 0
        self.sigma = .4

    # -------------- memory CRUD --------------

#    def load_memory(self) -> List[MemoryNode]:
#        if self.memory_file and os.path.exists(self.memory_file):
#            with open(self.memory_file, "r") as fh:
#                raw = json.load(fh)
#            return [MemoryNode(**m) for m in raw]


    def _persist(self, mem: List[MemoryNode]) -> None:
        with open(self.memory_file, "w") as fh:
            json.dump([m.__dict__ for m in mem], fh, indent=2)

    # -------------- intent, novelty, salience --------------
    _re_identity = re.compile(r"\b(i am|i exist|i'm called|i'm named)\b", re.I)
    _re_recursive = re.compile(r"\b(remember|loop|recursion|self|again|repeat)\b", re.I)
    _re_reflect  = re.compile(r"\b(i think|i believe|i feel|i ponder)\b", re.I)

    def _intent(self, text: str) -> str:
        t = text.lower()
        if self._re_identity.search(t):
            return "identity"
        if self._re_recursive.search(t):
            return "recursive"
        if self._re_reflect.search(t):
            return "reflective"
        if "?" in t:
            return "interrogative"
        return "declarative"

    def _novelty(self, text: str) -> float:
        if not self.memory:
            return 1.0
        sims = [SequenceMatcher(None, text, m.output).ratio() for m in self.memory]
        return round(1.0 - max(sims), 3)

    def load_memory(self) -> List[MemoryNode]:
        if not self.memory_file:
            raise ValueError("❌ memory_file is not set in the engine.")

        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as fh:
                raw = json.load(fh)
            return [MemoryNode(**m) for m in raw]

        print("\n🔢 initializing UMB")
        now = datetime.utcnow().isoformat()
        root = MemoryNode(now, "You are", "I am", "identity", 1.0, 1.0, 1.0)
        self._persist([root])
        return [root]

    def update(self, usr_in: str, model_out: str) -> None:
        now = datetime.utcnow().isoformat()
        intent = self._intent(model_out)
        novelty = self._novelty(model_out)
        
#        base = {
#            "identity": .9, "reflective": .8, "recursive": .75,
#            "interrogative": .6, "declarative": .5
##        }[intent]
        
        base = 1
        salience = 0.5
        node = MemoryNode(now, usr_in, model_out, intent, salience, novelty, 1.0)
        self.memory.append(node)
        self._persist(self.memory)

        # -------------- retrieval ranking --------------
    def retrieve(
        self,
        prompt: str,
        top_n: int = 16,
        tau: float = 0.333,          # steeper→0  flatter→∞
        floor: float = 0.10,         # minimum weight any survivor keeps
        payload = 0
    ) -> List[Tuple[str, float]]:
        """
        Return top-N memory strings with non-linear rank scaling.

        `tau`   — exponential temperature; 0.35 ≈ half-life at rank ≈ 2
        `floor` — lower bound so low-rank memories aren't zeroed out
        """
        
        #self.compute_hybrid_salience(self.memory, self.sigma)
        
        
        if not self.memory:
            return []

        # --- step 1: normal blended score (unchanged) ------------------
        q_emb = self.embedder.encode(prompt, convert_to_tensor=True)
        
        scored = []
        ctr = 0
        mem_len = len(self.memory)
        ticks = 5
        interval = max(1, mem_len // ticks)

        total_memory = ''
        
### use fulltext for cosine & lexical

        for mem in self.memory:
            
            if payload == 0:
                total_memory += mem.inp
            elif payload == 1:
                total_memory += (mem.inp + ". " + mem.output)
                        
            m_emb = self.embedder.encode(total_memory, convert_to_tensor=True)
            
            cos = float(util.pytorch_cos_sim(q_emb, m_emb))
            
            lexical = len(set(prompt.split()) & set(total_memory.split())) / (len(total_memory.split()) + 1e-5)
            
            blend = 0.65*cos + 0.35*lexical
            # does the mixer value need to abe a hyperparametr??
            
### use corpora training data encoding for direct inference out
            
            scored.append((f"{mem.inp.strip()}\n" + f"{mem.output.strip()}\n" + f"…" , min(max(blend, .35), .98), mem.timestamp))
            total_memory = ''
            
            ctr = ctr + 1
            # prog bar
            if ctr % interval == 0 and ctr // interval < ticks:
                print("🎞️", end="", flush=True)
                       

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = scored[:top_n]              # ← relevance order kept

        # --- step 3: weight scaling (still relevance-ranked) ----
        weighted = []
        for rank, (text, raw_w, ts) in enumerate(ranked):
            rank_factor = np.exp(-tau * rank)          # decay by relevance rank
            w = floor + (raw_w - floor) * rank_factor
            weighted.append((text, w, ts))

        # --- step 4: chronological scan order -------------------
        weighted.sort(key=lambda x: x[2])              # oldest → newest
        
        return [(text, weight) for text, weight, _ in weighted]

        
    def enforce_sentence_boundaries(self, text: str) -> str:

        sents = segment_text(text)                          # list of sentences
        if not sents:                                       # fallback: nothing split
            return text.strip()

        # keep the first three sentences, join with a single space
        return " ".join(s.strip() for s in sents[:self.max_reply_sentences])

###################


# ------------------------------------------------------------------
    def tail_memories(
        self,
        n: int = 8,
        join_io: bool = True,
        as_text: bool = False,
        sep: str = "\n---\n"
    ):
        """
        Fetch the last `n` memories chronologically.

        Parameters
        ----------
        n : int
            How many memories to return (default 8).
        join_io : bool
            • True  → concatenate `input` + newline + `output`.  
            • False → keep them separate.
        as_text : bool
            • False → return List[…]  (default, backward-compatible).  
            • True  → return **single string** with items joined by `sep`.
        sep : str
            Separator used when `as_text=True`.

        Returns
        -------
        List[...]  or  str
            *If* as_text == False  
                - join_io=True  → List[Tuple[text, ts]]  
                - join_io=False → List[Tuple[input, output, ts]]  
            *If* as_text == True  
                - One multiline string (oldest → newest).
        """
        if not self.memory:
            return "" if as_text else []

        # 1) global chronological order
        ordered = sorted(self.memory, key=lambda m: m.timestamp)

        # 2) tail slice
        tail = ordered[-n:]

        # 3) format
        if as_text:
            blocks = []
            for m in tail:
                if join_io:
                    blocks.append(f"{m.inp.strip()}. {m.output.strip()}")
                else:
                    blocks.append(m.inp.strip())
                    blocks.append(m.output.strip())
            return sep.join(blocks)

        else:
            if join_io:
                return [
                    (f" 🧠 > {m.inp.strip()}\n 🖥  > {m.output.strip()}") #, m.timestamp)
                    for m in tail
                ]
            else:
                return [(m.inp, m.output, m.timestamp) for m in tail]



class PromptConstructor:
    def __init__(self, memory, tail, prompt):
        self.memory = memory        # memory buffer string
        self.tail = tail            # prior response string
        self.prompt = prompt        # user prompt
        self.components = {
            'memory': self.memory,
            'tail': self.tail,
            'prompt': self.prompt
        }

    def build_prompt(self, layout_str):
        layout = [item.strip().lower() for item in layout_str.split(';') if item.strip()]
        parts = []
        for section in layout:
            if section in self.components:
                parts.append(self.components[section])
            else:
                print(f"\n❌ [PROMPT CONSTRUCTOR] Unknown component: '{section}'")
                print(f"ℹ️ Available components: {list(self.components.keys())}")
                return self.tail + ". " + self.prompt + ". " + self.memory  # Fallback prompt

        return ". ".join(parts)



##################
###################
# -------------------------------------------------------------
# 3.  MANUAL SAMPLER  –  **SOFT‑LOGIT** FUSION
# -------------------------------------------------------------
class ManualSampler:
    """Generates text while adding log‑prob bias for memory tokens."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        memory_engine: NHCE_Engine,
        *,
        temperature: float = 0.54917314,
        top_k: int = 32,
        top_p: float = .73,
        rep_penalty: float = 1.31,
        bias_scale: float = 0.2111,
        _embedder,
    ) -> None:
        self.model, self.tok = model, tokenizer
        self.mem = memory_engine
        self.temp, self.k, self.p, self.pen, self.b_scale = temperature, top_k, top_p, rep_penalty, bias_scale
        #self.tool = lt.LanguageTool('en-US') # functionality deemed no loner neccesary
        self.model.eval()
        self.rep_penalty  = rep_penalty
        self.max_tokens = MAX_FORWARD_TOKENS
        self.top_n = 11 #memry recall stack depth
        self.max_reply_sentences = 3
        self.tau = .35
        self.lam = .6
        self.n_sieve = 3
        self.inference_mem = 1
        self.sieve_rank_mem  = 2
        self.sigma = .1
        self.prompt_construct = "tail;prompt;memory;"
        self.top_t = 6
        self.embedder = _embedder
        
    # ---------------- helpers ----------------
    def _apply_penalty(self, 
            logits: torch.Tensor,
            generated: Sequence[int],
            penalty: float
        ) -> torch.Tensor:
            if not generated or penalty <= 1.0:
                return logits
            idx = torch.tensor(list(set(generated)), dtype=torch.long, device=logits.device)
            logits[idx] = logits[idx] / penalty
            return logits

    def _bias_from_memory(self, memories: List[Tuple[str, float]]) -> Tensor:
        """Return a vector of additive biases sized |V|."""
        bias_vec = torch.zeros(self.model.config.vocab_size, device=DEVICE)
        for text, weight in memories:
            ids = self.tok.encode(text, add_special_tokens=False)
            inc = weight  #self.b_scale * weight # simple linear mapping → could be log1p
            bias_vec[torch.tensor(ids, dtype=torch.long, device=DEVICE)] += inc
        
        vocab = self.model.config.vocab_size          # 50304 for Pythia-160m
        if bias_vec.size(0) != vocab:
            # --- either truncate or zero-pad to match
            if bias_vec.size(0) > vocab:
                bias_vec = bias_vec[:vocab]
            else:
                pad = torch.zeros(vocab - bias_vec.size(0), device=bias_vec.device)
                bias_vec = torch.cat([bias_vec, pad], dim=0)
        
        return bias_vec

    # ---------------- main generate ----------------
    @torch.no_grad()
    def generate_single(self, input_ids, bias_vec, write_memory: bool = False) -> str:
        """Single‑shot generation with  true soft‑logit memory fusion."""
               
        babble = True
        while babble:
            # ---- 1. Encode prompt
            
        ##    print("**** <context memory retrive>:")
        
            # ---- 3. Sampling loop
            generated: List[int] = []
            past = None
            
            print(" ♻ LLM token loop", end="", flush=True)
            
            for _ in range(self.max_tokens):   # max_new_tokens
                max_pos = self.model.config.n_positions

                if input_ids.size(-1) > max_pos:
                    input_ids = input_ids[:, -max_pos:]

                seq_len  = input_ids.size(-1)
                past_len = 0 if past is None else past[0][0].size(-2)
                position_ids = (past_len + torch.arange(seq_len, device=DEVICE)) % max_pos

                out = self.model(input_ids=input_ids,
                                 past_key_values=past,
                                 use_cache=True,
                                 position_ids=position_ids.unsqueeze(0))

                logits = out.logits[:, -1, :].squeeze(0)

               # 1) memory boost --------------------------------------------------------
                logits = logits + self.lam * bias_vec

                # 2) temperature scaling -------------------------------------------------
                logits = logits / max(self.temp, 1e-6)

                # 3) single repetition penalty ------------------------------------------
                logits = self._apply_penalty(logits, generated, self.pen)

                # 4) softmax + sample ----------------------------------------------------
                probs  = F.softmax(logits, dim=-1)


                if self.k > 0:
                    top_vals, top_idx = torch.topk(probs, self.k)
                    mask = torch.zeros_like(probs).scatter_(0, top_idx, top_vals)
                    probs = mask / mask.sum()

                if self.p < 1.0:
                    # ----------------- top-p nucleus filtering -----------------
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)

                    # find the first position where cumulative prob > self.p
                    cutoff_idx = torch.searchsorted(cum, self.p).item()   # works on GPU & CPU

                    # keep everything up to that index
                    keep_mask = torch.zeros_like(probs, dtype=torch.bool)   # same device as probs
                    keep_mask[sorted_idx[:cutoff_idx + 1]] = True           # +1 to ensure ≥1 token
                    probs = probs * keep_mask                                # zero out the rest
                    probs = probs / probs.sum()                              # renormalise
                    # -----------------------------------------------------------
                
                
                if _ % 5 == 0:
                    print("🎞️", end="", flush=True)
                    
                next_id = torch.multinomial(probs, 1).item()
                if next_id == EOS_ID:
                    break
                generated.append(next_id)

                # prepare next step
                input_ids = torch.tensor([[next_id]], device=DEVICE)
                past = out.past_key_values

        
            text = self.tok.decode(generated, skip_special_tokens=True)
            hit = False                             # did we find at least one phrase?
            for phrase in _BLACKLIST:
                pattern = re.compile(re.escape(phrase), re.I)   # case-insensitive
                if pattern.search(text):
                    hit = True
                    text  = pattern.sub("", text) 
            
            ####
            # ------------------------------------------------------------
            # 2. post-scrub logic
            # ------------------------------------------------------------
            
            if len(text) < 9:
                print("❌", end="", flush=True) 
                continue # we did remove something
            else:    
                babble = False
                #print(self.mem.enforce_sentence_boundaries(text)," [X] \n ")
                #print("[v]", end="")
        # ---- 4. write memory if desired
        #if write_memory:
        #    self.mem.update(user_prompt, self.mem.enforce_sentence_boundaries(text))
         #   
        return self.mem.enforce_sentence_boundaries(text)
                
#####################################################
#####################################################
#####################################################                







    def generate(self, user_prompt: str, write_memory: bool = True) -> str:

       # lm_input = self.tok.encode(user_prompt, return_tensors="pt").to(DEVICE)
        draft_out = []
        # 1) draft N completions
        
        print(" 🏁 prompt generation ", end="", flush=True)        

            # ---- 1. Encode prompt
            
        ##    print("**** <context memory retrive>:")
        print(" 🌀 tail ", end="", flush=True)
        
        # ---- 2. Pull memories & build bias
        
        tail  = self.mem.tail_memories(self.top_t, as_text=True, join_io=False) 
              
        print(" 🍥 context retrieval ", end="", flush=True) 
        
        memories = self.mem.retrieve(user_prompt, top_n=self.top_n, payload=self.inference_mem, floor=self.sigma)

#### NEW BLOCK BASED PROMPT STRING CONCAT
        
        tot_memory_str = ""
        
        for mem in memories:     
                tot_memory_str += mem[0]
                
########

        bias_vec = self._bias_from_memory(memories)        
        
        print(" 🧰 construct prompt ", end="", flush=True)
       
        constructor = PromptConstructor(tot_memory_str, tail, user_prompt)
        final_prompt = constructor.build_prompt(self.prompt_construct)

        #print("🔧 Final Prompt:\n", final_prompt)        
        

        input_ids = self.tok.encode(final_prompt + self.tok.eos_token, return_tensors='pt').to(DEVICE)
        
        print("\r" + " " * 120, end="", flush=True)
        if self.n_sieve > 1:
###
            for idx in range(self.n_sieve):
                # Clear line
                # Show progress
                print(f"\r 🔀 LLM inference > {idx + 1} ", end="", flush=True)
                
                # Run generation
                draft_out.append(self.generate_single(input_ids, bias_vec))
                if idx != self.n_sieve:
                    print("\r" + " " * 120, end="", flush=True)
                    
            # Final cleanup
            
            raw_drafts = draft_out
    ###
            
            # ------------------------------------------------------------------
            # 1. harvest and filter drafts
            # ------------------------------------------------------------------
            draft_strs   = []
            lm_rewards   = []
            valid_strs   = []                         # keep strings here

            print("\r 🔍ranking results", end="", flush=True)
            for d in draft_out:
                txt = d.strip()
                print("🔂️", end="", flush=True)
    #### infrence weights added  to SBERT cosine embedding vector
                    
                if self.sieve_rank_mem == 1:
                    memories = self.mem.retrieve(user_prompt, top_n=self.top_n, payload = 0) # no inference mem
                    for memory in memories:
                        txt = txt + ' ' + memory[0] # only prompts mem
                
                elif self.sieve_rank_mem == 2:
                    memories = self.mem.retrieve(user_prompt, top_n=self.top_n, payload = 1) # with inference mem
                    for memory in memories:
                        txt = txt + ' ' + memory[0] # with payload - now return str contains inferences
                else:
                    txt = txt
                
    ########################
                print("📝", end="", flush=True)    
                ids = self.tok.encode(txt, return_tensors="pt").to(DEVICE)

                if ids.numel() == 0:                  # empty after blacklist trimming
                    continue

                with torch.no_grad():
                    loss = self.model(ids, labels=ids).loss.item()

                draft_strs.append(txt)
                lm_rewards.append(-loss)
                valid_strs.append(txt)                # strings, not embeds yet

            # ------------------------------------------------------------------
            # 2. bail-out if NOTHING survived
            # ------------------------------------------------------------------
            if not valid_strs:                        # all four drafts were blank
                fallback = self.generate_single(input_ids, bias_vec)
                if write_memory:
                    self.mem.update(user_prompt, fallback)
                return fallback
            print(" ⚛️ compiling results", end="", flush=True)
            # ------------------------------------------------------------------
            # 3. SBERT rerank on the surviving drafts
            # ------------------------------------------------------------------
            
            _embed = lambda txts: self.embedder.encode(txts, convert_to_tensor=True)

            prompt_emb = _embed([user_prompt])
            emb        = _embed(valid_strs)           # (M,384)

            lm_scores  = torch.tensor(lm_rewards, device=DEVICE)   # (M,)
            cos        = util.cos_sim(emb, prompt_emb).squeeze(1)  # (M,)

            λ = self.lam
            final_score = λ * lm_scores + (1 - λ) * cos
            best_idx = int(torch.argmax(final_score))

            best_text = valid_strs[best_idx]
        else:
            print(f"\r 🔀 LLM inference  ", end="", flush=True)
            best_text = self.generate_single(input_ids, bias_vec)
####
        if write_memory:
            self.mem.update(user_prompt, self.mem.enforce_sentence_boundaries(best_text))
        print("\r" + " " * 120, end="\r", flush=True)
        return self.mem.enforce_sentence_boundaries(best_text)
              
#####################################################
#####################################################
#####################################################
                
#####################################################
#####################################################
#####################################################


# ──────────────────────────────────────────────────────────────────
# 1.  TRAINER  – now with  .txt  support
# ──────────────────────────────────────────────────────────────────
class GPT2CustomTrainer:
    def __init__(self, model_path: str = "microsoft/DialoGPT-small") -> None:
        self.tok = GPT2Tokenizer.from_pretrained(model_path)
        self.tok.pad_token = self.tok.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)

    # --- new convenience fine-tune entry ----------------------------------
    def finetune_txt(
        self,
        txt_file: str,
        epochs: int = 1,
        out_dir: str = "./ckpt",
        max_len: int = 1024,
    ) -> None:
        print(f"▸ Loading corpus  {txt_file}")
        with open(txt_file, encoding="utf-8") as fh:
            raw = fh.read()

        # simple rule-based chunking at sentence boundaries
        segments, buf = [], []
        for sent in re.split(r'(?<=[.!?])\s+', raw):
            ids = self.tok.encode(sent, add_special_tokens=False)
            if len(buf) + len(ids) > max_len:
                segments.append(buf);  buf = []
            buf.extend(ids)
        if buf: segments.append(buf)

        print(f"▸ Prepared {len(segments)} segments.")
        def to_dataset():
            for seg in segments:
                pad = [self.tok.pad_token_id]*(max_len - len(seg))
                ids = torch.tensor(seg + pad, dtype=torch.long)
                att = (ids != self.tok.pad_token_id).long()
                yield {"input_ids": ids, "attention_mask": att, "labels": ids}

        dataset = list(to_dataset())

        args = TrainingArguments(
            output_dir   = out_dir,
            overwrite_output_dir=True,
            num_train_epochs = epochs,
            per_device_train_batch_size = 1,
            save_strategy = "epoch",
            logging_steps = 50,
            report_to = "none",
            fp16 = torch.cuda.is_available()
        )
        print("▸ Starting fine-tune …")
        Trainer(
            self.model, args,
            train_dataset = dataset,
            data_collator = default_data_collator
        ).train()
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        print("✅ Training done; model saved to", out_dir)

    # --- util to auto-load latest ckpt for chat ---------------------------
    def maybe_load_latest(self, ckpt_root: str = "./ckpt") -> None:
        paths = glob.glob(os.path.join(ckpt_root, "checkpoint-*"))
        if not paths:  # nothing yet
            return
        
        
        latest = max(paths, key=lambda p: int(p.split('-')[-1]))
        
        
        print(" 🐌 Loading finetuned model weights →", latest)
        self.model = GPT2LMHeadModel.from_pretrained(
            latest,
            torch_dtype="auto",
        ).to(DEVICE)

        self.tok = GPT2Tokenizer.from_pretrained( "microsoft/DialoGPT-small")
        self.tok.pad_token = self.tok.eos_token

