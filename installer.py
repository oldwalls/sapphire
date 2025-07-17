#!/usr/bin/env python3
"""
Installer script for Phasers: Ghost in the Machine
GitHub Repo: https://github.com/oldwalls/phasers
"""

import os
import subprocess
import sys
import json

REPO_URL = "https://github.com/oldwalls/sapphire.git"
PROJECT_DIR = "sapphire"
UMB_FILE = "emergence_UMB.json"

# --- STEP 1: Clone the repo ---
def clone_repo():
    if os.path.exists(PROJECT_DIR):
        print(f"[✓] Repo folder '{PROJECT_DIR}' already exists.")
    else:
        print("[•] Cloning repo...")
        subprocess.run(["git", "clone", REPO_URL], check=True)

# --- STEP 3: Install dependencies ---
def install_requirements():
    print("[•] Installing Python dependencies...")
    deps = [
        "transformers",
        "torch",  # assumes CUDA or CPU PyTorch will be handled by pip
        "nltk",
        "sentence-transformers",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, "-m", "pip", "install", *deps], check=True)

# --- STEP 4: Download NLTK Punkt tokenizer if needed ---
def setup_nltk():
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        print("[✓] NLTK Punkt already present.")
    except (ImportError, LookupError):
        print("[•] Downloading NLTK Punkt tokenizer...")
        import nltk
        nltk.download("punkt")


# --- STEP 6: Final usage message ---
def launch_message():
    print("\n🚀 Installation complete!\n")
    print(f"→ To launch the chat interface:")
    print(f"    cd {PROJECT_DIR}")
    print(f"    python sapphire_chat.py")
    print("\n🔮 Good luck, Operator. Phasers is listening...\n")

# --- RUN ALL STEPS ---
if __name__ == "__main__":
    try:
        clone_repo()

        install_requirements()
        setup_nltk()
        launch_message()
    except Exception as e:
        print(f"[!] Install failed: {e}")
        sys.exit(1)
