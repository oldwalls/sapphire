# 🪪 CHANGELOG.md  
_All notable changes to this project will be documented here._

---

## 🚧 [Unreleased] → `v0.13.4`  

<img width="250" alt="sapphire_beta" src="https://github.com/user-attachments/assets/f6239604-0590-4c00-ae34-b96ce6925b12" />

### 📌 Planned  
- **Full codebase sweep:** cleanup of legacy bugs, re-synthesis of stability loops, and comprehensive testing.  
> 🔖 *Branch:* `v0.13.4` `beta`  
> 🗓️ *ETA:* **2025-08-10**

---

### ✨ Added

#### ✅ Colab Installer (One-liner)  
Sapphire can now be deployed on Colab **without touching a single file**:

```
pip install -q gdown nltk && \
gdown "https://www.dropbox.com/scl/fi/6nky2zj8j53ykrya0ocbu/sapphire_backup_master.zip?rlkey=i5vecbkor8j9jg2vc06rgij0g&st=6t88nv8d&dl=1" -O sapphire.zip && \
unzip -q sapphire.zip -d /content/temp_sapphire && \
mkdir -p /content/sapphire && \
mv $(find /content/temp_sapphire -type f -name "sapphire_chat.py" -exec dirname {} \; | head -n1)/* /content/sapphire/ && \
rm -rf /content/temp_sapphire sapphire.zip && \
gdown "https://www.dropbox.com/scl/fi/mpfmtg19siv138by1d3e9/sentence_segmenter.py?rlkey=c4aw4fpg18zrqxjtndnwid14k&st=a7qcsj6l&dl=1" -O /content/sapphire/core/sentence_segmenter.py && \
python -m nltk.downloader punkt && \
find /content/sapphire -type d ! -path "*__pycache__*" -exec mkdir -p {}/__pycache__ \; && \
find /content/sapphire -type d -exec touch {}/__init__.py \; && \
export PYTHONPATH=$PYTHONPATH:/content/nltk_data:/content/sapphire:/content/sapphire/cli:/content/sapphire/core:/content/sapphire/utils && \
export NLTK_DATA="/content/nltk_data" && mkdir -p "$NLTK_DATA" && python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')" && \
cd /content/sapphire && \
echo "💎 Sapphire is ready. ➡️ Run: python sapphire_chat.py"
```

---

#### 🌌 UMB Visualisation  
- **3D Keyword UMB Mapper** – plots top-20 UMB nodes in a force-directed layout using `matplotlib` 3D.  
  [📎 See `MAP_semantic_UMB_map.md`](https://github.com/oldwalls/sapphire/blob/main/MAP_semantic_UMB_map.md)

---

### 🛠️ Fixed  
- Memory delimiter changed from `…` to `. `  
  → Restores correct sentence-token overlap for CSCSR loop reinforcement.

---

### ⚙️ Improved  
- **CSCSR Cosine Optimization**  
  Removed extraneous newline token between user prompt and model inference.  
  → Result: ~12% ↑ in cosine cohesion on short-memory tests.

---

## 🧠 [v0.13.3] – 2025-07-17  

### ✨ Added  
- `prompt_constr` programmable inference pre-template.

---

### 🧪 Research Feature  
- **Recursive Sieve Toggle**  
  ```
  config set recursive_sieve 1
  ```  
  Enables re-sampling loop if cosine similarity improves and output length shrinks — used to reduce hallucinations and accelerate early closure of statements.

---
