# 🪪 CHANGELOG.md  
_All notable changes to this project will be documented here._

---

## 🚧 [Unreleased] → `Sapphire β`

<img width="300" alt="sapphire_b" src="https://github.com/user-attachments/assets/89d47a0f-a2df-477c-85ee-c56cc33cb07b" />


### 📌 Planned  
- **Full codebase sweep:** cleanup of legacy bugs, re-synthesis of stability loops, and comprehensive testing.  
> 🔖 *Branch:* `Sapphire β`
> 🗓️ *ETA:* **2025-08-10**

---

## Coming In Sapphire β

### Hyperparameter Dashboard

<img width="400" alt="image" src="https://github.com/user-attachments/assets/8a51727e-ae97-496e-97dd-0359f57d8c96" />

### Prompt Waveform Viewer

<img width="3827" height="1158" alt="image" src="https://github.com/user-attachments/assets/1b97b443-d897-4988-976a-6c5009ad662f" />

---
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
