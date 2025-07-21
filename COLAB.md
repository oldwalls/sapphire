## 🚀 Sapphire — Colab Quick Start

colab Sapphire chatbot installation one line command.

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

### How to use this

1. Paste the bash one line command into terminal on any new open colab session. You must be in the `/content` directory.
2. Wait for the bot to spawn, which will finish when you see this line:  `💎 Sapphire is ready. ➡️  Run: python sapphire_chat.py`
3. When the chatbot starts with command line type: `config load prospect` - this will load the best inference preset.


### 📸 colab screenshot

<img width="3840" height="1753" alt="image" src="https://github.com/user-attachments/assets/532cbef0-98e4-4337-a35f-230557395347" />



---

