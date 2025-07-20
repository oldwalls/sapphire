#!/bin/bash

# Sapphire Installer – Recursive auto-flatten with gdown
# Remy & Sapphire

# === SETUP ===
FILE_ID="1VEpG3GtfAn39TJcfhTi9kFdNV05cDlrU"
ZIP_NAME="sapphire_master.zip"
TARGET_DIR="/content/sapphire"

# === CLEANUP ===
rm -rf "$TARGET_DIR" /content/temp_unpack "$ZIP_NAME"
mkdir -p "$TARGET_DIR"

# === INSTALL gdown ===
pip install -q gdown

# === DOWNLOAD ===
echo "⬇️  Downloading ZIP from Google Drive..."
gdown --id "$FILE_ID" --output "$ZIP_NAME"

# === UNZIP TO TEMP ===
echo "🧩 Unzipping to temp..."
mkdir -p /content/temp_unpack
unzip -q "$ZIP_NAME" -d /content/temp_unpack

# === FIND DEEPEST FOLDER THAT CONTAINS sapphire_chat.py ===
DEEPEST=$(find /content/temp_unpack -type f -name "sapphire_chat.py" | head -n 1)
DEEPEST_DIR=$(dirname "$DEEPEST")

# === COPY CLEAN ===
echo "📂 Copying from: $DEEPEST_DIR"
cp -r "$DEEPEST_DIR"/* "$TARGET_DIR/"

# === CLEANUP ===
rm -rf /content/temp_unpack "$ZIP_NAME"

pip install -q sentence-transformers
echo "✅  Sapphire installed in $TARGET_DIR"
cd sapphire
python sapphire_chat.py

