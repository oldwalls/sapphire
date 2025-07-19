

# Model Download

To use the model with the correct checkpoint, please download the model files from the following link:

🔗 [Download Model Checkpoint](https://drive.google.com/drive/folders/1ib9pQtbJ2p2gxnAxc2d9EtlaKGCUpFJI?usp=sharing)

## Installation Instructions

1. Download all files from the link above.
2. Create the following directory if it does not exist:

```
./ckpt/checkpoint-930/
```

3. Place **all downloaded model files** into the `./ckpt/checkpoint-930/` directory.

Your folder structure should look like this:

```
phasers/
├── sapphire_chat.py
├── sapphire_core.py
├── ...
└── ckpt/
    └── checkpoint-930/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer_config.json
```

After placing the files correctly, you can run the Sapphire system using:

```
python sapphire_chat.py
```

Make sure all dependencies are installed.

## Model specifics
* base model: "microsoft/DialoGPT-small"
* Layer 1: "zen and the art of motorcycle maintenance" - 15 epochs at 5e-5 learning rate
* Layer 2: "Tao Te Ching" - 7 epochs at 5e-5 learning rate

> the overlays wake up the base corpus and make it emergent.
