# Installation

> **Toate comenzile se rulează din terminalul VSCode** (Ctrl+`)

## 1. Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install PyTorch + TorchAudio (CPU)
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 3. Install OpenWakeWord
```bash
pip install openwakeword
```

## 4. Install Ollama + Model
```bash
sudo snap install ollama
ollama pull qwen2.5:3b
```

## 5. Install Piper TTS
```bash
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

## 6. Install num2words (for number-to-word conversion in TTS)
```bash
pip install num2words
```

## 7. Install System Audio Tools
```bash
sudo apt install pavucontrol pulseaudio portaudio19-dev
```

## 8. Run
```bash
source .venv/bin/activate
LOG_LEVEL=INFO python -m src.app
python -m src.server.api --host 0.0.0.0 --port 8001
```
