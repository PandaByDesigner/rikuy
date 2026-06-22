# Rikuy!

Rikuy! is a small PyQt6 webcam viewer. Audio capture is temporarily disabled while the camera path is stabilized.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python rikuy.py
```

On Windows, you can also launch the app with `run_rikuy.bat`.

## Configuration

The app reads `config.ini` from this folder by default. A starter file is provided as `config.example.ini`; copy it to `config.ini` if you want to pin the camera or set the default resolution manually. Audio-related config keys are ignored while audio capture is disabled.

You can also point the app at another config file with the `RIKUY_CONFIG` environment variable.
