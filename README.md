# Rikuy!

Rikuy! is a Windows-first PyQt6 webcam viewer built on OpenCV. It scans for
available cameras, restores the last selected camera and resolution, and
provides a manual device refresh control.

Audio capture is intentionally disabled while camera startup reliability is
being stabilized.

## Requirements

- Windows
- Python 3 with `pip`
- A webcam available to Windows

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, either adjust the
execution policy for the current process or run the environment's Python
directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Launch

With the virtual environment activated:

```powershell
python rikuy.py
```

You can also run:

```powershell
.\run_rikuy.bat
```

The batch launcher first tries `python`, then falls back to the Windows
`py -3` launcher.

## Configuration

Rikuy reads `config.ini` beside `rikuy.py`. The file is local-only and ignored
by Git. The app saves the selected camera and resolution there when it exits.

To start from the provided defaults:

```powershell
Copy-Item config.example.ini config.ini
```

The main video settings are:

- `last_video_id`: preferred OpenCV camera index; use `-1` to fall back to an
  available camera.
- `last_resolution_w` and `last_resolution_h`: preferred capture resolution.

Audio-related keys remain in the example for compatibility, but they have no
effect while audio is disabled.

To use another configuration file, set `RIKUY_CONFIG` before launch. Relative
paths are resolved from the repository root:

```powershell
$env:RIKUY_CONFIG = "config.testing.ini"
python rikuy.py
```

## Troubleshooting

### No camera is found

- Close applications that may have exclusive access to the webcam.
- Confirm camera access is enabled under Windows privacy settings.
- Disconnect and reconnect external cameras, then click **Refresh Devices**.
- Run Rikuy from a terminal and review the camera scan messages.

### The wrong camera or resolution opens

- Select the desired device and resolution in the app, then close it normally
  so the settings are saved.
- Delete or rename `config.ini` to return to automatic selection.
- Set `last_video_id` in `config.ini` if the camera index is stable.

### The preview is black or startup fails

- Wait briefly while the camera warms up.
- Close other camera software and relaunch Rikuy.
- Update the webcam driver through Windows.
- Remove and recreate `.venv`, then reinstall `requirements.txt` if imports or
  Qt platform plugins fail.

### Audio controls are unavailable

This is expected. Audio is intentionally disabled in the current application
behavior.
