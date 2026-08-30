# Rikuy!

Rikuy! is a Windows-first PyQt6 webcam viewer built on OpenCV, with native
Linux Video4Linux2 support. It scans for available cameras, restores the last
selected camera and resolution, and presents a responsive live-preview layout
with clear camera controls.

The camera status badge reports startup, the active backend and negotiated
resolution, live video, interrupted signals, and errors. Audio capture remains
intentionally disabled while camera startup reliability is being stabilized.

## Controls

| Control | Shortcut | Behavior |
| --- | --- | --- |
| **Mirror** | `Alt+M` | Mirrors the displayed preview without changing camera capture. |
| **Save Frame** | `Ctrl+S` | Opens an explicit file dialog and saves the visible frame as a PNG. |
| **Fullscreen** | `F11` | Toggles fullscreen; press `Esc` to return to windowed mode. |
| **Refresh Cameras** | `F5` | Scans for connected cameras again while preserving selections when possible. |

The preview extends to the bottom of the window and scales with it. Camera,
resolution, and action controls live in a responsive top bar that wraps only
when the window is narrow, and remains available in fullscreen use.

## Requirements

- Windows or Linux
- Python 3 with `pip`
- A webcam available to the operating system

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

The batch launcher prefers `.venv\Scripts\python.exe`, then tries `python` and
the Windows `py -3` launcher. It pauses only when no usable interpreter is
available or Rikuy exits with an error.

### Linux

On Arch Linux, install the native runtime packages and then run the user-local
installer:

```bash
sudo pacman -S --needed python-opencv python-numpy python-pyqt6 imagemagick
./install_linux.sh
```

Launch **Rikuy** from the app menu or run `rikuy`. The Linux launcher stores
settings under `${XDG_CONFIG_HOME:-$HOME/.config}/rikuy/config.ini` and the
application under `${XDG_DATA_HOME:-$HOME/.local/share}/rikuy`.

For a repository-local launch, run `./run_rikuy.sh`. Maintainers can run
`./run_rikuy.sh --smoke-test` to open the real interface and close it
automatically after a short startup check.

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

## Tests

The standard-library test suite uses fakes and does not require a webcam:

```powershell
python -m unittest discover -s tests -v
```

Automated tests cannot reproduce every driver or device behavior. Real Windows
camera validation still matters for DirectShow and Media Foundation fallback,
startup and refresh, camera or resolution switching, unplug/replug behavior,
and clean shutdown.

## Troubleshooting

### No camera is found

- Close applications that may have exclusive access to the webcam.
- Confirm camera access is enabled under Windows privacy settings.
- On Linux, confirm the camera appears as `/dev/video*` and that the desktop
  session grants access to it.
- Disconnect and reconnect external cameras, then choose **Refresh Cameras**
  or press `F5`.
- Run Rikuy from a terminal and review the camera scan messages.

### The wrong camera or resolution opens

- Select the desired device and resolution in the app, then close it normally
  so the settings are saved.
- Delete or rename `config.ini` to return to automatic selection.
- Set `last_video_id` in `config.ini` if the camera index is stable.

### The preview is black or startup fails

- Wait briefly while the camera warms up.
- Check the camera status badge for the active backend, resolution, or recovery
  message.
- Close other camera software and relaunch Rikuy.
- Update the webcam driver through Windows, or check the V4L2 device and input
  signal on Linux.
- Remove and recreate `.venv`, then reinstall `requirements.txt` if imports or
  Qt platform plugins fail.

### Audio controls are unavailable

This is expected. Audio is intentionally disabled in the current application
behavior.
