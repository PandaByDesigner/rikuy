"""
Rikuy! Webcam Viewer
--------------------
A PyQt6-based application for viewing webcam video.
"""

# --- Standard Library Imports ---
import os
import sys
import time
import logging
import configparser
import tempfile
import glob

# --- Third-Party Imports ---
import cv2
import numpy as np
try:
    import pyaudio
except ImportError:
    pyaudio = None
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QComboBox, QProgressBar, QPushButton, QSizePolicy,
    QMessageBox, QFrame, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QKeySequence
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QStandardPaths

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_CONFIG_FILE = 'config.ini'
DEFAULT_STYLE_FILE = 'style.qss'
CONFIG_ENV_VAR = 'RIKUY_CONFIG'
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DEVICE_SCAN_LIMIT = 10
AUDIO_FEATURE_ENABLED = False

CHUNK = 1024
FORMAT = pyaudio.paInt16 if pyaudio else None
CHANNELS = 1
RATE = 44100
AUDIO_RATE_FALLBACKS = [48000, 44100, 32000, 22050, 16000]
VIDEO_FRAME_RATE = 30
VIDEO_WARMUP_FRAMES = 6
VISIBLE_FRAME_MEAN_THRESHOLD = 3.0
VIDEO_SCAN_MISSES_AFTER_LAST_DEVICE = 2
VIDEO_READ_FAILURE_LIMIT = 30
VIDEO_READ_WARNING_INTERVAL = 5
VIDEO_THREAD_STOP_TIMEOUT_MS = 3000
SMOKE_TEST_DURATION_MS = 4000
TOP_BAR_NARROW_WIDTH = 840
TOP_BAR_WIDE_WIDTH = 1450
BLACK_FRAME_STATUS_MESSAGE = (
    'Camera connected, but the image is black.\n'
    'Check the privacy shutter, lens, cable, or input signal.'
)
DEFAULT_RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]
MAX_AUDIO_VALUE = np.iinfo(np.int16).max


# --- Helper Functions ---
def get_config_file_path():
    """Return the config path, allowing override via environment variable."""
    configured_path = os.environ.get(CONFIG_ENV_VAR)
    if configured_path:
        if os.path.isabs(configured_path):
            return configured_path
        return os.path.join(APP_DIR, configured_path)
    return os.path.join(APP_DIR, DEFAULT_CONFIG_FILE)


def get_app_file_path(filename):
    """Build an absolute path for a file stored alongside the application."""
    return os.path.join(APP_DIR, filename)


def get_video_capture_backend():
    """Pick a backend that works well on the current platform."""
    return get_video_capture_backends()[0][1]


def get_video_capture_backends():
    """Return capture backends in preference order for the current platform."""
    if sys.platform.startswith('win'):
        return [
            ('DirectShow', cv2.CAP_DSHOW),
            ('Media Foundation', cv2.CAP_MSMF),
        ]
    if sys.platform.startswith('linux'):
        return [('Video4Linux2', cv2.CAP_V4L2)]
    return [('Default', cv2.CAP_ANY)]


def linux_video_device_supports_capture(device_path):
    """Return True for V4L2 video-capture nodes, excluding metadata nodes."""
    import fcntl
    import struct

    vidioc_querycap = 0x80685600
    v4l2_cap_video_capture = 0x00000001
    v4l2_cap_video_capture_mplane = 0x00001000
    v4l2_cap_device_caps = 0x80000000
    capability_buffer = bytearray(104)

    try:
        device_fd = os.open(device_path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as error:
        logging.warning('Could not inspect %s: %s', device_path, error)
        return True

    try:
        fcntl.ioctl(device_fd, vidioc_querycap, capability_buffer, True)
    except OSError as error:
        logging.warning('Could not query V4L2 capabilities for %s: %s', device_path, error)
        return True
    finally:
        os.close(device_fd)

    _, capabilities, device_capabilities = struct.unpack_from('=III', capability_buffer, 80)
    effective_capabilities = (
        device_capabilities
        if capabilities & v4l2_cap_device_caps
        else capabilities
    )
    capture_flags = v4l2_cap_video_capture | v4l2_cap_video_capture_mplane
    return bool(effective_capabilities & capture_flags)


def get_video_device_indices():
    """Return camera indices that are worth probing on the current platform."""
    if sys.platform.startswith('linux'):
        indices = []
        for device_path in glob.glob('/dev/video*'):
            suffix = os.path.basename(device_path).removeprefix('video')
            if suffix.isdigit() and linux_video_device_supports_capture(device_path):
                indices.append(int(suffix))
        return sorted(set(indices))
    return list(range(VIDEO_DEVICE_SCAN_LIMIT))


def get_video_device_name(index):
    """Return a useful platform-native camera name for a capture index."""
    if sys.platform.startswith('linux'):
        name_path = f'/sys/class/video4linux/video{index}/name'
        try:
            with open(name_path, 'r', encoding='utf-8') as name_file:
                device_name = name_file.read().strip()
            if device_name:
                return f'{device_name} (/dev/video{index})'
        except OSError:
            pass
    return f'Camera {index}'


def frame_looks_visible(frame):
    """Return True when a captured frame is not effectively black/empty."""
    if frame is None or frame.size == 0:
        return False
    return float(frame.mean()) > VISIBLE_FRAME_MEAN_THRESHOLD


def normalize_audio_device_name(name):
    """Normalize a device name for case-insensitive matching."""
    return str(name or '').replace(' (Default)', '').strip().casefold()


def is_audio_device_locked(settings):
    """Return True when config pins the app to a specific microphone."""
    return settings.get('locked_audio_id', -1) != -1 or bool(settings.get('locked_audio_name', '').strip())


def describe_locked_audio(settings):
    """Build a readable description of the configured audio lock."""
    details = []
    locked_audio_id = settings.get('locked_audio_id', -1)
    locked_audio_name = settings.get('locked_audio_name', '').strip()
    if locked_audio_id != -1:
        details.append(f'ID {locked_audio_id}')
    if locked_audio_name:
        details.append(f'"{locked_audio_name}"')
    return ' / '.join(details) if details else 'configured microphone'


def is_output_monitoring_enabled(settings):
    """Return True when live mic playback is enabled."""
    return settings.get('monitor_audio', False)


def is_audio_output_locked(settings):
    """Return True when config pins playback to a specific output device."""
    return settings.get('locked_output_id', -1) != -1 or bool(settings.get('locked_output_name', '').strip())


def describe_locked_output(settings):
    """Build a readable description of the configured output lock."""
    details = []
    locked_output_id = settings.get('locked_output_id', -1)
    locked_output_name = settings.get('locked_output_name', '').strip()
    if locked_output_id != -1:
        details.append(f'ID {locked_output_id}')
    if locked_output_name:
        details.append(f'"{locked_output_name}"')
    return ' / '.join(details) if details else 'configured speakers'


def format_resolution(width, height):
    """Build the resolution label shown in the combo box."""
    return f"{width}x{height}"


def parse_resolution_text(text):
    """Parse a resolution label like '1280x720' into a tuple."""
    if not isinstance(text, str) or not text or "x" not in text:
        return None
    try:
        width_text, height_text = text.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except ValueError:
        return None


def build_audio_open_attempts(device_info):
    """Build channel/rate attempts for a given audio input device."""
    max_input_channels = max(1, int(device_info.get('maxInputChannels', 1)))
    default_rate = int(device_info.get('defaultSampleRate', RATE) or RATE)

    candidate_channels = [1]
    if max_input_channels > 1:
        candidate_channels.append(min(max_input_channels, 2))
    if max_input_channels not in candidate_channels:
        candidate_channels.append(max_input_channels)

    candidate_rates = [default_rate]
    for rate in AUDIO_RATE_FALLBACKS:
        if rate not in candidate_rates:
            candidate_rates.append(rate)

    attempts = []
    for channels in candidate_channels:
        for rate in candidate_rates:
            attempts.append((channels, rate))
    return attempts


def probe_audio_input_device(p, device_info):
    """Probe whether an audio input device supports any format we can use."""
    device_index = int(device_info.get('index', -1))
    last_error = None
    for channels, rate in build_audio_open_attempts(device_info):
        try:
            p.is_format_supported(
                rate,
                input_device=device_index,
                input_channels=channels,
                input_format=FORMAT,
            )
            return True, channels, rate, None
        except Exception as e:
            last_error = e
    return False, None, None, last_error


def read_config_value(config, option, fallback, getter):
    """Read one setting without allowing a malformed value to discard others."""
    try:
        return getattr(config, getter)('Settings', option, fallback=fallback)
    except (ValueError, configparser.Error) as error:
        logging.warning(
            "Invalid config value for %s=%r; using %r (%s)",
            option,
            config.get('Settings', option, raw=True, fallback=None),
            fallback,
            error,
        )
        return fallback


def load_settings():
    """
    Load settings from the config file.
    Returns a dict with keys for device ids, resolution, and optional audio lock.
    """
    config_path = get_config_file_path()
    config = configparser.ConfigParser()
    settings = {
        'video_id': -1,
        'audio_id': -1,
        'resolution_w': -1,
        'resolution_h': -1,
        'locked_audio_id': -1,
        'locked_audio_name': '',
        'monitor_audio': False,
        'locked_output_id': -1,
        'locked_output_name': '',
    }
    try:
        if config.read(config_path, encoding='utf-8'):
            if 'Settings' in config:
                settings['video_id'] = read_config_value(config, 'last_video_id', -1, 'getint')
                settings['audio_id'] = read_config_value(config, 'last_audio_id', -1, 'getint')
                settings['resolution_w'] = read_config_value(config, 'last_resolution_w', -1, 'getint')
                settings['resolution_h'] = read_config_value(config, 'last_resolution_h', -1, 'getint')
                settings['locked_audio_id'] = read_config_value(config, 'locked_audio_id', -1, 'getint')
                settings['locked_audio_name'] = read_config_value(config, 'locked_audio_name', '', 'get').strip()
                settings['monitor_audio'] = read_config_value(config, 'monitor_audio', False, 'getboolean')
                settings['locked_output_id'] = read_config_value(config, 'locked_output_id', -1, 'getint')
                settings['locked_output_name'] = read_config_value(config, 'locked_output_name', '', 'get').strip()
            logging.info("Loaded settings from %s: %s", config_path, settings)
        else:
            logging.info("%s not found, using default settings.", config_path)
    except Exception as e:
        logging.error("Error loading settings from %s: %s", config_path, e, exc_info=True)
    return settings


def save_settings(
    video_id,
    audio_id,
    resolution,
    locked_audio_id=-1,
    locked_audio_name='',
    monitor_audio=False,
    locked_output_id=-1,
    locked_output_name='',
):
    """
    Save the provided settings to the config file.
    Args:
        video_id (int): Last used video device ID
        audio_id (int): Last used audio device ID
        resolution (tuple): (width, height) or None
        locked_audio_id (int): Pinned audio device id, or -1 to unlock
        locked_audio_name (str): Optional case-insensitive name match for the pinned device
        monitor_audio (bool): Enable live mic playback
        locked_output_id (int): Pinned output device id, or -1 to unlock
        locked_output_name (str): Optional case-insensitive name match for the playback device
    """
    config_path = get_config_file_path()
    config = configparser.ConfigParser()
    config['Settings'] = {
        'last_video_id': str(video_id),
        'last_audio_id': str(audio_id),
        'last_resolution_w': str(resolution[0] if resolution else -1),
        'last_resolution_h': str(resolution[1] if resolution else -1),
        'locked_audio_id': str(locked_audio_id),
        'locked_audio_name': locked_audio_name.strip(),
        'monitor_audio': str(bool(monitor_audio)).lower(),
        'locked_output_id': str(locked_output_id),
        'locked_output_name': locked_output_name.strip(),
    }
    temporary_path = None
    try:
        config_directory = os.path.dirname(os.path.abspath(config_path))
        config_filename = os.path.basename(config_path)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f'.{config_filename}.',
            suffix='.tmp',
            dir=config_directory,
            text=True,
        )
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as configfile:
            config.write(configfile)
        os.replace(temporary_path, config_path)
        logging.info(
            'Saved settings to %s: Video=%s, Audio=%s, Res=%s, '
            'LockedAudioId=%s, LockedAudioName=%s, MonitorAudio=%s, '
            'LockedOutputId=%s, LockedOutputName=%s',
            config_path,
            video_id,
            audio_id,
            resolution,
            locked_audio_id,
            locked_audio_name,
            monitor_audio,
            locked_output_id,
            locked_output_name,
        )
        return True
    except Exception as e:
        logging.error(f"Error saving settings to {config_path}: {e}", exc_info=True)
        return False
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                logging.warning("Could not remove temporary config file %s", temporary_path)


def list_video_devices(show_error=False, parent=None):
    """
    List available video capture devices using OpenCV.
    Returns:
        List[dict]: List of device dicts with 'id' and 'name'.
    """
    devices = []

    consecutive_misses = 0
    for index in get_video_device_indices():
        found_device = False
        for backend_name, backend in get_video_capture_backends():
            cap_test = None
            try:
                cap_test = cv2.VideoCapture(index, backend)
                if cap_test.isOpened():
                    ret, frame = cap_test.read()
                    if ret and frame is not None and frame.size > 0:
                        try:
                            width = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            name = f"{get_video_device_name(index)} ({width}x{height})"
                        except Exception:
                            logging.warning(f"Could not get properties for Camera {index}", exc_info=False)
                            name = get_video_device_name(index)
                        logging.info("Found video device %s using %s", index, backend_name)
                        devices.append({'id': index, 'name': name})
                        found_device = True
                        break
            except Exception as error:
                logging.warning(
                    "Could not probe camera %s with %s: %s",
                    index,
                    backend_name,
                    error,
                )
            finally:
                if cap_test is not None:
                    try:
                        cap_test.release()
                    except Exception as error:
                        logging.warning(
                            "Could not release camera %s probe for %s: %s",
                            index,
                            backend_name,
                            error,
                        )

        if found_device:
            consecutive_misses = 0
        elif devices and not sys.platform.startswith('linux'):
            consecutive_misses += 1
            if consecutive_misses >= VIDEO_SCAN_MISSES_AFTER_LAST_DEVICE:
                break

    if not devices:
        logging.warning('No video devices found.')
        if show_error and not sys.platform.startswith('linux'):
            QMessageBox.warning(parent, 'No Cameras Found', 'No video devices were found on this system.')
        devices.append({'id': -1, 'name': 'No Video Devices Found'})

    return devices


def list_audio_devices(p, settings=None):
    """
    List available audio input devices using PyAudio.
    Args:
        p (pyaudio.PyAudio): PyAudio instance
        settings (dict | None): Unused; kept for call compatibility
    Returns:
        List[dict]: List of device dicts with 'id' and 'name'.
    """
    if not AUDIO_FEATURE_ENABLED or p is None or pyaudio is None:
        return [{'id': -1, 'name': 'Audio Disabled', 'available': False}]

    devices = []
    try:
        default_input_idx = -1
        try:
            default_input_idx = p.get_default_input_device_info()['index']
        except (IOError, OSError):
            pass

        for index in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(index)
            if device_info.get('maxInputChannels', 0) > 0:
                device_index = int(device_info.get('index', index))
                name = device_info.get('name', f'Microphone {device_index}')
                if device_index == default_input_idx:
                    name += ' (Default)'
                available, preferred_channels, preferred_rate, last_error = probe_audio_input_device(p, device_info)
                if not available:
                    logging.warning(
                        "Audio device %s (%s) could not be probed successfully: %s",
                        device_index,
                        name,
                        last_error,
                    )
                devices.append({
                    'id': device_index,
                    'name': name,
                    'available': available,
                    'preferred_channels': preferred_channels,
                    'preferred_rate': preferred_rate,
                })
    except Exception as e:
        logging.error(f"Error listing audio devices: {e}", exc_info=True)

    if not devices:
        logging.warning('No audio input devices found.')
        QMessageBox.critical(None, 'No Microphones Found', 'No audio input devices were found on this system.')
        return [{'id': -1, 'name': 'No Audio Input Devices Found'}]

    return devices


def list_output_devices(p):
    """List available audio output devices using PyAudio."""
    if not AUDIO_FEATURE_ENABLED or p is None or pyaudio is None:
        return []

    devices = []
    try:
        default_output_idx = -1
        try:
            default_output_idx = p.get_default_output_device_info()['index']
        except (IOError, OSError):
            pass

        for index in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(index)
            if device_info.get('maxOutputChannels', 0) > 0:
                device_index = int(device_info.get('index', index))
                name = device_info.get('name', f'Speakers {device_index}')
                if device_index == default_output_idx:
                    name += ' (Default)'
                devices.append({'id': device_index, 'name': name})
    except Exception as e:
        logging.error(f"Error listing audio output devices: {e}", exc_info=True)
    return devices


def resolve_output_device(settings, devices):
    """Resolve the output device that should receive live mic playback."""
    if not is_output_monitoring_enabled(settings):
        return None

    if not is_audio_output_locked(settings):
        return None

    locked_output_id = settings.get('locked_output_id', -1)
    locked_output_name = settings.get('locked_output_name', '').strip().casefold()
    for device in devices:
        name_match = bool(locked_output_name) and locked_output_name in normalize_audio_device_name(device['name'])
        id_match = locked_output_id != -1 and device['id'] == locked_output_id
        if id_match or name_match:
            return device

    return None


# --- Audio Thread ---
class AudioThread(QThread):
    """Thread for capturing audio input and optionally playing it back live."""
    volume_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, device_index, output_device_index=None, monitor_audio=False, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.output_device_index = output_device_index
        self.monitor_audio = monitor_audio
        self._running = False
        self.p_audio = None
        self.stream = None
        self.output_stream = None
        self.input_channels = CHANNELS
        self.input_rate = RATE
        self.output_channels = CHANNELS

    def get_input_open_attempts(self):
        """Build a list of input channel/rate attempts for the selected device."""
        device_info = self.p_audio.get_device_info_by_index(self.device_index)
        return build_audio_open_attempts(device_info)

    def open_input_stream(self):
        """Open the input stream with device-specific fallbacks."""
        last_error = None
        for channels, rate in self.get_input_open_attempts():
            try:
                self.stream = self.p_audio.open(
                    format=FORMAT,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=self.device_index,
                )
                self.input_channels = channels
                self.input_rate = rate
                logging.info(
                    'Audio input stream opened on device %s with %s channel(s) at %s Hz',
                    self.device_index,
                    channels,
                    rate,
                )
                return
            except Exception as e:
                last_error = e
                logging.warning(
                    'Audio input open failed on device %s with %s channel(s) at %s Hz: %s',
                    self.device_index,
                    channels,
                    rate,
                    e,
                )

        raise last_error if last_error else RuntimeError('No audio input formats could be opened')

    def open_output_stream(self):
        """Open the playback stream with a couple of fallback channel layouts."""
        if not self.monitor_audio or self.output_device_index is None:
            return

        playback_attempts = [self.input_channels]
        if self.input_channels == 1:
            playback_attempts.append(2)
        else:
            playback_attempts.append(1)

        for channels in playback_attempts:
            try:
                self.output_stream = self.p_audio.open(
                    format=FORMAT,
                    channels=channels,
                    rate=self.input_rate,
                    output=True,
                    frames_per_buffer=CHUNK,
                    output_device_index=self.output_device_index,
                )
                self.output_channels = channels
                logging.info(
                    'Audio playback stream opened on device %s with %s channel(s)',
                    self.output_device_index,
                    channels,
                )
                return
            except Exception as e:
                logging.warning(
                    'Audio playback open failed on device %s with %s channel(s): %s',
                    self.output_device_index,
                    channels,
                    e,
                )

        self.monitor_audio = False
        self.output_stream = None
        logging.warning('Live playback disabled because the locked output device could not be opened.')

    def run(self):
        """Start the audio input stream, emit levels, and optionally play back audio."""
        self._running = True
        try:
            self.p_audio = pyaudio.PyAudio()
            self.open_input_stream()
        except Exception as e:
            error_msg = f"Failed to start audio input stream on device {self.device_index}: {e}"
            logging.error(error_msg)
            self.error_signal.emit(error_msg)
            self.cleanup()
            return

        self.open_output_stream()

        try:
            while self._running:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    audio_level = np.abs(audio_np.astype(np.int32)).max() if audio_np.size else 0
                    volume = min(100, int(audio_level / MAX_AUDIO_VALUE * 100))
                    self.volume_signal.emit(volume)
                    if self.output_stream:
                        playback_data = data
                        if self.input_channels != self.output_channels:
                            audio_frames = audio_np.reshape(-1, self.input_channels)
                            if self.input_channels == 1 and self.output_channels == 2:
                                converted_audio = np.repeat(audio_frames, 2, axis=1)
                            else:
                                mono_audio = audio_frames.mean(axis=1, dtype=np.float32)
                                converted_audio = mono_audio.astype(np.int16).reshape(-1, 1)
                            playback_data = converted_audio.astype(np.int16).tobytes()
                        self.output_stream.write(playback_data)
                except Exception as e:
                    if self._running:
                        logging.error(f"Error in audio thread loop: {e}")
                        self.error_signal.emit(f"Audio input stopped unexpectedly: {e}")
                    break
        finally:
            self.cleanup()

    def stop(self):
        """Stop the audio thread and close streams."""
        self._running = False
        self.wait(1500)

    def cleanup(self):
        """Release audio resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None
        if self.p_audio:
            try:
                self.p_audio.terminate()
            except Exception:
                pass
            self.p_audio = None


# --- Video Thread ---
class VideoThread(QThread):
    """
    Thread for capturing video frames from a selected camera and emitting them as QImage.
    Emits:
        - frame_signal(QImage): on new frame
        - error_signal(str): on error
    """
    frame_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    capture_opened_signal = pyqtSignal(str, int, int)

    def __init__(self, device_index, resolution, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.target_resolution = resolution
        self._running = True
        self.video_capture = None

    def stop_requested(self):
        """Return True once shutdown has been requested, including before run()."""
        return not self._running or self.isInterruptionRequested()

    def candidate_resolutions(self):
        """Return requested resolution plus conservative fallbacks."""
        candidates = []
        for resolution in [self.target_resolution, (1280, 720), (640, 480), None]:
            if resolution not in candidates:
                candidates.append(resolution)
        return candidates

    def configure_capture(self, capture, resolution):
        """Apply capture hints without assuming the camera accepts all of them."""
        if sys.platform.startswith('win'):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if resolution:
            target_w, target_h = resolution
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    def read_startup_frame(self, capture):
        """Read several startup frames and prefer the first visible one."""
        last_frame = None
        for _ in range(VIDEO_WARMUP_FRAMES):
            if self.stop_requested():
                break
            ret, frame = capture.read()
            if ret and frame is not None and frame.size > 0:
                last_frame = frame
                if frame_looks_visible(frame):
                    return frame, True
            time.sleep(0.03)
        return last_frame, frame_looks_visible(last_frame)

    def open_capture(self):
        """Open the first camera mode that returns a readable frame."""

        for backend_name, backend in get_video_capture_backends():
            for resolution in self.candidate_resolutions():
                if self.stop_requested():
                    return None, None

                logging.info(
                    "[VideoThread %s] Opening capture with %s at %s",
                    self.device_index,
                    backend_name,
                    format_resolution(*resolution) if resolution else 'default resolution',
                )
                capture = None
                selected = False
                try:
                    capture = cv2.VideoCapture(self.device_index, backend)
                    if not capture.isOpened():
                        continue

                    self.configure_capture(capture, resolution)
                    first_frame, visible = self.read_startup_frame(capture)
                    if self.stop_requested():
                        return None, None

                    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if first_frame is not None:
                        if actual_w <= 0 or actual_h <= 0:
                            actual_h, actual_w = first_frame.shape[:2]
                        logging.info(
                            "[VideoThread %s] Using %s at %sx%s%s",
                            self.device_index,
                            backend_name,
                            actual_w,
                            actual_h,
                            '' if visible else ' (black/no visible signal)',
                        )
                        self.capture_opened_signal.emit(backend_name, actual_w, actual_h)
                        selected = True
                        return capture, first_frame

                    logging.warning(
                        "[VideoThread %s] %s at %sx%s did not return a readable startup frame.",
                        self.device_index,
                        backend_name,
                        actual_w,
                        actual_h,
                    )
                except Exception as error:
                    logging.warning(
                        "[VideoThread %s] %s at %s failed: %s",
                        self.device_index,
                        backend_name,
                        format_resolution(*resolution) if resolution else 'default resolution',
                        error,
                    )
                finally:
                    if capture is not None and not selected:
                        try:
                            capture.release()
                        except Exception as error:
                            logging.warning(
                                "[VideoThread %s] Could not release failed %s capture: %s",
                                self.device_index,
                                backend_name,
                                error,
                            )

        return None, None

    def emit_frame(self, frame):
        """Convert an OpenCV frame to QImage and emit it."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.frame_signal.emit(qt_image.copy())

    def run(self):
        """Start the video capture and emit frames as QImage."""
        try:
            if self.stop_requested():
                logging.info("[VideoThread %s] Startup cancelled.", self.device_index)
                return

            logging.info(f"[VideoThread {self.device_index}] Opening capture...")
            self.video_capture, first_frame = self.open_capture()

            if self.stop_requested():
                return

            if not self.video_capture or not self.video_capture.isOpened():
                error_msg = f"Could not open video device {self.device_index}"
                logging.error(error_msg)
                self.error_signal.emit(error_msg)
                return

            if first_frame is not None:
                if frame_looks_visible(first_frame):
                    self.emit_frame(first_frame)
                else:
                    self.status_signal.emit(BLACK_FRAME_STATUS_MESSAGE)

            waiting_for_visible_frame = not frame_looks_visible(first_frame)
            consecutive_read_failures = 0
            while not self.stop_requested():
                ret, frame = self.video_capture.read()
                if self.stop_requested():
                    break

                frame_is_valid = ret and frame is not None and frame.size > 0
                if frame_is_valid:
                    recovered_after_failures = consecutive_read_failures > 0
                    try:
                        if waiting_for_visible_frame:
                            if frame_looks_visible(frame):
                                waiting_for_visible_frame = False
                                self.emit_frame(frame)
                            elif recovered_after_failures:
                                self.status_signal.emit(BLACK_FRAME_STATUS_MESSAGE)
                        else:
                            self.emit_frame(frame)
                        consecutive_read_failures = 0
                    except Exception as e:
                        consecutive_read_failures += 1
                        logging.error(f"[VideoThread {self.device_index}] Error processing frame: {e}")
                else:
                    consecutive_read_failures += 1

                if consecutive_read_failures:
                    if (
                        consecutive_read_failures == 1
                        or consecutive_read_failures % VIDEO_READ_WARNING_INTERVAL == 0
                    ):
                        logging.warning(
                            "[VideoThread %s] Failed to process frame (%s/%s).",
                            self.device_index,
                            consecutive_read_failures,
                            VIDEO_READ_FAILURE_LIMIT,
                        )
                    if consecutive_read_failures == VIDEO_READ_WARNING_INTERVAL:
                        self.status_signal.emit('Camera signal interrupted. Retrying...')
                    if consecutive_read_failures >= VIDEO_READ_FAILURE_LIMIT:
                        error_msg = (
                            f"Camera {self.device_index} stopped responding after "
                            f"{VIDEO_READ_FAILURE_LIMIT} consecutive frame failures."
                        )
                        logging.error(error_msg)
                        self._running = False
                        self.error_signal.emit(error_msg)
                        break
                    time.sleep(0.1)

                frame_delay = max((1.0 / VIDEO_FRAME_RATE) - 0.005, 0.001)
                time.sleep(frame_delay)
        except Exception as e:
            if not self.stop_requested():
                error_msg = f"Error in VideoThread {self.device_index}: {e}"
                logging.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)
        finally:
            if self.video_capture:
                try:
                    self.video_capture.release()
                    logging.info(f"[VideoThread {self.device_index}] Video capture released.")
                except Exception as error:
                    logging.warning(
                        "[VideoThread %s] Could not release video capture: %s",
                        self.device_index,
                        error,
                    )
                self.video_capture = None
            self._running = False
            logging.info(f"[VideoThread {self.device_index}] Thread finished.")

    def stop(self, timeout_ms=VIDEO_THREAD_STOP_TIMEOUT_MS):
        """Request shutdown and report whether the worker actually stopped."""
        self._running = False
        self.requestInterruption()
        stopped = self.wait(timeout_ms)
        if not stopped:
            logging.error(
                "[VideoThread %s] Did not stop within %sms.",
                self.device_index,
                timeout_ms,
            )
        return stopped


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rikuy! - Webcam Viewer')
        self.resize(1120, 780)
        self.setMinimumSize(640, 440)

        icon_path = get_app_file_path('Rikuy_Condor_Icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            logging.warning(f"Icon file not found at {icon_path}")

        self.load_stylesheet()

        self.audio_thread = None
        self.video_thread = None
        self.current_frame = None
        self.capture_description = ''
        self.video_ready = False
        self.video_session_accepts_signals = False
        self.is_closing = False
        self.audio_enabled = False
        self.p_audio_main = pyaudio.PyAudio() if AUDIO_FEATURE_ENABLED and pyaudio else None

        self.last_settings = load_settings()
        self.audio_device_locked = False
        self.output_monitoring_enabled = AUDIO_FEATURE_ENABLED and is_output_monitoring_enabled(self.last_settings)
        self.available_video_devices = list_video_devices()
        self.available_audio_devices = list_audio_devices(self.p_audio_main, self.last_settings)
        self.available_output_devices = list_output_devices(self.p_audio_main)
        self.output_device = resolve_output_device(self.last_settings, self.available_output_devices)

        self.current_video_device_id = -1
        self.current_audio_device_id = -1

        self.setup_ui()
        self.populate_devices()
        self.populate_resolutions()
        self.apply_initial_settings()
        self.update_audio_lock_ui()

    def setup_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName('appRoot')
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 12)
        main_layout.setSpacing(9)

        self.top_bar = QFrame()
        self.top_bar.setObjectName('topBar')
        self.top_bar_layout = QGridLayout(self.top_bar)
        self.top_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.top_bar_layout.setHorizontalSpacing(10)
        self.top_bar_layout.setVerticalSpacing(6)

        self.title_block = QWidget()
        self.title_block.setObjectName('titleBlock')
        self.title_block.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        title_layout = QVBoxLayout(self.title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        self.app_title = QLabel('Rikuy!')
        self.app_title.setObjectName('appTitle')
        self.app_subtitle = QLabel('A focused, friendly camera preview')
        self.app_subtitle.setObjectName('appSubtitle')
        title_layout.addWidget(self.app_title)
        title_layout.addWidget(self.app_subtitle)

        self.status_label = QLabel('Ready')
        self.status_label.setObjectName('cameraStatus')
        self.status_label.setProperty('state', 'idle')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMaximumWidth(440)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.status_label.setAccessibleName('Camera status')

        self.camera_control = QWidget()
        self.camera_control.setObjectName('cameraControl')
        self.camera_control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        camera_layout = QVBoxLayout(self.camera_control)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(2)
        self.camera_label = QLabel('&Camera')
        self.camera_label.setObjectName('fieldLabel')
        self.video_combo = QComboBox()
        self.video_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.video_combo.setMinimumContentsLength(24)
        self.video_combo.setMinimumWidth(180)
        self.video_combo.setAccessibleName('Camera')
        self.video_combo.setAccessibleDescription('Changing cameras restarts the live preview.')
        self.video_combo.setToolTip('Choose the camera used for the preview (Alt+C).')
        self.video_combo.currentIndexChanged.connect(self.change_video_device)
        self.camera_label.setBuddy(self.video_combo)
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.video_combo)

        self.audio_combo = QComboBox()
        self.audio_combo.currentIndexChanged.connect(self.change_audio_device)
        self.audio_toggle_button = QPushButton('Connect Mic')
        self.audio_toggle_button.clicked.connect(self.toggle_audio)
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setTextVisible(False)
        self.volume_bar.setFixedWidth(150)
        self.audio_controls = QWidget()
        self.audio_controls.setObjectName('audioControls')
        audio_controls_layout = QHBoxLayout(self.audio_controls)
        audio_controls_layout.setContentsMargins(0, 0, 0, 0)
        audio_controls_layout.setSpacing(8)
        if AUDIO_FEATURE_ENABLED:
            audio_controls_layout.addWidget(QLabel('Audio:'))
            audio_controls_layout.addWidget(self.audio_combo)
            self.audio_toggle_button.setText('Disconnect Mic')
            audio_controls_layout.addWidget(self.audio_toggle_button)
            audio_controls_layout.addWidget(QLabel('Mic Level:'))
            audio_controls_layout.addWidget(self.volume_bar)
        else:
            self.audio_combo.setEnabled(False)
            self.audio_toggle_button.setEnabled(False)
            self.audio_controls.hide()

        self.resolution_control = QWidget()
        self.resolution_control.setObjectName('resolutionControl')
        self.resolution_control.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        resolution_layout = QVBoxLayout(self.resolution_control)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(2)
        self.resolution_label = QLabel('&Resolution')
        self.resolution_label.setObjectName('fieldLabel')
        self.resolution_combo = QComboBox()
        self.resolution_combo.setMinimumWidth(110)
        self.resolution_combo.setAccessibleName('Resolution')
        self.resolution_combo.setAccessibleDescription('Changing resolution restarts the live preview.')
        self.resolution_combo.setToolTip('Choose the requested camera resolution (Alt+R).')
        self.resolution_combo.currentIndexChanged.connect(self.change_resolution)
        self.resolution_label.setBuddy(self.resolution_combo)
        resolution_layout.addWidget(self.resolution_label)
        resolution_layout.addWidget(self.resolution_combo)

        self.actions_widget = QWidget()
        self.actions_widget.setObjectName('topBarActions')
        self.actions_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        actions_layout = QHBoxLayout(self.actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        self.mirror_button = QPushButton('&Mirror')
        self.mirror_button.setCheckable(True)
        self.mirror_button.setShortcut(QKeySequence('Alt+M'))
        self.mirror_button.setToolTip('Mirror the preview without changing camera capture (Alt+M).')
        self.mirror_button.toggled.connect(self.toggle_mirror)
        actions_layout.addWidget(self.mirror_button)

        self.snapshot_button = QPushButton('Save &Frame')
        self.snapshot_button.setEnabled(False)
        self.snapshot_button.setShortcut(QKeySequence('Ctrl+S'))
        self.snapshot_button.setToolTip('Save the current visible frame as a PNG (Ctrl+S).')
        self.snapshot_button.clicked.connect(self.save_snapshot)
        actions_layout.addWidget(self.snapshot_button)

        self.fullscreen_button = QPushButton('Fullscreen')
        self.fullscreen_button.setShortcut(QKeySequence('F11'))
        self.fullscreen_button.setToolTip('Toggle full-screen mode (F11; Esc to exit).')
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        actions_layout.addWidget(self.fullscreen_button)

        self.refresh_button = QPushButton('Refresh Cameras')
        self.refresh_button.setObjectName('primaryButton')
        self.refresh_button.setShortcut(QKeySequence('F5'))
        self.refresh_button.setToolTip('Scan for connected cameras again (F5).')
        self.refresh_button.clicked.connect(self.refresh_devices)
        actions_layout.addWidget(self.refresh_button)

        self._top_bar_mode = None
        self.arrange_top_bar(self.top_bar_mode_for_width(self.width()))
        main_layout.addWidget(self.top_bar)

        self.preview_frame = QFrame()
        self.preview_frame.setObjectName('previewFrame')
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel('Initializing...')
        self.video_label.setObjectName('videoPreview')
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setWordWrap(True)
        self.video_label.setAccessibleName('Camera preview')
        preview_layout.addWidget(self.video_label)
        main_layout.addWidget(self.preview_frame, 1)

    def top_bar_mode_for_width(self, width):
        """Choose a top-bar arrangement that keeps every control readable."""
        if width < TOP_BAR_NARROW_WIDTH:
            return 'narrow'
        if width < TOP_BAR_WIDE_WIDTH:
            return 'medium'
        return 'wide'

    def restore_control_shortcuts(self):
        """Reapply explicit shortcuts after button text or mode changes."""
        self.mirror_button.setShortcut(QKeySequence('Alt+M'))
        self.snapshot_button.setShortcut(QKeySequence('Ctrl+S'))
        self.fullscreen_button.setShortcut(QKeySequence('F11'))
        self.refresh_button.setShortcut(QKeySequence('F5'))

    def arrange_top_bar(self, mode):
        """Keep controls in the top bar, wrapping only as width requires."""
        narrow = mode == 'narrow'
        snapshot_text = 'Save &Frame' if narrow else 'Save'
        refresh_text = 'Refresh Cameras' if narrow else 'Refresh'
        if self.snapshot_button.text() != snapshot_text:
            self.snapshot_button.setText(snapshot_text)
        if self.refresh_button.text() != refresh_text:
            self.refresh_button.setText(refresh_text)
        self.restore_control_shortcuts()
        self.app_subtitle.setVisible(mode == 'wide' and self.width() >= 1800)

        self.actions_widget.setMaximumWidth(16777215)
        self.actions_widget.layout().invalidate()
        self.actions_widget.updateGeometry()
        if not narrow:
            self.actions_widget.setMaximumWidth(self.actions_widget.sizeHint().width())

        if self._top_bar_mode == mode:
            return
        self._top_bar_mode = mode

        top_bar_widgets = (
            self.title_block,
            self.camera_control,
            self.resolution_control,
            self.actions_widget,
            self.status_label,
            self.audio_controls,
        )
        for widget in top_bar_widgets:
            self.top_bar_layout.removeWidget(widget)
        for column in range(6):
            self.top_bar_layout.setColumnStretch(column, 0)

        if mode == 'narrow':
            self.top_bar_layout.addWidget(self.title_block, 0, 0)
            self.top_bar_layout.addWidget(
                self.status_label,
                0,
                1,
                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            self.top_bar_layout.addWidget(self.camera_control, 1, 0)
            self.top_bar_layout.addWidget(self.resolution_control, 1, 1)
            self.top_bar_layout.addWidget(self.actions_widget, 2, 0, 1, 2)
            if AUDIO_FEATURE_ENABLED:
                self.top_bar_layout.addWidget(self.audio_controls, 3, 0, 1, 2)
            self.top_bar_layout.setColumnStretch(0, 1)
        elif mode == 'medium':
            self.top_bar_layout.addWidget(self.title_block, 0, 0)
            self.top_bar_layout.addWidget(
                self.status_label,
                0,
                1,
                1,
                2,
                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            self.top_bar_layout.addWidget(self.camera_control, 1, 0)
            self.top_bar_layout.addWidget(self.resolution_control, 1, 1)
            self.top_bar_layout.addWidget(
                self.actions_widget,
                1,
                2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )
            if AUDIO_FEATURE_ENABLED:
                self.top_bar_layout.addWidget(self.audio_controls, 2, 0, 1, 3)
            self.top_bar_layout.setColumnStretch(0, 1)
        else:
            self.top_bar_layout.addWidget(self.title_block, 0, 0)
            self.top_bar_layout.addWidget(self.camera_control, 0, 1)
            self.top_bar_layout.addWidget(self.resolution_control, 0, 2)
            self.top_bar_layout.addWidget(
                self.actions_widget,
                0,
                3,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )
            self.top_bar_layout.addWidget(
                self.status_label,
                0,
                4,
                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            if AUDIO_FEATURE_ENABLED:
                self.top_bar_layout.addWidget(self.audio_controls, 1, 0, 1, 5)
            self.top_bar_layout.setColumnStretch(1, 1)

        self.actions_widget.layout().invalidate()
        self.actions_widget.updateGeometry()
        self.top_bar_layout.activate()

    def load_stylesheet(self):
        """Load the optional QSS file if it exists."""
        style_path = get_app_file_path(DEFAULT_STYLE_FILE)
        if not os.path.exists(style_path):
            logging.info(f"Stylesheet not found at {style_path}; using default Qt styling.")
            return

        try:
            with open(style_path, 'r', encoding='utf-8') as style_file:
                QApplication.instance().setStyleSheet(style_file.read())
            logging.info(f"Loaded stylesheet from {style_path}")
        except Exception as e:
            logging.error(f"Failed to load stylesheet from {style_path}: {e}", exc_info=True)

    def set_camera_status(self, message, state='idle'):
        """Update the compact camera status badge and its visual state."""
        self.status_label.setText(message)
        if self.status_label.property('state') != state:
            self.status_label.setProperty('state', state)
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
        self.status_label.setAccessibleDescription(message)

    def set_controls_busy(self, busy):
        """Prevent overlapping camera changes during a synchronous device scan."""
        self.video_combo.setEnabled(not busy)
        self.resolution_combo.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)

    def video_signal_is_current(self):
        """Ignore queued events emitted by a camera thread that was superseded."""
        source = self.sender()
        if source is None:
            return True
        return self.video_session_accepts_signals and source is self.video_thread

    def handle_capture_opened(self, backend_name, width, height):
        """Show which backend and negotiated resolution opened successfully."""
        if not self.video_signal_is_current():
            logging.info('Ignored capture-opened signal from an old video thread.')
            return
        self.capture_description = f'{backend_name} - {width}x{height}'
        self.set_camera_status(f'Connected - {self.capture_description}', 'connecting')

    def toggle_mirror(self, enabled):
        """Mirror only the rendered preview; camera frames remain unchanged."""
        self.mirror_button.setText('Mirrored' if enabled else '&Mirror')
        self.mirror_button.setShortcut(QKeySequence('Alt+M'))
        self.render_video_frame()

    def save_snapshot(self):
        """Save the currently displayed frame after an explicit file choice."""
        if self.current_frame is None or self.current_frame.isNull():
            self.set_camera_status('No camera frame is available to save', 'warning')
            return

        pictures_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        ) or APP_DIR
        default_name = f"Rikuy-{time.strftime('%Y%m%d-%H%M%S')}.png"
        default_path = os.path.join(pictures_dir, default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Camera Frame',
            default_path,
            'PNG Images (*.png)',
        )
        if not file_path:
            return
        if not os.path.splitext(file_path)[1]:
            file_path += '.png'

        image_to_save = self.current_frame
        if self.mirror_button.isChecked():
            image_to_save = image_to_save.mirrored(True, False)

        if image_to_save.save(file_path, 'PNG'):
            logging.info('Saved camera frame to %s', file_path)
            self.set_camera_status('Snapshot saved', 'live')
        else:
            logging.error('Could not save camera frame to %s', file_path)
            self.set_camera_status('Could not save the camera frame', 'error')
            QMessageBox.warning(self, 'Save Failed', 'Rikuy could not save that camera frame.')

    def toggle_fullscreen(self):
        """Toggle full-screen viewing while keeping all controls available."""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText('Fullscreen')
        else:
            self.showFullScreen()
            self.fullscreen_button.setText('Windowed')
        self.fullscreen_button.setShortcut(QKeySequence('F11'))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def populate_devices(self):
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        for device in self.available_video_devices:
            self.video_combo.addItem(device['name'], userData=device['id'])
            item_index = self.video_combo.count() - 1
            model_item = self.video_combo.model().item(item_index)
            if model_item is not None and device['id'] == -1:
                model_item.setEnabled(False)
        self.video_combo.blockSignals(False)

        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        for device in self.available_audio_devices:
            item_label = device['name']
            if not device.get('available', True):
                item_label = f"{item_label} [Unavailable]"
            self.audio_combo.addItem(item_label, userData=device['id'])
            item_index = self.audio_combo.count() - 1
            model_item = self.audio_combo.model().item(item_index)
            if model_item is not None and not device.get('available', True):
                model_item.setEnabled(False)
        self.audio_combo.blockSignals(False)
        self.update_audio_button_state()

    def populate_resolutions(self):
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for width, height in DEFAULT_RESOLUTIONS:
            resolution_label = format_resolution(width, height)
            self.resolution_combo.addItem(resolution_label, userData=resolution_label)

        loaded_res = (self.last_settings['resolution_w'], self.last_settings['resolution_h'])
        if loaded_res != (-1, -1):
            loaded_res_label = format_resolution(*loaded_res)
            index = self.resolution_combo.findText(loaded_res_label)
            if index != -1:
                self.resolution_combo.setCurrentIndex(index)
                logging.info(f"Set resolution from settings: {loaded_res_label}")
            else:
                logging.warning(f"Saved resolution {loaded_res} not found or unsupported.")

        if self.resolution_combo.currentIndex() == -1:
            index_1080p = self.resolution_combo.findText(format_resolution(1920, 1080))
            if index_1080p != -1:
                self.resolution_combo.setCurrentIndex(index_1080p)
            elif self.resolution_combo.count() > 0:
                self.resolution_combo.setCurrentIndex(0)

        self.resolution_combo.blockSignals(False)

    def apply_initial_settings(self):
        """Apply loaded settings or defaults to combo boxes and start capture."""
        self.video_combo.blockSignals(True)
        self.audio_combo.blockSignals(True)
        self.resolution_combo.blockSignals(True)

        video_applied = False
        if self.last_settings['video_id'] != -1:
            index = self.video_combo.findData(self.last_settings['video_id'])
            if self.is_combo_index_selectable(self.video_combo, index):
                self.video_combo.setCurrentIndex(index)
                logging.info(f"Set video device from settings: ID={self.last_settings['video_id']}")
                video_applied = True
            else:
                logging.warning(f"Saved video device ID {self.last_settings['video_id']} not found or unavailable.")

        if not video_applied:
            self.restore_combo_selection(self.video_combo, None)

        audio_applied = False
        if self.last_settings['audio_id'] != -1:
            index = self.audio_combo.findData(self.last_settings['audio_id'])
            if self.is_combo_index_selectable(self.audio_combo, index):
                self.audio_combo.setCurrentIndex(index)
                logging.info(f"Set audio device from settings: ID={self.last_settings['audio_id']}")
                audio_applied = True
            else:
                logging.warning(f"Saved audio device ID {self.last_settings['audio_id']} not found or unavailable.")

        if not audio_applied:
            self.restore_combo_selection(self.audio_combo, None)

        self.video_combo.blockSignals(False)
        self.audio_combo.blockSignals(False)
        self.resolution_combo.blockSignals(False)

        self.start_capture()

    def update_video_display(self, qt_image):
        """Receive QImage from VideoThread and update the label."""
        if not self.video_signal_is_current():
            logging.info('Ignored frame signal from an old video thread.')
            return
        if not qt_image.isNull():
            first_visible_frame = not self.video_ready
            self.current_frame = qt_image
            self.video_ready = True
            self.snapshot_button.setEnabled(True)
            self.render_video_frame()
            if first_visible_frame:
                details = self.capture_description or f'Camera {self.current_video_device_id}'
                self.set_camera_status(f'Live - {details}', 'live')
        else:
            logging.warning('Received null QImage in update_video_display')

    def update_video_status(self, message):
        """Show a non-fatal camera status until a visible frame arrives."""
        if not self.video_signal_is_current():
            logging.info('Ignored status signal from an old video thread.')
            return
        self.current_frame = None
        self.video_ready = False
        self.snapshot_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText(message)
        status_message = (
            'No visible image'
            if '\n' in message
            else message
        )
        self.set_camera_status(status_message, 'warning')
        self.status_label.setAccessibleDescription(message.replace('\n', ' '))

    def render_video_frame(self):
        """Scale and show the latest frame using the current label size."""
        if self.current_frame is None or self.current_frame.isNull():
            return

        display_image = self.current_frame
        if self.mirror_button.isChecked():
            display_image = display_image.mirrored(True, False)
        pixmap = QPixmap.fromImage(display_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def handle_video_error(self, error_msg):
        """Handle errors signaled from VideoThread on the UI thread."""
        if not self.video_signal_is_current():
            logging.info('Ignored error signal from an old video thread: %s', error_msg)
            return
        logging.error(f"Video Error: {error_msg}")
        self.current_frame = None
        self.video_ready = False
        self.snapshot_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText(f'Video Error: {error_msg}')
        self.set_camera_status('Camera unavailable', 'error')
        self.stop_capture(stop_video=True, stop_audio=False)
        if not self.is_closing:
            QMessageBox.critical(self, 'Video Error', error_msg)

    def handle_audio_error(self, error_msg):
        """Handle errors signaled from AudioThread on the UI thread."""
        if not AUDIO_FEATURE_ENABLED:
            return
        logging.error(f"Audio Error: {error_msg}")
        self.audio_enabled = False
        self.volume_bar.setValue(0)
        self.stop_capture(stop_video=False, stop_audio=True)
        self.update_audio_button_state()
        if not self.is_closing:
            QMessageBox.critical(self, 'Audio Error', error_msg)

    def update_volume(self, level):
        if not AUDIO_FEATURE_ENABLED:
            return
        self.volume_bar.setValue(level)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'top_bar_layout'):
            self.arrange_top_bar(self.top_bar_mode_for_width(event.size().width()))
            self.centralWidget().layout().activate()
        self.render_video_frame()

    def restore_combo_selection(self, combo_box, preferred_value):
        """Restore a combo-box selection, falling back to the first valid device."""
        combo_box.blockSignals(True)
        try:
            if preferred_value is not None:
                preferred_index = combo_box.findData(preferred_value)
                if self.is_combo_index_selectable(combo_box, preferred_index):
                    combo_box.setCurrentIndex(preferred_index)
                    return

            for index in range(combo_box.count()):
                if self.is_combo_index_selectable(combo_box, index):
                    combo_box.setCurrentIndex(index)
                    return

            if combo_box.count() > 0:
                combo_box.setCurrentIndex(0)
        finally:
            combo_box.blockSignals(False)

    def is_combo_index_selectable(self, combo_box, index):
        """Return True when a combo-box entry is valid and enabled."""
        if index is None or index < 0:
            return False
        if combo_box.itemData(index) in (None, -1):
            return False
        model_item = combo_box.model().item(index)
        if model_item is None:
            return True
        return model_item.isEnabled()

    def update_audio_button_state(self):
        """Refresh the mic toggle button based on current state and availability."""
        if not AUDIO_FEATURE_ENABLED:
            self.audio_enabled = False
            self.audio_toggle_button.setText('Audio Disabled')
            self.audio_toggle_button.setEnabled(False)
            return

        has_selectable_audio = any(
            self.is_combo_index_selectable(self.audio_combo, index)
            for index in range(self.audio_combo.count())
        )
        self.audio_toggle_button.setText('Disconnect Mic' if self.audio_enabled else 'Connect Mic')
        self.audio_toggle_button.setEnabled(has_selectable_audio or self.audio_enabled)

    def update_audio_lock_ui(self):
        """Reflect current audio routing in the UI without locking mic selection."""
        if not AUDIO_FEATURE_ENABLED:
            self.audio_combo.setEnabled(False)
            self.audio_combo.setToolTip('Audio is temporarily disabled.')
            self.update_audio_button_state()
            return

        self.audio_combo.setEnabled(True)
        tooltips = []
        if self.output_monitoring_enabled:
            if self.output_device:
                tooltips.append(f'Live playback routed to {self.output_device["name"]}.')
            else:
                tooltips.append(f'Live playback target {describe_locked_output(self.last_settings)} is unavailable.')
        self.audio_combo.setToolTip(' '.join(tooltips))
        self.update_audio_button_state()

    def refresh_devices(self):
        """Re-enumerate audio and video devices and restart capture."""
        previous_video = self.video_combo.currentData()
        previous_audio = self.audio_combo.currentData()

        logging.info('Refreshing device lists.')
        self.set_camera_status('Refreshing cameras...', 'connecting')
        self.set_controls_busy(True)
        QApplication.processEvents()
        try:
            if not self.stop_capture():
                self.set_camera_status('Camera is still shutting down', 'error')
                return
            self.available_video_devices = list_video_devices(show_error=True, parent=self)
            if AUDIO_FEATURE_ENABLED:
                self.available_audio_devices = list_audio_devices(self.p_audio_main, self.last_settings)
                self.available_output_devices = list_output_devices(self.p_audio_main)
                self.output_device = resolve_output_device(self.last_settings, self.available_output_devices)
            self.populate_devices()
            self.restore_combo_selection(self.video_combo, previous_video)
            if AUDIO_FEATURE_ENABLED:
                self.restore_combo_selection(self.audio_combo, previous_audio)
            self.update_audio_lock_ui()
        finally:
            self.set_controls_busy(False)
        self.start_capture()

    def toggle_audio(self):
        """Disconnect or reconnect the current microphone without affecting video."""
        if not AUDIO_FEATURE_ENABLED:
            self.audio_enabled = False
            self.update_audio_button_state()
            return

        self.audio_enabled = not self.audio_enabled
        if self.audio_enabled:
            self.start_capture(start_video=False, start_audio=True)
        else:
            self.stop_capture(stop_video=False, stop_audio=True)
        self.update_audio_button_state()

    def change_video_device(self, index):
        if index < 0:
            return
        selected_id = self.video_combo.itemData(index)
        current_id = getattr(self, 'current_video_device_id', -1)
        if selected_id is not None and selected_id != current_id and selected_id != -1:
            device_name = self.video_combo.itemText(index)
            logging.info(f"Video device change requested: ID={selected_id}, Name='{device_name}'")
            if self.stop_capture(stop_video=True, stop_audio=False):
                self.start_capture(start_video=True, start_audio=False)
            else:
                self.set_camera_status('Camera is still shutting down', 'error')
        elif selected_id == -1:
            logging.info('Video device set to None.')
            self.stop_capture(stop_video=True, stop_audio=False)
            self.set_camera_status('No camera selected', 'idle')

    def change_audio_device(self, index):
        if not AUDIO_FEATURE_ENABLED:
            return
        if not self.is_combo_index_selectable(self.audio_combo, index):
            return
        selected_id = self.audio_combo.itemData(index)
        current_id = getattr(self, 'current_audio_device_id', -1)
        if selected_id is not None and selected_id != current_id and selected_id != -1:
            device_name = self.audio_combo.itemText(index)
            logging.info(f"Audio device change requested: ID={selected_id}, Name='{device_name}'")
            self.current_audio_device_id = selected_id
            if self.audio_enabled:
                self.stop_capture(stop_video=False, stop_audio=True)
                self.start_capture(start_video=False, start_audio=True)
        elif selected_id == -1:
            logging.info('Audio device set to None.')
            self.stop_capture(stop_video=False, stop_audio=True)
            self.current_audio_device_id = -1
        else:
            self.current_audio_device_id = selected_id if selected_id is not None else -1

    def change_resolution(self, index):
        if index < 0:
            return
        selected_res = parse_resolution_text(self.resolution_combo.currentText())
        if selected_res:
            res_text = self.resolution_combo.itemText(index)
            logging.info(f"Resolution change requested: {res_text}")
            if self.stop_capture(stop_video=True, stop_audio=False):
                self.start_capture(start_video=True, start_audio=False)
            else:
                self.set_camera_status('Camera is still shutting down', 'error')

    def start_capture(self, start_video=True, start_audio=True):
        """Start new capture threads based on current selections and flags."""
        video_id = self.video_combo.currentData()
        audio_id = self.audio_combo.currentData() if AUDIO_FEATURE_ENABLED else -1
        resolution = parse_resolution_text(self.resolution_combo.currentText())

        self.current_video_device_id = video_id if video_id is not None else -1
        self.current_audio_device_id = audio_id if AUDIO_FEATURE_ENABLED and audio_id is not None else -1

        if start_video and video_id is not None and video_id != -1:
            if self.video_thread is not None:
                if self.video_thread.isRunning():
                    logging.error('Refusing to start a second VideoThread while one is active.')
                    self.set_camera_status('Previous camera is still shutting down', 'error')
                    return
                if not self.stop_capture(stop_video=True, stop_audio=False):
                    self.set_camera_status('Previous camera is still shutting down', 'error')
                    return

            logging.info(f"Starting VideoThread for device {video_id} with resolution {resolution}")
            video_thread = VideoThread(video_id, resolution, parent=self)
            self.video_thread = video_thread
            self.video_session_accepts_signals = True
            video_thread.frame_signal.connect(self.update_video_display)
            video_thread.status_signal.connect(self.update_video_status)
            video_thread.error_signal.connect(self.handle_video_error)
            video_thread.capture_opened_signal.connect(self.handle_capture_opened)
            self.current_frame = None
            self.capture_description = ''
            self.video_ready = False
            self.snapshot_button.setEnabled(False)
            self.video_label.clear()
            self.video_label.setText('Starting camera...')
            self.set_camera_status('Starting camera...', 'connecting')
            video_thread.start()
        elif start_video:
            logging.info('No video device selected, capture not started.')
            self.current_frame = None
            self.capture_description = ''
            self.video_ready = False
            self.video_session_accepts_signals = False
            self.snapshot_button.setEnabled(False)
            self.video_label.clear()
            self.video_label.setText('No camera is available.\nConnect one, then choose Refresh Cameras.')
            self.set_camera_status('No camera selected', 'idle')

        if start_audio and not AUDIO_FEATURE_ENABLED:
            logging.info('Audio capture is disabled.')
            self.volume_bar.setValue(0)
        elif start_audio and not self.audio_enabled:
            logging.info('Audio capture is currently disconnected by the user.')
            self.volume_bar.setValue(0)
        elif start_audio and audio_id is not None and audio_id != -1:
            output_device_index = self.output_device['id'] if self.output_device else None
            if self.output_monitoring_enabled and output_device_index is None:
                logging.warning('Live mic playback is enabled, but the locked output device is not available.')
            logging.info(f"Starting AudioThread for device {audio_id}")
            self.audio_thread = AudioThread(
                audio_id,
                output_device_index=output_device_index,
                monitor_audio=self.output_monitoring_enabled and output_device_index is not None,
            )
            self.audio_thread.volume_signal.connect(self.update_volume)
            self.audio_thread.error_signal.connect(self.handle_audio_error)
            self.audio_thread.start()
        elif start_audio:
            logging.info('No audio device selected, capture not started.')
            self.volume_bar.setValue(0)

        self.update_audio_button_state()

    def stop_capture(
        self,
        stop_video=True,
        stop_audio=True,
        video_timeout_ms=VIDEO_THREAD_STOP_TIMEOUT_MS,
    ):
        """Stop capture workers, retaining any thread that fails to exit."""
        video_stopped = True
        if stop_video and self.video_thread:
            video_thread = self.video_thread
            vid_id = video_thread.device_index
            self.video_session_accepts_signals = False
            was_running = video_thread.isRunning()
            if was_running:
                logging.info(f"Stopping VideoThread for device {vid_id}...")
            else:
                logging.info(f"Cancelling pending VideoThread for device {vid_id}...")
            video_stopped = video_thread.stop(video_timeout_ms)
            if video_stopped and was_running:
                logging.info(f"VideoThread stopped for device {vid_id}.")
            if video_stopped and self.video_thread is video_thread:
                self.video_thread = None
                self.current_frame = None
                self.capture_description = ''
                self.video_ready = False
                self.snapshot_button.setEnabled(False)
                video_thread.deleteLater()
            elif not video_stopped:
                logging.error('Retaining VideoThread %s because it is still running.', vid_id)

        if stop_audio and self.audio_thread:
            aud_id = self.audio_thread.device_index
            if self.audio_thread.isRunning():
                logging.info(f"Stopping AudioThread for device {aud_id}...")
                self.audio_thread.stop()
                logging.info(f"AudioThread stopped for device {aud_id}.")
            self.audio_thread = None
            self.volume_bar.setValue(0)
        return video_stopped

    def closeEvent(self, event):
        """Ensure cleanup when window is closed."""
        logging.info('Closing application...')
        self.is_closing = True
        if not self.stop_capture(video_timeout_ms=5000):
            self.is_closing = False
            event.ignore()
            self.set_camera_status('Camera did not shut down; close was cancelled', 'error')
            QMessageBox.warning(
                self,
                'Camera Still Running',
                'The camera worker did not shut down safely. Wait a moment and try closing again.',
            )
            return

        current_res = self.resolution_combo.currentData()
        if isinstance(current_res, str):
            current_res = parse_resolution_text(current_res)
        save_settings(
            self.current_video_device_id,
            self.current_audio_device_id if AUDIO_FEATURE_ENABLED else -1,
            current_res,
            locked_audio_id=self.last_settings.get('locked_audio_id', -1),
            locked_audio_name=self.last_settings.get('locked_audio_name', ''),
            monitor_audio=AUDIO_FEATURE_ENABLED and self.last_settings.get('monitor_audio', False),
            locked_output_id=self.last_settings.get('locked_output_id', -1),
            locked_output_name=self.last_settings.get('locked_output_name', ''),
        )

        if self.p_audio_main:
            try:
                self.p_audio_main.terminate()
                logging.info('Main PyAudio instance terminated.')
            except Exception as e:
                logging.error(f"Error terminating main PyAudio: {e}", exc_info=True)
        event.accept()


# --- Main Execution ---
if __name__ == '__main__':
    smoke_test = '--smoke-test' in sys.argv
    qt_arguments = [argument for argument in sys.argv if argument != '--smoke-test']
    app = QApplication(qt_arguments)
    app.setApplicationName('Rikuy')
    app.setDesktopFileName('rikuy')
    window = MainWindow()
    window.show()
    if smoke_test:
        QTimer.singleShot(SMOKE_TEST_DURATION_MS, window.close)
    logging.info('Application started.')
    sys.exit(app.exec())
