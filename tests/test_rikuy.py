"""Unit tests for Rikuy's camera-independent behavior."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import rikuy


class SequenceCapture:
    """Minimal capture double that returns a fixed sequence of frames."""

    def __init__(self, frames):
        self._frames = iter(frames)
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return True, next(self._frames)


class FakeVideoCapture:
    """Capture double with controllable reads and release tracking."""

    def __init__(self, reads=(), *, opened=True, width=640, height=480):
        self._reads = iter(reads)
        self._opened = opened
        self._width = width
        self._height = height
        self.read_count = 0
        self.release_count = 0

    def isOpened(self):
        return self._opened

    def read(self):
        self.read_count += 1
        result = next(self._reads)
        if isinstance(result, BaseException):
            raise result
        return result

    def get(self, property_id):
        if property_id == rikuy.cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if property_id == rikuy.cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        return 0

    def release(self):
        self.release_count += 1


class RikuyHelperTests(unittest.TestCase):
    def test_audio_feature_remains_disabled(self):
        self.assertIs(rikuy.AUDIO_FEATURE_ENABLED, False)

    def test_resolution_format_and_parse(self):
        label = rikuy.format_resolution(1280, 720)

        self.assertEqual(label, "1280x720")
        self.assertEqual(rikuy.parse_resolution_text(label), (1280, 720))

    def test_parse_resolution_rejects_invalid_text(self):
        invalid_values = (
            None,
            "",
            "1280",
            "x720",
            "1280x",
            "1280x720x30",
            "1280*720",
            "0x720",
            "1280x0",
            "-1x720",
            "widextall",
            123,
        )

        for value in invalid_values:
            with self.subTest(value=value):
                self.assertIsNone(rikuy.parse_resolution_text(value))

    def test_frame_visibility_rejects_missing_empty_and_dark_frames(self):
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        black_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        threshold_frame = np.full(
            (2, 2, 3),
            rikuy.VISIBLE_FRAME_MEAN_THRESHOLD,
            dtype=np.float32,
        )

        self.assertFalse(rikuy.frame_looks_visible(None))
        self.assertFalse(rikuy.frame_looks_visible(empty_frame))
        self.assertFalse(rikuy.frame_looks_visible(black_frame))
        self.assertFalse(rikuy.frame_looks_visible(threshold_frame))

    def test_frame_visibility_accepts_frame_above_threshold(self):
        visible_frame = np.full(
            (2, 2, 3),
            rikuy.VISIBLE_FRAME_MEAN_THRESHOLD + 1,
            dtype=np.float32,
        )

        self.assertTrue(rikuy.frame_looks_visible(visible_frame))


class RikuyPlatformTests(unittest.TestCase):
    def test_linux_uses_v4l2_without_generic_fallback(self):
        with mock.patch.object(rikuy.sys, 'platform', 'linux'):
            self.assertEqual(
                rikuy.get_video_capture_backends(),
                [('Video4Linux2', rikuy.cv2.CAP_V4L2)],
            )

    def test_windows_backend_order_is_unchanged(self):
        with mock.patch.object(rikuy.sys, 'platform', 'win32'):
            self.assertEqual(
                rikuy.get_video_capture_backends(),
                [
                    ('DirectShow', rikuy.cv2.CAP_DSHOW),
                    ('Media Foundation', rikuy.cv2.CAP_MSMF),
                ],
            )

    def test_linux_uses_only_existing_capture_device_nodes(self):
        with (
            mock.patch.object(rikuy.sys, 'platform', 'linux'),
            mock.patch.object(
                rikuy.glob,
                'glob',
                return_value=[
                    '/dev/video2',
                    '/dev/video0',
                    '/dev/video1',
                    '/dev/video2',
                    '/dev/video-meta',
                ],
            ),
            mock.patch.object(
                rikuy,
                'linux_video_device_supports_capture',
                side_effect=lambda path: not path.endswith('video1'),
            ),
        ):
            self.assertEqual(rikuy.get_video_device_indices(), [0, 2])

    def test_non_linux_scan_limit_is_unchanged(self):
        with mock.patch.object(rikuy.sys, 'platform', 'win32'):
            self.assertEqual(
                rikuy.get_video_device_indices(),
                list(range(rikuy.VIDEO_DEVICE_SCAN_LIMIT)),
            )

    def test_linux_camera_name_uses_sysfs_label_and_device_node(self):
        with (
            mock.patch.object(rikuy.sys, 'platform', 'linux'),
            mock.patch('builtins.open', mock.mock_open(read_data='USB Camera\n')),
        ):
            self.assertEqual(
                rikuy.get_video_device_name(2),
                'USB Camera (/dev/video2)',
            )

    def test_linux_no_camera_state_does_not_show_blocking_dialog(self):
        with (
            mock.patch.object(rikuy.sys, 'platform', 'linux'),
            mock.patch.object(rikuy, 'get_video_device_indices', return_value=[]),
            mock.patch.object(rikuy.QMessageBox, 'warning') as warning,
        ):
            self.assertEqual(
                rikuy.list_video_devices(show_error=True),
                [{'id': -1, 'name': 'No Video Devices Found'}],
            )
            warning.assert_not_called()


class RikuyConfigTests(unittest.TestCase):
    def test_malformed_setting_does_not_discard_later_valid_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "malformed.ini"
            config_path.write_text(
                """[Settings]
last_video_id = 3
last_audio_id = not-an-integer
last_resolution_w = 1920
last_resolution_h = 1080
locked_audio_id = 7
locked_audio_name = Desk Microphone
monitor_audio = true
locked_output_id = 9
locked_output_name = Desk Speakers
""",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {rikuy.CONFIG_ENV_VAR: str(config_path.resolve())},
            ):
                settings = rikuy.load_settings()

        self.assertEqual(settings["video_id"], 3)
        self.assertEqual(settings["audio_id"], -1)
        self.assertEqual(settings["resolution_w"], 1920)
        self.assertEqual(settings["resolution_h"], 1080)
        self.assertEqual(settings["locked_audio_id"], 7)
        self.assertEqual(settings["locked_audio_name"], "Desk Microphone")
        self.assertTrue(settings["monitor_audio"])
        self.assertEqual(settings["locked_output_id"], 9)
        self.assertEqual(settings["locked_output_name"], "Desk Speakers")

    def test_atomic_save_and_load_round_trip_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_path = config_dir / "round-trip.ini"
            real_replace = os.replace

            with mock.patch.dict(
                os.environ,
                {rikuy.CONFIG_ENV_VAR: str(config_path.resolve())},
            ):
                with mock.patch("rikuy.os.replace", wraps=real_replace) as replace_mock:
                    rikuy.save_settings(
                        video_id=2,
                        audio_id=-1,
                        resolution=(1280, 720),
                        locked_audio_id=4,
                        locked_audio_name="Café Microphone",
                        monitor_audio=True,
                        locked_output_id=6,
                        locked_output_name="Desk Speakers",
                    )

                settings = rikuy.load_settings()

            replace_mock.assert_called_once()
            replace_source, replace_target = replace_mock.call_args.args[:2]
            self.assertEqual(Path(replace_target).resolve(), config_path.resolve())
            self.assertFalse(Path(replace_source).exists())
            self.assertEqual(
                {entry.name for entry in config_dir.iterdir()},
                {config_path.name},
            )
            self.assertEqual(settings["video_id"], 2)
            self.assertEqual(settings["audio_id"], -1)
            self.assertEqual(settings["resolution_w"], 1280)
            self.assertEqual(settings["resolution_h"], 720)
            self.assertEqual(settings["locked_audio_id"], 4)
            self.assertEqual(settings["locked_audio_name"], "Café Microphone")
            self.assertTrue(settings["monitor_audio"])
            self.assertEqual(settings["locked_output_id"], 6)
            self.assertEqual(settings["locked_output_name"], "Desk Speakers")

    def test_atomic_replace_failure_preserves_existing_config_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_path = config_dir / "existing.ini"
            original_text = "[Settings]\nlast_video_id = 42\n"
            config_path.write_text(original_text, encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {rikuy.CONFIG_ENV_VAR: str(config_path.resolve())},
            ):
                with mock.patch(
                    "rikuy.os.replace",
                    side_effect=OSError("simulated replacement failure"),
                ) as replace_mock:
                    with self.assertLogs(level="ERROR"):
                        saved = rikuy.save_settings(
                            video_id=2,
                            audio_id=-1,
                            resolution=(1280, 720),
                        )

            self.assertFalse(saved)
            replace_mock.assert_called_once()
            replace_source, replace_target = replace_mock.call_args.args[:2]
            self.assertEqual(Path(replace_target).resolve(), config_path.resolve())
            self.assertFalse(Path(replace_source).exists())
            self.assertEqual(config_path.read_text(encoding="utf-8"), original_text)
            self.assertEqual(
                {entry.name for entry in config_dir.iterdir()},
                {config_path.name},
            )

    def test_repeated_saves_use_unique_same_directory_temp_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "shared.ini"
            replace_sources = []
            real_replace = os.replace

            def record_replace(source, target):
                replace_sources.append(Path(source))
                real_replace(source, target)

            with mock.patch.dict(
                os.environ,
                {rikuy.CONFIG_ENV_VAR: str(config_path.resolve())},
            ), mock.patch("rikuy.os.replace", side_effect=record_replace):
                self.assertTrue(rikuy.save_settings(1, -1, (640, 480)))
                self.assertTrue(rikuy.save_settings(2, -1, (1280, 720)))

        self.assertEqual(len(replace_sources), 2)
        self.assertNotEqual(replace_sources[0], replace_sources[1])
        self.assertTrue(all(path.parent == config_path.parent for path in replace_sources))


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        settings = {
            "video_id": -1,
            "audio_id": -1,
            "resolution_w": 1280,
            "resolution_h": 720,
            "locked_audio_id": -1,
            "locked_audio_name": "",
            "monitor_audio": False,
            "locked_output_id": -1,
            "locked_output_name": "",
        }
        self.patchers = [
            mock.patch.object(rikuy, "load_settings", return_value=settings),
            mock.patch.object(
                rikuy,
                "list_video_devices",
                return_value=[{"id": -1, "name": "No Video Devices Found"}],
            ),
            mock.patch.object(rikuy, "save_settings", return_value=True),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.window = rikuy.MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.video_thread = None
        self.window.close()
        self.app.processEvents()
        for patcher in reversed(self.patchers):
            patcher.stop()

    def test_remixed_shell_exposes_controls_and_no_camera_state(self):
        self.assertEqual(self.window.centralWidget().objectName(), "appRoot")
        self.assertEqual(self.window.video_label.objectName(), "videoPreview")
        self.assertEqual(self.window.top_bar.objectName(), "topBar")
        self.assertIsNone(self.window.findChild(rikuy.QFrame, "controlsPanel"))
        self.assertTrue(self.window.top_bar.isAncestorOf(self.window.video_combo))
        self.assertTrue(self.window.top_bar.isAncestorOf(self.window.resolution_combo))
        self.assertTrue(self.window.top_bar.isAncestorOf(self.window.refresh_button))
        self.assertEqual(self.window.status_label.property("state"), "idle")
        self.assertEqual(self.window.status_label.text(), "No camera selected")
        self.assertEqual(self.window.mirror_button.shortcut().toString(), "Alt+M")
        self.assertEqual(self.window.snapshot_button.shortcut().toString(), "Ctrl+S")
        self.assertEqual(self.window.fullscreen_button.shortcut().toString(), "F11")
        self.assertEqual(self.window.refresh_button.shortcut().toString(), "F5")
        self.assertFalse(self.window.snapshot_button.isEnabled())
        self.assertFalse(self.window.audio_toggle_button.isEnabled())

    def test_stale_video_status_signal_is_ignored(self):
        stale_thread = rikuy.VideoThread(0, (640, 480))
        active_thread = rikuy.VideoThread(1, (1280, 720))
        self.window.video_thread = active_thread
        self.window.video_session_accepts_signals = True
        original_text = self.window.status_label.text()

        stale_thread.status_signal.connect(self.window.update_video_status)
        stale_thread.status_signal.emit("stale status")

        self.assertEqual(self.window.status_label.text(), original_text)

        active_thread.status_signal.connect(self.window.update_video_status)
        active_thread.status_signal.emit("Camera signal interrupted. Retrying...")

        self.assertEqual(
            self.window.status_label.text(),
            "Camera signal interrupted. Retrying...",
        )
        self.assertEqual(self.window.status_label.property("state"), "warning")

    def test_snapshot_saves_the_visible_mirrored_frame(self):
        frame = QImage(2, 1, QImage.Format.Format_RGB32)
        frame.setPixelColor(0, 0, QColor("red"))
        frame.setPixelColor(1, 0, QColor("blue"))
        self.window.current_frame = frame
        self.window.video_ready = True
        self.window.mirror_button.setChecked(True)

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshot.png"
            with mock.patch.object(
                rikuy.QFileDialog,
                "getSaveFileName",
                return_value=(str(snapshot_path), "PNG Images (*.png)"),
            ):
                self.window.save_snapshot()
            saved_frame = QImage(str(snapshot_path))

        self.assertFalse(saved_frame.isNull())
        self.assertEqual(saved_frame.pixelColor(0, 0), QColor("blue"))
        self.assertEqual(saved_frame.pixelColor(1, 0), QColor("red"))

    def test_rendered_frame_fits_inside_padded_preview_contents(self):
        self.window.resize(640, 440)
        self.app.processEvents()
        self.window.current_frame = QImage(1600, 900, QImage.Format.Format_RGB32)

        self.window.render_video_frame()

        rendered_frame = self.window.video_label.pixmap()
        preview_size = self.window.video_label.contentsRect().size()
        self.assertIsNotNone(rendered_frame)
        self.assertLessEqual(rendered_frame.width(), preview_size.width())
        self.assertLessEqual(rendered_frame.height(), preview_size.height())

    def test_preview_dominates_the_larger_default_window(self):
        self.window.resize(1120, 780)
        self.app.processEvents()
        preview_frame = self.window.findChild(rikuy.QFrame, "previewFrame")

        self.assertIsNotNone(preview_frame)
        self.assertGreater(preview_frame.height(), self.window.height() * 0.82)
        bottom_gap = (
            self.window.centralWidget().contentsRect().bottom()
            - preview_frame.geometry().bottom()
        )
        self.assertLessEqual(bottom_gap, 12)

    def test_top_bar_wraps_responsively_and_preserves_shortcuts(self):
        cases = (
            (640, 440, "narrow", False),
            (1120, 780, "medium", False),
            (2048, 1120, "wide", True),
        )
        for width, height, expected_mode, subtitle_visible in cases:
            with self.subTest(width=width):
                self.window.resize(width, height)
                self.app.processEvents()
                self.assertEqual(self.window._top_bar_mode, expected_mode)
                self.assertEqual(self.window.app_subtitle.isVisible(), subtitle_visible)
                for widget in (
                    self.window.title_block,
                    self.window.camera_control,
                    self.window.resolution_control,
                    self.window.actions_widget,
                    self.window.status_label,
                ):
                    self.assertTrue(self.window.top_bar.rect().contains(widget.geometry()))
                self.assertEqual(self.window.mirror_button.shortcut().toString(), "Alt+M")
                self.assertEqual(self.window.snapshot_button.shortcut().toString(), "Ctrl+S")
                self.assertEqual(self.window.fullscreen_button.shortcut().toString(), "F11")
                self.assertEqual(self.window.refresh_button.shortcut().toString(), "F5")

        self.window.mirror_button.setChecked(True)
        self.window.toggle_fullscreen()
        self.app.processEvents()
        self.assertEqual(self.window.mirror_button.shortcut().toString(), "Alt+M")
        self.assertEqual(self.window.fullscreen_button.shortcut().toString(), "F11")
        self.window.toggle_fullscreen()

    def test_stop_timeout_retains_live_thread_reference(self):
        stuck_thread = mock.Mock()
        stuck_thread.device_index = 9
        stuck_thread.isRunning.return_value = True
        stuck_thread.stop.return_value = False
        self.window.video_thread = stuck_thread

        stopped = self.window.stop_capture(stop_audio=False, video_timeout_ms=25)

        self.assertFalse(stopped)
        self.assertIs(self.window.video_thread, stuck_thread)
        stuck_thread.stop.assert_called_once_with(25)
        self.window.video_thread = None

    def test_stop_timeout_invalidates_queued_worker_signals(self):
        stopping_thread = rikuy.VideoThread(9, (640, 480))
        stopping_thread.status_signal.connect(self.window.update_video_status)
        self.window.video_thread = stopping_thread
        self.window.video_session_accepts_signals = True
        original_status = self.window.status_label.text()

        with mock.patch.object(
            stopping_thread,
            "isRunning",
            return_value=True,
        ), mock.patch.object(
            stopping_thread,
            "stop",
            return_value=False,
        ):
            stopped = self.window.stop_capture(stop_audio=False, video_timeout_ms=25)
            stopping_thread.status_signal.emit("late worker status")

        self.assertFalse(stopped)
        self.assertEqual(self.window.status_label.text(), original_status)
        self.assertFalse(self.window.video_session_accepts_signals)
        self.window.video_thread = None

    def test_stop_capture_cancels_thread_before_run_begins(self):
        pending_thread = mock.Mock()
        pending_thread.device_index = 4
        pending_thread.isRunning.return_value = False
        pending_thread.stop.return_value = True
        self.window.video_thread = pending_thread

        stopped = self.window.stop_capture(stop_audio=False, video_timeout_ms=25)

        self.assertTrue(stopped)
        pending_thread.stop.assert_called_once_with(25)
        self.assertIsNone(self.window.video_thread)


class VideoThreadTests(unittest.TestCase):
    def test_candidate_resolutions_preserve_order_and_remove_duplicates(self):
        cases = (
            (
                (1920, 1080),
                [(1920, 1080), (1280, 720), (640, 480), None],
            ),
            (
                (1280, 720),
                [(1280, 720), (640, 480), None],
            ),
            (
                None,
                [None, (1280, 720), (640, 480)],
            ),
        )

        for target, expected in cases:
            with self.subTest(target=target):
                thread = rikuy.VideoThread(device_index=0, resolution=target)
                self.assertEqual(thread.candidate_resolutions(), expected)

    def test_read_startup_frame_prefers_first_visible_frame(self):
        black_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        dim_frame = np.full((2, 2, 3), 1, dtype=np.uint8)
        visible_frame = np.full((2, 2, 3), 20, dtype=np.uint8)
        capture = SequenceCapture([black_frame, dim_frame, visible_frame])
        thread = rikuy.VideoThread(device_index=0, resolution=(1280, 720))

        with mock.patch("rikuy.time.sleep") as sleep_mock:
            frame, visible = thread.read_startup_frame(capture)

        self.assertIs(frame, visible_frame)
        self.assertTrue(visible)
        self.assertEqual(capture.read_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_open_capture_continues_after_attempt_exceptions_in_order(self):
        backend_a = 101
        backend_b = 202
        backends = [("Backend A", backend_a), ("Backend B", backend_b)]
        resolutions = [(1920, 1080), (1280, 720), None]
        visible_frame = np.full((2, 2, 3), 20, dtype=np.uint8)
        configure_failure = FakeVideoCapture()
        read_failure = FakeVideoCapture(
            reads=[RuntimeError("simulated read failure")],
        )
        successful_capture = FakeVideoCapture(
            reads=[(True, visible_frame)],
            width=1280,
            height=720,
        )
        configure_calls = []
        thread = rikuy.VideoThread(device_index=7, resolution=resolutions[0])

        def configure_capture(capture, resolution):
            configure_calls.append((capture, resolution))
            if capture is configure_failure:
                raise RuntimeError("simulated configuration failure")

        constructor_results = [
            RuntimeError("simulated constructor failure"),
            configure_failure,
            read_failure,
            successful_capture,
        ]
        with mock.patch(
            "rikuy.get_video_capture_backends",
            return_value=backends,
        ), mock.patch.object(
            thread,
            "candidate_resolutions",
            return_value=resolutions,
        ), mock.patch.object(
            thread,
            "configure_capture",
            side_effect=configure_capture,
        ), mock.patch(
            "rikuy.cv2.VideoCapture",
            side_effect=constructor_results,
        ) as video_capture_factory:
            capture, first_frame = thread.open_capture()

        self.assertIs(capture, successful_capture)
        self.assertIs(first_frame, visible_frame)
        self.assertEqual(
            video_capture_factory.call_args_list,
            [
                mock.call(7, backend_a),
                mock.call(7, backend_a),
                mock.call(7, backend_a),
                mock.call(7, backend_b),
            ],
        )
        self.assertEqual(
            configure_calls,
            [
                (configure_failure, (1280, 720)),
                (read_failure, None),
                (successful_capture, (1920, 1080)),
            ],
        )
        self.assertEqual(configure_failure.release_count, 1)
        self.assertEqual(read_failure.release_count, 1)
        self.assertEqual(successful_capture.release_count, 0)

    def test_sustained_invalid_reads_emit_one_error_and_release_capture(self):
        black_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        capture = FakeVideoCapture(
            reads=[
                (False, None),
                (True, None),
                (True, empty_frame),
            ],
        )
        thread = rikuy.VideoThread(device_index=3, resolution=(640, 480))
        errors = []
        thread.error_signal.connect(errors.append)

        with mock.patch.object(
            thread,
            "open_capture",
            return_value=(capture, black_frame),
        ) as open_capture, mock.patch.object(
            rikuy,
            "VIDEO_READ_FAILURE_LIMIT",
            3,
        ), mock.patch.object(
            rikuy,
            "VIDEO_READ_WARNING_INTERVAL",
            2,
        ), mock.patch(
            "rikuy.time.sleep",
        ):
            thread.run()

        open_capture.assert_called_once_with()
        self.assertEqual(capture.read_count, 3)
        self.assertEqual(len(errors), 1)
        self.assertIn("3 consecutive frame failures", errors[0])
        self.assertEqual(capture.release_count, 1)
        self.assertIsNone(thread.video_capture)

    def test_stop_requested_before_run_prevents_capture_open(self):
        class RecordingVideoThread(rikuy.VideoThread):
            def __init__(self):
                super().__init__(device_index=0, resolution=(640, 480))
                self.open_capture_called = False

            def open_capture(self):
                self.open_capture_called = True
                return None, None

        thread = RecordingVideoThread()

        thread.stop()
        thread.run()

        self.assertFalse(thread.open_capture_called)


if __name__ == "__main__":
    unittest.main()
