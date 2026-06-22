# Rikuy Project Instructions

These instructions apply to the entire repository.

## Engineering priorities

- Preserve camera startup reliability. Treat the Windows capture-backend
  order, device scanning, warm-up frames, frame visibility checks, and
  fallback behavior as reliability-sensitive code.
- Do not re-enable audio or change audio behavior unless the user explicitly
  asks for it. `AUDIO_FEATURE_ENABLED` must remain disabled by default.
- Prefer small, testable changes with a narrow behavioral surface.
- Keep Windows and PyQt6 support as the primary platform target.

## Change guidance

- Keep `rikuy.py`, `style.qss`, `Rikuy_Condor_Icon.ico`, and the default
  `config.ini` location compatible with launching from the repository root.
- Do not commit `config.ini`; use `config.example.ini` for documented defaults.
- Avoid broad camera enumeration or threading rewrites without a reproducible
  failure case and focused validation.
- Preserve current camera and audio behavior during documentation, packaging,
  and repository-maintenance changes.

## Validation

- Run `python -m py_compile rikuy.py` after Python changes.
- Run `git diff --check` before committing.
- When camera logic changes, test startup, device refresh, camera switching,
  resolution switching, and clean shutdown on Windows hardware when possible.
