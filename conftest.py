"""pytest: give the whole session its own config, state and data folders, so a test that reaches the server's per-user files
(strata_control.paths: presets, tuning history, update records) never touches the real ones."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="llmserve-pytest-")
for _var in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
    os.environ[_var] = _home
