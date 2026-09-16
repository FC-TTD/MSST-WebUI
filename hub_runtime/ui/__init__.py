"""Copied native UI; only GPU execution callbacks cross the managed task bridge."""


def build_ui(runtime, tasks):
    """Build the native queued UI without exposing physical GPUs in its parent.

    ``tasks`` owns SDK activity lifetime, including training subprocess completion.
    UI construction does not load an engine or claim a GPU lease.
    """
    import os
    import platform
    from .app import app
    from utils.constant import WEBUI_CONFIG, THEME_FOLDER
    from webui.utils import load_configs

    config = load_configs(WEBUI_CONFIG)
    theme = os.path.join(THEME_FOLDER, config.get("settings", {}).get("theme") or "theme_blue.json")
    return app(
        platform.system(),
        {0: "0: Hub allocated GPU (cuda:0)"},
        False,
        theme=theme,
        tasks=tasks,
    ).queue()
