"""Original MSST API and root UI with a CPU HTTP parent and managed supervisor."""
import json
import os
import shutil
import sys

from .adapter import load_model, completion, release, cleanup
from .api import build_api

UI_PATH = '/__gradio__'


class NativeUIRoutes:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http', 'websocket'):
            path = scope.get('path', '/')
            if not (path.startswith('/api/v1/') or path in ('/health', '/openapi.json', '/docs', '/redoc') or path.startswith(UI_PATH)):
                scope = dict(scope)
                scope['path'] = UI_PATH + path
                scope['raw_path'] = UI_PATH.encode() + scope.get('raw_path', path.encode())
            if scope.get('path', '').startswith(UI_PATH):
                scope = dict(scope)
                scope['headers'] = [(k,v) for k,v in scope.get('headers', []) if k.lower()!=b'x-forwarded-host']
        await self.app(scope, receive, send)


def create_app(runtime=None):
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    from .ui import build_ui
    runtime = runtime or Runtime(load_model, completion=completion, release=release,
                                 cleanup=cleanup, execution_timeout=None)
    app = build_api(runtime)
    app.add_middleware(NativeUIRoutes)
    return attach(app, runtime=runtime, ui_factory=lambda:build_ui(runtime, app.state.hub_tasks), ui_path=UI_PATH)


def prepare_native_files():
    # Preserve the native bootstrap's missing-file-only initialization.
    if not os.path.exists('configs'):
        shutil.copytree('configs_backup', 'configs')
    if not os.path.exists('data'):
        shutil.copytree('data_backup', 'data')
    else:
        for name in ['webui_config.json', 'language.json', 'models_info.json']:
            if not os.path.exists(os.path.join('data', name)):
                shutil.copy(os.path.join('data_backup', name), os.path.join('data', name))
    # Native setup_webui may replace the entire data directory on version
    # mismatch, including the task DB. Adoption must not perform that migration.
    from webui.utils import load_configs, get_main_link
    from utils.constant import WEBUI_CONFIG, PACKAGE_VERSION, MODEL_FOLDER
    config = load_configs(WEBUI_CONFIG)
    if config.get('version') != PACKAGE_VERSION:
        raise RuntimeError('Preserve MSST data: configuration version requires an explicit migration')
    for directory in ('input', 'results', 'cache'):
        os.makedirs(directory, exist_ok=True)
    os.environ['HF_HOME'] = os.path.abspath(MODEL_FOLDER)
    os.environ['HF_ENDPOINT'] = 'https://' + get_main_link()
    os.environ['PATH'] = os.path.abspath('ffmpeg/bin') + os.pathsep + os.environ['PATH']
    os.environ['GRADIO_TEMP_DIR'] = os.path.abspath('cache')


def main():
    if sys.argv[1:] == ['describe']:
        print(json.dumps(build_api(None).openapi(), ensure_ascii=False))
        return
    if sys.argv[1:]:
        raise SystemExit('Use python -m hub_runtime [describe]')
    prepare_native_files()
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
