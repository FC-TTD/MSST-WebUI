"""UI callback boundary tests; native GPU dependencies are deliberately unavailable."""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "hub_runtime/ui/pages"
GPU = {
    "msst": ["run_inference_single", "run_multi_inference", "run_folder_batch_inference"],
    "vr": ["vr_inference_single", "vr_inference_multi"],
    "preset": ["preset_inference", "preset_inference_audio"],
    "ensemble": ["inference_audio_func", "inference_folder_func"],
    "tools": ["some_inference"],
    "train": ["validate_model"],
}
STOPS = {"msst": "stop_msst_inference", "vr": "stop_vr_inference", "preset": "stop_preset", "ensemble": "stop_ensemble_func", "train": "stop_msst_valid"}


class Tasks:
    def __init__(self):
        self.calls = []

    def run_ui(self, entry, *args):
        self.calls.append((entry, args))
        return ("native result", []) if entry.startswith("msst.") else "native result"

    def stop_ui(self, kind):
        self.calls.append(("stop:" + kind, ()))
        return "stopped"

    def start_training(self, *args):
        self.calls.append(("train.start_training", args))
        return "native training started"


class UIContractTest(unittest.TestCase):
    def setUp(self):
        self.events, self.components, self.cpu_calls, self.themes, self.queues = [], [], [], [], []
        test = self

        class Component:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get("value", args[0] if args else None)
                self.kwargs = kwargs
                test.components.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def click(self, fn=None, inputs=None, outputs=None, **kwargs):
                test.events.append((fn, inputs or [], kwargs))
                return self

            change = select = click

            def queue(self, *args, **kwargs):
                test.queues.append((args, kwargs))
                return self

        gr = types.ModuleType("gradio")
        gr.__getattr__ = lambda name: Component
        gr.update = lambda **kwargs: kwargs
        gr.Theme = types.SimpleNamespace(load=lambda value: self.themes.append(value))
        pd = types.ModuleType("pandas")
        pd.DataFrame = lambda *args, **kwargs: []
        modules = {"gradio": gr, "pandas": pd}
        self.config = json.loads((ROOT / "data_backup/webui_config.json").read_text())
        self.config["inference"]["device"] = ["2: previous physical GPU"]
        self.config["training"]["device"] = ["1: previous physical GPU"]
        self.config["inference"]["force_cpu"] = True

        def native(module, name):
            def callback(*args, **kwargs):
                self.cpu_calls.append((module + "." + name, args))
                if name == "i18n":
                    return args[0]
                if name == "init_selected_model":
                    return 1, 4, 44100, False
                if name == "init_selected_vr_model":
                    return "Vocals", "Instrumental"
                if name == "load_configs":
                    return {"Auto": "Auto"} if args[0].endswith("language.json") else self.config
                return []
            callback.__name__ = name
            callback.native_name = module + "." + name
            return callback

        # Supplying only explicitly imported CPU helpers makes a raw GPU import fail.
        for source in list(PAGES.glob("*.py")) + [ROOT / "hub_runtime/ui/app.py", ROOT / "hub_runtime/ui/__init__.py"]:
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("webui."):
                    continue
                module = modules.setdefault(node.module, types.ModuleType(node.module))
                for alias in node.names:
                    forbidden = GPU.get(node.module.split(".")[-1], []) + list(STOPS.values()) + ["start_training"]
                    self.assertNotIn(alias.name, forbidden, "GPU function imported into UI parent")
                    setattr(module, alias.name, native(node.module, alias.name))
        parent = types.ModuleType("webui")
        parent.__path__ = []
        modules["webui"] = parent
        # Remove previous copied modules so each test captures its own dependencies.
        self.old_ui = {k: v for k, v in sys.modules.items() if k == "hub_runtime.ui" or k.startswith("hub_runtime.ui.")}
        for key in self.old_ui:
            del sys.modules[key]
        self.mock_modules = patch.dict(sys.modules, modules)
        self.mock_modules.start()
        self.previous_dir = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, self.previous_dir)
        self.addCleanup(self.mock_modules.stop)
        self.addCleanup(self.restore_ui)

    def restore_ui(self):
        for key in list(sys.modules):
            if key == "hub_runtime.ui" or key.startswith("hub_runtime.ui."):
                del sys.modules[key]
        sys.modules.update(self.old_ui)

    def build(self, tasks):
        return importlib.import_module("hub_runtime.ui").build_ui(object(), tasks)

    def test_all_gpu_callbacks_and_stop_controls_use_task_bridge(self):
        tasks = Tasks()
        self.build(tasks)
        for fn, inputs, options in self.events:
            if fn is None:
                continue
            if getattr(fn, "native_name", None):
                continue
            if getattr(fn, "__name__", "") == "<lambda>":
                continue
            values = [item.value for item in inputs] if isinstance(inputs, list) else [inputs.value]
            fn(*values)
        expected = {f"{module}.{name}" for module, names in GPU.items() for name in names}
        expected |= {"train.start_training", "stop:msst", "stop:vr", "stop:preset", "stop:ensemble", "stop:valid"}
        self.assertEqual({entry for entry, _ in tasks.calls}, expected)
        calls = dict(tasks.calls)
        # User CPU choices and original input arity cross unchanged; no silent device-mode rewrite.
        self.assertIs(calls["msst.run_inference_single"][6], True)
        self.assertIs(calls["vr.vr_inference_single"][4], True)
        self.assertEqual(len(calls["train.start_training"]), 18)
        self.assertEqual(len(calls["train.validate_model"]), 11)
        self.assertEqual(self.queues, [((), {})], "native queue defaults changed")
        self.assertEqual(self.themes, ["tools/themes/theme_blue.json"])
        self.assertNotIn("webui.ui", sys.modules)

    def test_cpu_tools_do_not_enter_pool(self):
        tasks = Tasks()
        self.build(tasks)
        cpu = {"webui.ensemble.ensemble_files", "webui.tools.convert_audio", "webui.tools.merge_audios", "webui.tools.caculate_sdr"}
        invoked = set()
        for fn, inputs, options in self.events:
            name = getattr(fn, "native_name", None)
            if name in cpu:
                fn(*[item.value for item in inputs])
                invoked.add(name)
        self.assertEqual(invoked, cpu)
        self.assertEqual(tasks.calls, [])

    def test_builds_do_not_rebind_each_others_callbacks_or_persist_device_choices(self):
        first, second = Tasks(), Tasks()
        self.build(first)
        first_events = list(self.events)
        self.build(second)
        for fn, inputs, options in first_events:
            if getattr(fn, "args", ()) == ("tools.some_inference",):
                fn("audio.wav", 120, "results")
        self.assertEqual(first.calls, [("tools.some_inference", ("audio.wav", 120, "results"))])
        self.assertEqual(second.calls, [])
        self.assertEqual(self.config["inference"]["device"], ["2: previous physical GPU"])
        self.assertEqual(self.config["training"]["device"], ["1: previous physical GPU"])
        gpu_controls = [c for c in self.components if c.kwargs.get("label") == "选择使用的GPU"]
        self.assertTrue(gpu_controls)
        for control in gpu_controls:
            self.assertEqual(control.kwargs["choices"], ["0: Hub allocated GPU (cuda:0)"])
            self.assertEqual(control.value, ["0: Hub allocated GPU (cuda:0)"])

    def test_native_page_controls_and_event_options_are_preserved(self):
        # Beyond the six local callback bindings and injected tasks argument, the
        # copied page function bodies must remain equivalent to native source.
        for module in GPU:
            original = ast.parse((ROOT / f"webui/ui/{module}.py").read_text())
            copied = ast.parse((PAGES / f"{module}.py").read_text())
            orig_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == module)
            copy_fn = next(n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == module)
            bindings = len(GPU[module]) + (1 if module in STOPS else 0) + (1 if module == "train" else 0)
            copy_fn.body = copy_fn.body[bindings:]
            copy_fn.args.kwonlyargs = []
            copy_fn.args.kw_defaults = []
            self.assertEqual(ast.dump(orig_fn), ast.dump(copy_fn), module)


if __name__ == "__main__":
    unittest.main()
