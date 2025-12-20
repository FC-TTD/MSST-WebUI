from __future__ import annotations

import os
from typing import Dict, List

from api.models import FileResult


def collect_task_output_files(task_output_dir: str) -> List[FileResult]:
    if not task_output_dir or not os.path.exists(task_output_dir):
        return []

    output_files: List[str] = []
    for root, dirnames, filenames in os.walk(task_output_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
        filenames = [f for f in filenames if not f.startswith(("@", "."))]
        for filename in filenames:
            if not filename.endswith((".wav", ".mp3")):
                continue
            file_path = os.path.join(root, filename)
            try:
                if os.path.getsize(file_path) <= 0:
                    continue
            except Exception:
                continue
            output_files.append(os.path.basename(file_path))

    input_file_groups: Dict[str, List[str]] = {}
    for output_file in output_files:
        input_name = None
        if "_Vocals" in output_file:
            input_name = output_file.split("_Vocals")[0]
        elif "_Instrumental" in output_file:
            input_name = output_file.split("_Instrumental")[0]
        else:
            input_name = f"file_{len(input_file_groups) + 1}"

        input_file_groups.setdefault(input_name, []).append(output_file)

    files: List[FileResult] = []
    for input_name, files_list in input_file_groups.items():
        files.append(
            FileResult(
                input_file=input_name,
                output_files=files_list,
                status="success",
            )
        )
    return files
