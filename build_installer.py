#!/usr/bin/env python3
"""Rebuild NeuralNetwork.cmd from the editable maa_value_plugin source."""
from pathlib import Path
import base64
import io
import re
import zipfile

ROOT = Path(__file__).resolve().parent
INSTALLER = ROOT / "NeuralNetwork.cmd"
PLUGIN = ROOT / "maa_value_plugin"
raw = INSTALLER.read_bytes()
match = re.search(rb"(?m)^::STELLA_VALUE_MODEL_PAYLOAD::(?=\r?$)", raw)
if match is None:
    raise SystemExit("payload marker not found in NeuralNetwork.cmd")
newline = b"\r\n" if b"\r\n" in raw else b"\n"
prefix = raw[:match.start()]
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(PLUGIN.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(ROOT).as_posix())
payload = base64.b64encode(archive.getvalue())
wrapped = newline.join(payload[i:i + 76] for i in range(0, len(payload), 76))
INSTALLER.write_bytes(prefix + b"::STELLA_VALUE_MODEL_PAYLOAD::" + newline + wrapped + newline)
print(f"Rebuilt {INSTALLER.name}")
