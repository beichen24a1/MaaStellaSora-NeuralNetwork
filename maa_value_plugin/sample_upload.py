"""Persistent per-run upload queue controlled by Maa settings."""
from pathlib import Path
import hashlib
import http.client
import json
import random
import time
import threading
import urllib.error
import urllib.request
import zipfile

from .sampling_data import timestamp, write_json


class SampleUpload:
    def __init__(self, app_dir, module_root, directory, logger):
        self.app_dir, self.module_root, self.directory = Path(app_dir), Path(module_root), Path(directory)
        self.logger = logger
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread = None
        self.settings = json.loads((self.module_root / "upload_settings.json").read_text(encoding="utf-8-sig"))

    def upload_allowed(self):
        try:
            settings = json.loads((self.app_dir / "appsettings.json").read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False
        value = settings.get("HelpImproveSoftware", True) if isinstance(settings, dict) else None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and (value := value.strip().lower()) in ("true", "false"):
            return value == "true"
        return False

    def start(self):
        if not self.settings["endpoint"]:
            return
        self.thread = threading.Thread(target=self._worker, name="tower-sample-upload", daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop.is_set():
            self.wake.clear()
            try:
                delay = self.flush()
            except (OSError, ValueError):
                delay = 30
            self.wake.wait(delay)

    def _retry_path(self, marker):
        return self.directory / "retries" / marker.name

    def _defer(self, marker, state, permanent=False):
        attempts = state.get("attempts", 0) + 1
        delay = 1800 if permanent else min(120, 3 * 2 ** min(attempts - 1, 6))
        write_json(self._retry_path(marker), {
            "attempts": attempts, "next_attempt": time.time() + delay + random.uniform(0, delay * 0.25)
        })

    def flush(self):
        delay = 30
        if not self.settings["endpoint"] or not self.upload_allowed():
            return delay
        # Serve the just-finished run ahead of older retries. Retry metadata stays
        # outside the immutable ZIP and does not change the ready marker's age.
        for marker in sorted((self.directory / "queue").glob("*.ready.json"),
                             key=lambda path: path.stat().st_mtime_ns, reverse=True):
            if not self.upload_allowed():
                return delay
            retry = self._retry_path(marker)
            state = json.loads(retry.read_text(encoding="utf-8")) if retry.exists() else {}
            remaining = state.get("next_attempt", 0) - time.time()
            if remaining > 0:
                delay = min(delay, remaining)
                continue
            try:
                manifest = json.loads(marker.read_text(encoding="utf-8"))
                self.send(marker, manifest)
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                self._defer(marker, state, permanent=400 <= code < 500 and code not in (408, 425, 429))
            except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException):
                self._defer(marker, state)
            else:
                if not marker.exists():
                    retry.unlink(missing_ok=True)
            if retry.exists():
                next_attempt = json.loads(retry.read_text(encoding="utf-8"))["next_attempt"]
                delay = min(delay, max(0, next_attempt - time.time()))
            if self.wake.is_set():
                return 0  # A newly finished run preempts older retries.
        return delay

    def send(self, marker, manifest):
        run_id = manifest["run_id"]
        destination = self.directory / "uploads" / f"{run_id}.zip"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".zip.tmp")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
                archive.write(self.directory / manifest["data_file"], "observations.jsonl")
                tail = (self.directory / manifest["data_file"]).with_suffix(".interrupted-tail")
                if tail.exists():
                    archive.write(tail, "interrupted-tail.txt")
                archive.writestr("manifest.json", json.dumps(dict(manifest,
                    format="stella-tower-observations-v2", consent_source="MFA.HelpImproveSoftware",
                    consent_policy=self.settings["policy_version"],
                    images_included=False), ensure_ascii=False))
            temporary.replace(destination)
        raw = destination.read_bytes()
        headers = {"Content-Type": "application/zip", "X-Upload-ID": run_id,
                   "X-Content-SHA256": hashlib.sha256(raw).hexdigest()}
        if self.settings.get("access_token"):
            headers["Authorization"] = "Bearer " + self.settings["access_token"]
        request = urllib.request.Request(self.settings["endpoint"], data=raw, headers=headers, method="POST")
        if not self.upload_allowed():
            return
        # Acknowledgement is any HTTP 2xx; the receiver deduplicates X-Upload-ID.
        with urllib.request.urlopen(request, timeout=max(60, min(180, 20 + len(raw) * 8 / 3_400_000 * 3))) as response:
            if not 200 <= response.status < 300:
                raise OSError(f"接收端返回 HTTP {response.status}")
        write_json(marker, dict(manifest, uploaded_at=timestamp(), upload_id=run_id))
        sent = self.directory / "sent" / f"{run_id}.json"
        sent.parent.mkdir(parents=True, exist_ok=True)
        marker.replace(sent)
