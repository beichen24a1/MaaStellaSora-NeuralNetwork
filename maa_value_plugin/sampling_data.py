"""Run-scoped, image-free observations. Model beliefs never become game facts."""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import json
import os
import threading
import sysconfig
import uuid

from .sampling_evidence import ObservedInventory, ReadEvidence, raw_ocr, recognition_roi
from .name_matching import clean_name, match_name


VERSION = "maa-run-observations-20261105"
SCHEMA_VERSION = 3
SKIP_FIELDS = {"image", "raw_image", "draw_images", "layout", "potential_layouts"}
LINK_FIELDS = {"operation_id", "origin_operation_id", "offer_id", "reward_chain_id",
               "stock_scan_id", "shop_visit_id", "event_id", "source_type"}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def snapshot(value):
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.name
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith("_") and f.name not in SKIP_FIELDS}
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items() if str(k) not in SKIP_FIELDS}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: snapshot(v) for k, v in vars(value).items()
                if not k.startswith("_") and k not in SKIP_FIELDS}
    if hasattr(value, "__slots__"):
        return {k: snapshot(getattr(value, k)) for k in value.__slots__
                if not k.startswith("_") and k not in SKIP_FIELDS and hasattr(value, k)}
    return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_alive(pid):
    if pid == os.getpid():
        return True
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5  # An elevated, running Maa may deny access.
    try:
        code = ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


class SamplingCollector:
    def __init__(self, app_dir, logger, module_root=None, owned_provider=None, counter_provider=None):
        self.app_dir, self.logger = Path(app_dir), logger
        self.module_root = Path(module_root) if module_root else None
        config = self.app_dir / "agent/agent_config/tower_sampling.json"
        try:
            self.config = json.loads(config.read_text(encoding="utf-8-sig")) if config.exists() else {}
            if not isinstance(self.config, dict):
                raise ValueError("Sampling config must be a JSON object")
        except (OSError, ValueError) as error:
            self.logger.warning("Could not load sampling config: %s", error)
            self.config = {"enabled": False}
        self.enabled = self.config.get("enabled", True)
        self.directory = self.app_dir / "debug/tower_sampling"
        self.session_id = uuid.uuid4().hex
        self.task_id = None
        self.sequence = 0
        self.lock = threading.RLock()
        self.owned_provider = owned_provider or (lambda: ())
        self.counter_provider = counter_provider or (lambda: {})
        self._read = ContextVar("tower_sampling_read", default=None)
        self._operations = []
        self.inventory = ObservedInventory()
        self.offer_index = 0
        self.last_wallet = None
        self.tracker = None
        self.preset = None
        self.preset_id = None
        self.target_vector = None
        self.names = defaultdict(list)
        self.local_names = defaultdict(list)
        self.language = "unknown"
        self.run = None
        self.reward = None
        self.pending_selection = None
        self.start_pending = False
        self.settlement_seen = False
        self.settlement_result = None
        self.event = None
        self.event_question = ""
        self.stock_scan_id = None
        self.upload = None
        self.runtime_versions = {}
        self.participant_id = None

    def context(self, context):
        from .tower_sampling_collector import SamplingContext
        return context if not self.enabled or isinstance(context, SamplingContext) else SamplingContext(context, self)

    def loaded(self):
        if not self.enabled:
            return
        try:
            self._load_collector()
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.enabled = False
            self.logger.warning("Could not initialize sampling: %s", error)

    def _load_collector(self):
        legacy_identity = self.directory / "collector_identity.json"
        identity = self.app_dir / "config/neural_network/collector_identity.json"
        if not identity.exists():
            value = (json.loads(legacy_identity.read_text(encoding="utf-8-sig")) if legacy_identity.exists()
                     else {"participant_id": uuid.uuid4().hex, "created_at": timestamp()})
            write_json(identity, value)
        self.participant_id = json.loads(identity.read_text(encoding="utf-8-sig"))["participant_id"]
        write_json(legacy_identity, json.loads(identity.read_text(encoding="utf-8-sig")))
        self.recover_interrupted()
        interface = self.app_dir / "interface.json"
        if interface.exists():
            # Only public software versions, never the app's instance/account settings.
            from .installer import read_jsonc
            data = read_jsonc(interface)
            self.runtime_versions = {key: data[key] for key in ("version", "interface_version") if key in data}
        self.runtime_versions["python_platform"] = sysconfig.get_platform()
        self.emit("collector_loaded", supported_languages=["cn", "jp"],
                  capture="all visible tower OCR and actions; local UI boundaries; no images or server IDs")
        self.logger.info("[OCR采集] 已启用完整对局记录")
        if self.module_root:
            from .sample_upload import SampleUpload
            self.upload = SampleUpload(self.app_dir, self.module_root, self.directory, self.logger)
            self.upload.start()

    def _safe(self, value):
        if isinstance(value, str):
            for path, replacement in ((self.app_dir, "$APP"), (Path.home(), "$USER")):
                value = value.replace(str(path), replacement).replace(path.as_posix(), replacement)
            return value
        if isinstance(value, dict):
            return {k: self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        return value

    def emit(self, event_type, **payload):
        if not self.enabled:
            return
        try:
            with self.lock:
                sequence = self.sequence + 1
                run_id = self.run["run_id"] if self.run else None
                run_sequence = self.run["record_count"] + 1 if self.run else None
                row = dict(schema_version=SCHEMA_VERSION, collector_version=VERSION, event_type=event_type,
                           record_type=event_type, timestamp=timestamp(), record_id=uuid.uuid4().hex,
                           participant_id=self.participant_id, session_id=self.session_id,
                           task_id=self.task_id, run_id=run_id, exploration_id=run_id,
                           sequence=sequence, run_sequence=run_sequence,
                           lang_type=self.language, observer="maa_local_UI_and_cached_OCR",
                           server_confirmed=False, preset_id=self.preset_id,
                           target_vector_complete=self.target_vector is not None,
                           target_vector=self.target_vector)
                row.update(payload)
                folder = self.directory / ("runs" if run_id else "sessions")
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / f"{run_id or self.session_id}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(self._safe(snapshot(row)), ensure_ascii=False) + "\n")
                self.sequence = sequence
                if self.run:
                    self.run["record_count"] = run_sequence
                return row["record_id"]
        except (OSError, TypeError, ValueError) as error:
            self.logger.warning("[OCR采集] 记录失败：%s", error)

    def new_read(self, kind, **links):
        parent = self._read.get()
        with self.lock:
            inherited = dict(self._operations[-1]) if self._operations else {}
        if parent:
            inherited.update(parent.links)
        inherited.update({key: value for key, value in links.items() if key in LINK_FIELDS})
        return ReadEvidence(kind, inherited)

    @property
    def current_read(self):
        return self._read.get()

    @contextmanager
    def action_scope(self, **links):
        # Nested Maa run_task callbacks can execute on another Python thread.
        # Keep their causal action on this collector, not in official classes.
        links = {key: value for key, value in links.items() if key in LINK_FIELDS}
        with self.lock:
            self._operations.append(links)
        try:
            yield
        finally:
            with self.lock:
                self._operations.remove(links)

    @contextmanager
    def capture(self, kind=None, *, read=None, **links):
        read = read if read is not None else self.new_read(kind, **links)
        token = self._read.set(read)
        try:
            yield read
        finally:
            self._read.reset(token)

    def record_recognition(self, entry, args, kwargs, result, language):
        read = self._read.get()
        links = dict(read_id=read.read_id, read_kind=read.kind, **read.links) if read else {}
        raw = snapshot(result)
        roi = recognition_roi(entry, args, kwargs)
        record_id = self.emit("context_run_recognition", phase="after", node_name=entry,
                              lang_type=language, method="run_recognition", roi=roi,
                              arguments=snapshot(args), keyword_arguments=snapshot(kwargs),
                              raw_result=raw, **links)
        if read:
            read.recognitions.append(dict(record_id=record_id, node_name=entry, roi=roi, raw_result=raw))
        if entry == "星塔_通用_识别当前金币_agent" and raw and raw.get("filtered_results"):
            values = [r.get("text", "") for r in raw["filtered_results"]]
            text = values[0]
            coin = int(text) if text.isdecimal() else None
            wallet_id = self.emit("wallet_observed", node_name=entry, values=values, coin=coin,
                preceding_event_id=self.event["event_id"] if self.event else None,
                recognition_record_id=record_id, raw_result=raw, **links)
            self.last_wallet = dict(coin=coin, record_id=wallet_id, recognition_record_id=record_id)

    def owned_fields(self):
        return dict(self.inventory.fields(), maa_owned_cache=self.owned(), maa_owned_complete=False)

    def counters(self):
        return dict(snapshot(self.counter_provider()),
                    basis="Maa_inferred_counters_before_policy_choice_not_server_counters")

    def begin_task(self, language, strategy):
        if not self.enabled:
            return
        self.finish_at_boundary("task_restarted")
        self.task_id = uuid.uuid4().hex
        self.language = language
        self.tracker = self.preset = self.preset_id = self.target_vector = None
        self.names.clear()
        self.local_names.clear()
        self.start_pending = False
        self.reward = self.pending_selection = self.event = None
        self.emit("task_start", strategy=strategy)

    def bind_preset(self, preset, tracker):
        if not self.enabled:
            return
        self.preset, self.tracker = snapshot(preset), tracker
        self.target_vector = [level for pool in preset["pools"] for level in pool]
        identities = []
        self.names.clear()
        self.local_names.clear()
        for member in preset["party"]:
            for target in member["targets"]:
                identities.append(dict(target, role=member["role"], character_id=member["character_id"]))
        identities.extend(preset.get("recognition_catalog", []))
        for item in identities:
            for field in ("name_ja_jp", "name_zh_cn", "name_en_us", "name_zh_tw"):
                if item.get(field):
                    key = clean_name(item[field])
                    if item not in self.names[key]:
                        self.names[key].append(item)
                    if field == ("name_zh_cn" if self.language == "cn" else "name_ja_jp"):
                        self.local_names[key].append(item)
        key = [{"role": m["role"], "character_id": m["character_id"], "targets": m["targets"]}
               for m in preset["party"]]
        self.preset_id = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        self.emit("preset_observed", original_preset=self.preset, source="in_game_current_preset_OCR",
                  model_versions=snapshot(tracker.model.versions))

    def start_intent(self, node):
        if not self.enabled:
            return
        self.finish_at_boundary("new_game_started")
        self.start_pending = True
        self.emit("exploration_start_attempt", node_name=node)

    def ensure_run(self, evidence):
        if not self.enabled:
            return
        with self.lock:
            if not self.run:
                self._start_run(evidence)

    def _start_run(self, evidence):
        run_id = uuid.uuid4().hex
        if self.tracker:
            self.tracker.run_id = run_id
        self.run = dict(run_id=run_id, session_id=self.session_id, participant_id=self.participant_id,
                        pid=os.getpid(), started_at=timestamp(), record_count=0,
                        full_run_start_observed=self.start_pending, language=self.language,
                        unresolved_candidate_count=0)
        self.inventory = ObservedInventory(self.start_pending)
        self.offer_index = 0
        self.last_wallet = None
        self.reward = self.pending_selection = self.event = None
        self.stock_scan_id = None
        self.settlement_seen = False
        self.settlement_result = None
        self.start_pending = False
        try:
            write_json(self.directory / "queue" / f"{run_id}.open.json", self.run)
        except OSError as error:
            self.logger.warning("Could not open sampling run marker: %s", error)
        self.emit("exploration_start", full_run_start_observed=self.run["full_run_start_observed"],
                  resumed=not self.run["full_run_start_observed"], boundary_evidence=evidence,
                  original_preset=self.preset, runtime_versions=self.runtime_versions,
                  model_versions=snapshot(self.tracker.model.versions) if self.tracker else None,
                  **self.owned_fields())

    def owned(self):
        return [self.potential(p) for p in self.owned_provider()]

    def potential(self, potential):
        row = snapshot(potential)
        role = getattr(getattr(potential, "trekker", None), "index", -1) + 1
        resolution = match_name(row.get("name", ""), self.names, role=role or None,
                                rarity=row.get("rarity"), preferred=self.local_names)
        matches = resolution.items
        item = matches[0] if len(matches) == 1 else None
        resolved_role = item["role"] if item else role or None
        core = row.get("rarity") == "core"
        before, after = row.get("old_level"), row.get("new_level")
        row.update(potential_id=item["potential_id"] if item else None,
                   character_id=item["character_id"] if item else None,
                   role=resolved_role,
                   target_level=item["target_level"] if item else None,
                   identity_basis=(resolution.method + ("_and_role" if role else "_unique_in_party")) if item else "unresolved",
                   identity_name=(item.get("name_zh_cn" if self.language == "cn" else "name_ja_jp") if item else None),
                   identity_similarity=resolution.similarity,
                   identity_inferred=bool(item and resolution.method == "closest_name"),
                   id_candidates=[m["potential_id"] for m in matches],
                   canonical_rarity=item["rarity_class"] if item else None,
                   trekker_index=resolved_role-1 if resolved_role else None, core=core,
                   level_before=before, offered_level_after=after,
                   level_after=min(after, 1 if core else 6) if after is not None and after >= 0 else None,
                   level_delta=after-before if after is not None and before is not None else None,
                   is_new=before == 0 if before is not None else None,
                   level_basis="Maa_core_0_to_1_convention" if core else "card_level_OCR")
        return row

    def inferred_state(self):
        if not self.tracker:
            return None
        return dict(floor=self.tracker.state.floor, node=self.tracker.state.node,
                    phase=self.tracker.state.phase, source="neural_network_route_inference",
                    observation_not_game_truth=True)

    def finish_at_boundary(self, reason):
        self.finish_run("settlement_saved" if self.settlement_seen else reason,
                        completion_observed=self.settlement_seen, settlement=self.settlement_result)

    def finish_run(self, reason, *, completion_observed, settlement=None):
        if not self.enabled or not self.run:
            return
        with self.lock:
            if not self.run:
                return
            if self.pending_selection:
                self.inventory.uncertain("take_not_confirmed", offer_id=self.pending_selection["offer_id"])
            end_record = self.emit("exploration_end", completion_observed=completion_observed, reason=reason,
                      final_owned=self.inventory.fields()["owned"], **self.owned_fields(),
                      settlement=settlement, tracker_estimate=self.inferred_state(),
                      last_wallet_observation=self.last_wallet,
                      pending_selection=snapshot(self.pending_selection),
                      full_run_start_observed=self.run["full_run_start_observed"])
            manifest = dict(self.run, ended_at=timestamp(), reason=reason,
                            completion_observed=completion_observed,
                            complete=self.run["full_run_start_observed"] and completion_observed,
                            owned_history_complete=self.inventory.complete,
                            target_vector_complete=self.target_vector is not None, preset_id=self.preset_id,
                            data_file=f"runs/{self.run['run_id']}.jsonl", collector_version=VERSION)
            marker = self.directory / "queue" / f"{self.run['run_id']}.open.json"
            if end_record:
                try:
                    write_json(marker, manifest)
                    marker.replace(marker.with_name(f"{self.run['run_id']}.ready.json"))
                except OSError as error:
                    self.logger.warning("Could not finalize sampling run: %s", error)
            self.run = None
            self.reward = self.pending_selection = self.event = None
        if self.upload:
            self.upload.wake.set()

    def recover_interrupted(self):
        for marker in (self.directory / "queue").glob("*.open.json"):
            try:
                self._recover_marker(marker)
            except (OSError, KeyError, TypeError, ValueError) as error:
                self.logger.warning("Could not recover sampling run %s: %s", marker.name, error)

    def _recover_marker(self, marker):
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        run_id = manifest["run_id"]
        if uuid.UUID(run_id).hex != run_id or marker.name != f"{run_id}.open.json":
            raise ValueError("Invalid sampling run ID")
        if process_alive(manifest["pid"]):
            return
        path = self.directory / "runs" / f"{manifest['run_id']}.jsonl"
        if not path.exists():
            # The open marker is written before the first observation.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        count, last = 0, {}
        with path.open("rb+") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    # Keep an interrupted byte tail as evidence, outside the JSONL stream.
                    tail = path.with_suffix(".interrupted-tail")
                    tail.write_bytes(line)
                    stream.seek(-len(line), 1)
                    stream.truncate()
                    break
                last = json.loads(line)
                count += 1
            has_end = last.get("event_type") == "exploration_end"
            completion = has_end and last.get("completion_observed", False)
            reason = last.get("reason") if has_end else "process_ended_without_settlement"
            if not has_end:
                row = {key: last.get(key) for key in ("task_id", "lang_type", "preset_id",
                       "target_vector_complete", "target_vector")}
                row.update(schema_version=SCHEMA_VERSION, collector_version=VERSION,
                           event_type="exploration_end", record_type="exploration_end", record_id=uuid.uuid4().hex,
                           run_id=manifest["run_id"], exploration_id=manifest["run_id"],
                           session_id=manifest["session_id"], participant_id=manifest["participant_id"],
                           sequence=last.get("sequence", 0) + 1, run_sequence=count + 1,
                           timestamp=timestamp(), reason=reason, observer="local_log_recovery",
                           completion_observed=False, boundary_recovered=True, server_confirmed=False)
                stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        manifest.update(data_file=f"runs/{manifest['run_id']}.jsonl",
                        complete=manifest["full_run_start_observed"] and completion,
                        completion_observed=completion, record_count=count + (not has_end),
                        target_vector_complete=last.get("target_vector_complete", False),
                        preset_id=last.get("preset_id"), collector_version=VERSION,
                        reason=reason, recovered_at=timestamp(),
                        ended_at=last.get("timestamp", manifest["started_at"]))
        write_json(marker, manifest)
        marker.replace(marker.with_name(f"{manifest['run_id']}.ready.json"))

    def reward_chain(self, source):
        self.ensure_run("potential_page")
        if not self.reward or self.reward["closed"]:
            read = self._read.get()
            self.reward = dict(reward_chain_id=uuid.uuid4().hex, source=source, pages=[],
                               page=None, refresh_operation_id=None, closed=False,
                               origin_operation_id=read.links.get("operation_id") if read else None)
        return self.reward

    def prepare_offer(self, handler):
        chain = self.reward_chain(handler.data.params.potential_source)
        if chain["page"] is None:
            self._new_offer_page(chain)
        return chain["page"]

    def _new_offer_page(self, chain):
        self.offer_index += 1
        previous = chain["pages"][-1] if chain["pages"] else None
        page = dict(offer_id=uuid.uuid4().hex, offer_index=self.offer_index,
                    refresh_index=len(chain["pages"]),
                    refresh_parent_offer_id=previous["offer_id"] if previous else None,
                    refresh_operation_id=chain["refresh_operation_id"],
                    observation_count=0, fingerprint=None)
        chain["pages"].append(page)
        chain["page"] = page
        return page

    def observe_offer(self, handler, read=None):
        data = handler.data
        chain = self.reward_chain(data.params.potential_source)
        page = self.prepare_offer(handler)
        repeated = page["observation_count"] > 0
        offer_id = page["offer_id"]
        candidates = [self.potential(p) for p in data.potentials]
        read = read if read is not None else self.new_read("potential_page", offer_id=offer_id)
        names = read.matching("星塔_节点_选择潜能_识别潜能名称_agent")
        levels = read.matching("星塔_节点_选择潜能_识别潜能等级_agent")
        for slot, p in enumerate(candidates):
            name = names[slot:slot+1]
            level = levels[slot:slot+1]
            errors = []
            if p["potential_id"] is None:
                errors.append("unresolved_identity")
            if p["level_before"] < 0 or p["offered_level_after"] < 1:
                errors.append("level_not_read")
            if p["canonical_rarity"] is not None and p["rarity"] != p["canonical_rarity"]:
                errors.append("rarity_catalog_conflict")
            p.update(slot=slot, initial_level=p["offered_level_after"] if p["is_new"] else None,
                     server_raw_increment=None, raw_ocr=dict(name=raw_ocr(name), level=raw_ocr(level)),
                     name_roi=name[0]["roi"] if name else None,
                     level_roi=level[0]["roi"] if level else None,
                     recognition_record_ids=[r["record_id"] for r in name+level],
                     recognition_status="ok" if not errors else "unresolved",
                     recognition_errors=errors)
        unresolved = sum(p["recognition_status"] != "ok" for p in candidates)
        fingerprint = [(p["potential_id"] or p["name"], p["role"], p["rarity"],
                        p["level_before"], p["offered_level_after"]) for p in candidates]
        consistent = not repeated or page["fingerprint"] == fingerprint
        if not repeated:
            page["fingerprint"] = fingerprint
        page["observation_count"] += 1
        page["candidates"] = candidates
        conflicts = self.inventory.audit(candidates, offer_id)
        if not repeated:
            self.run["unresolved_candidate_count"] += unresolved
        source_raw = data.params.potential_source
        source = ("core" if data.core_potential else "level_up" if data.level_upped
                  else source_raw if source_raw != "default" else "unknown")
        self.emit("offer_reobserved" if repeated else "offer_observed", offer_id=offer_id,
                  offer_index=page["offer_index"], reward_chain_id=chain["reward_chain_id"],
                  refresh_index=page["refresh_index"], refresh_parent_offer_id=page["refresh_parent_offer_id"],
                  refresh_operation_id=page["refresh_operation_id"], maa_refresh_count=data.refresh_count,
                  page_generation_basis="successful_refresh" if page["refresh_index"] else "new_reward_after_confirmed_take_or_run_start",
                  same_page_read_index=page["observation_count"], recognition_consistent=consistent,
                  read_id=read.read_id, recognition_record_ids=read.record_ids,
                  origin_operation_id=chain["origin_operation_id"],
                  source=source, source_raw=source_raw, source_type=source_raw, source_attribution="Maa_attach_and_banner",
                  level_upped=data.level_upped, core_potential=data.core_potential,
                  candidates=candidates, candidate_read_complete=unresolved == 0,
                  candidates_complete=bool(candidates) and unresolved == 0,
                  recognition_status="ok" if not unresolved and consistent else "unresolved",
                  raw_rarity=list(data.potential_rarities),
                  level_basis="Maa_core_0_to_1_convention" if data.core_potential else "card_level_OCR",
                  inventory_conflicts=conflicts, **self.owned_fields(), counters=self.counters(),
                  main_trekker_index=0, trekker_index_basis="game_preset_party_roles",
                  coin=data.current_coin, refresh_price=data.refresh_cost, floor=None,
                  state=snapshot(data), tracker_estimate=self.inferred_state(),
                  preceding_event_id=self.event["event_id"] if self.event else None,
                  preceding_event_relation="last_observed_event_not_reward_attribution")
        return offer_id

    def cache_selection(self, potential, offer_id, operation_id):
        page = self.reward["page"]
        selected = dict(next((p for p in page["candidates"] if p["slot"] == potential.index), {}),
                        **self.potential(potential))
        self.pending_selection = dict(offer_id=offer_id, operation_id=operation_id,
            reward_chain_id=self.reward["reward_chain_id"], selected=selected,
            source_type=self.reward["source"])
        self.emit("selection_cache", **self.pending_selection, **self.owned_fields(),
                  operation_confirmed=False, cache_timing="before_final_take_button_confirmation")

    def selection_confirmed(self, evidence):
        if not self.pending_selection:
            return
        record_id = uuid.uuid4().hex
        previous = self.inventory.fields()
        self.inventory.confirm(self.pending_selection["selected"], record_id)
        self.emit("selection_ui_transition_observed", **self.pending_selection,
                  record_id=record_id, owned_before=previous["owned"],
                  owned_before_complete=previous["owned_complete"],
                  operation_confirmed=True, confirmation_basis=evidence,
                  observed_state_after=self.owned(), **self.owned_fields())
        self.pending_selection = None
        if self.reward:
            self.reward["closed"] = True

    def refresh_succeeded(self, offer_id, operation_id):
        chain = self.reward
        if chain and chain["page"] and chain["page"]["offer_id"] == offer_id:
            chain["refresh_operation_id"] = operation_id
            return self._new_offer_page(chain)["offer_id"]

    def model_decision(self, record):
        self.ensure_run("model_decision")
        # The preset and software versions are already in this run's header.
        reduced = {k: v for k, v in record.items()
                   if k not in {"original_preset", "versions", "run_id"}}
        chain = self.reward if self.reward and not self.reward["closed"] else None
        phase = record.get("phase")
        self.emit("model_decision", reward_chain_id=chain["reward_chain_id"] if chain else None,
                  offer_id=chain["page"]["offer_id"] if chain and chain["page"] else None,
                  event_id=self.event["event_id"] if phase == "EVENT" and self.event else None,
                  stock_scan_id=self.stock_scan_id if phase == "SHOP" else None, decision=reduced)

    def event_menu(self, choices):
        self.ensure_run("event_menu")
        menu = dict(question=self.event_question, choices=snapshot(choices))
        signature = json.dumps(menu, sort_keys=True, ensure_ascii=False)
        repeated = self.event and self.event["signature"] == signature and not self.event["clicked"]
        if not repeated:
            self.event = dict(event_id=uuid.uuid4().hex, signature=signature, clicked=False, **menu)
        self.emit("event_menu_reobserved" if repeated else "event_menu_observed",
                  event_id=self.event["event_id"], **menu, **self.owned_fields(),
                  tracker_estimate=self.inferred_state())

    def event_choice(self, result, attach):
        if self.event is None or result.box is None:
            return
        self.emit("event_choice", event_id=self.event["event_id"],
                  question=attach.get("last_question"), selected_text=attach.get("last_choice"),
                  displayed_consequence=attach.get("last_consequence"),
                  selected_box=snapshot(result.box), model_mapping=snapshot(result.detail),
                  operation_confirmed=False)

    def event_clicked(self, evidence):
        if self.event is None:
            return
        self.event["clicked"] = True
        self.emit("event_choice_click_observed", event_id=self.event["event_id"],
                  evidence=evidence, server_confirmed=False)

    def operation(self, event, handler, function, *args, before=False, offer_id=None, operation_id=None):
        if not self.enabled:
            return function(*args)
        self.ensure_run(event)
        data = handler.data
        initial = snapshot(data)
        common = dict(operation_id=operation_id or uuid.uuid4().hex, offer_id=offer_id,
                      reward_chain_id=self.reward["reward_chain_id"] if self.reward and not self.reward["closed"] else None,
                      stock_scan_id=self.stock_scan_id, arguments=snapshot(args),
                      source_type=getattr(data.params, "potential_source", None))
        if before:
            self.emit(event + "_attempt", phase="before", state=initial, **self.owned_fields(), **common)
        with self.action_scope(**common), self.capture(event, **common) as read:
            result = function(*args)
        next_offer_id = (self.refresh_succeeded(offer_id, common["operation_id"])
                         if event == "refresh_result" and result else None)
        self.emit(event, phase="after", state=snapshot(data), previous_state=initial,
                  result=snapshot(result), Maa_reported_success=result if isinstance(result, bool) else None,
                  next_offer_id=next_offer_id,
                  coin_before=initial.get("current_coin"),
                  coin_after=getattr(data, "current_coin", None),
                  read_id=read.read_id, recognition_record_ids=read.record_ids,
                  **self.owned_fields(), **common)
        return result

    def observe_shop(self, handler, read, previous_state):
        data = handler.data
        remaining = data.refresh_remaining
        repeated = remaining in handler.stock_generations
        previous_generation = handler.stock_generation_id
        generation = handler.stock_generations.setdefault(remaining, uuid.uuid4().hex)
        handler.stock_generation_id = generation
        items = snapshot(data.items)
        by_slot = {p["grid_num"]-1: p for p in items}
        slots = []
        for slot, rois in enumerate(data.params.item_rois):
            name = read.matching("星塔_节点_商店_购物_识别物品内容_agent", rois["name_roi"])
            price = read.matching("星塔_节点_商店_购物_识别物品价格_agent", rois["price_roi"])
            item = by_slot.get(slot)
            slots.append(dict(slot=slot, **snapshot(rois),
                item_instance_id=f"{generation}:{slot}", item=item,
                effective_price=item["price"] if item else None,
                raw_ocr=dict(name=raw_ocr(name), price=raw_ocr(price)),
                recognition_record_ids=[r["record_id"] for r in name+price]))
        self.emit("shop_observed", phase="after", shop_event_type="stock_snapshot",
                  stock_scan_id=self.stock_scan_id, shop_visit_id=handler.shop_visit_id,
                  shop_entry_id=handler.shop_visit_id, stock_generation_id=generation,
                  shelf_generation_id=generation, previous_shelf_generation_id=previous_generation,
                  shelf_generation_basis="same_visit_and_visible_remaining_refresh_count",
                  repeated_scan=repeated, refresh_remaining=remaining, stock_snapshot=items,
                  payload=dict(slots=slots, maa_items=items), state=snapshot(data),
                  previous_state=previous_state, coin=data.current_coin,
                  refresh_price=data.refresh_cost, read_id=read.read_id,
                  recognition_record_ids=read.record_ids, **self.owned_fields())
