"""Export a frozen, image-free tower sample bundle using Python's standard library.

This program never starts Maa, acts on the game, accesses the network, or deletes
the source data. Run from Maa/tools, or pass --root for an offline fixture.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile


VERSION = '1.0.0-20261021'
JST = timezone(timedelta(hours=9))
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
NOTE_NAMES = frozenset(('preset_notes.txt', 'preset_notes.md', 'preset_notes.json',
                        'tower_sampling_preset_notes.txt', 'tower_sampling_preset_notes.md',
                        'tower_sampling_preset_notes.json'))
CONFIG_PATHS = ('agent/agent_config/tower_sampling.json',
                'agent/agent_config/tower_sampling_catalog.json')

README = '''星塔采样数据包（不含图片）

raw/ 保存采样器原始 JSONL 的完整行，未知字段原样保留。
metadata/ 保存采集配置、匿名采集ID、目录、进度及可选预设备注。
BUNDLE_INFO.json 记录冻结时间、源文件大小、排除的未完行字节和计数口径。
MANIFEST.json、SHA256SUMS.txt 用于检查包内文件完整性。

本包不含候选截图、预设图片、Maa账户/实例配置、游戏账号信息文件或全量Maa日志。
原始采样记录中的字段与路径字符串原样保留；引用截图的路径不代表截图已装入本包。

计数说明：
1. JSONL 行数、候选屏、选择尝试、实际界面流转分别计数，不等于独立样本数。
2. run_start / run_end 是旧采集器的 Maa 状态重置分组，不是真实完整探索。
3. exploration_start / exploration_end 是本地 UI 局界标记，不是服务器对局ID。
   只有看见完整开始、结算并返回菜单的同一 exploration_id 才列入完整结束数。
4. shop_observed 的 stock_snapshot 是观察到的货架；购买尝试、Maa返回成功、
   可见强化额外等级分别保留，不能代替服务器结算或真实 LuckyIds。
5. 所有统计只是导出清点，没有估计概率、判断策略好坏或过滤未选候选。

新版 schema_version=3 在每局 exploration_start 中保存游戏读取的完整预设，含零目标。
候选、刷新、选择确认、商店、事件和神经网络决策均使用相同的 exploration_id。
完整开局与结算、接续半局、任务停止和进程中断分别标记；未解析候选保留未知。
候选页、刷新和拿走确认在采集时直接关联；导出不改写原始行或补配旧数据。
owned为界面确认取得的持有历史；Maa提前缓存另存maa_owned_cache。
神经网络推断的路线和持有历史不当作服务端真值。
旧版没有完整预设的记录，仍请把 ZIP 与本次实际使用的完整预设一起发回分析。
预设可另外发送截图、分享码或完整文本，并说明队伍、主位、客户端/官服、塔难度及
中途是否更换。日志中出现的推荐等级片段不等于完整预设；本包不会替你猜补预设。
如果没有更换预设，可明确说明整批都使用同一份预设。

文件是在 Maa 继续运行时按每个源文件当时已存在的字节前缀冻结的。
没有换行符的尾部暂不导出（即使暂时看似完整JSON），排除字节数已记录。
随后追加的新记录留在原文件中，下次双击导出即可带上。原始记录不会删除。

无需安装 Python、pip、Codex 或连接模型。采集与导出均不需要 agent 持续运行。
这里的 agent 指在线分析助手；Maa 自带的本地 Python 扩展随 Maa 正常执行。
'''


def timestamp():
    return datetime.now(JST).isoformat(timespec='milliseconds')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def inside(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('拒绝读取采集目录之外的路径：' + str(path))
    return resolved


def source_files(root, sampling):
    """Allowlist samples and metadata; never enumerate screenshot contents."""
    raw_paths = []
    metadata_paths = []
    for path in sampling.rglob('*'):
        relative = path.relative_to(sampling)
        # Export files must never be recursively included in future exports.
        if 'exports' in relative.parts or 'screenshots' in relative.parts:
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() == '.jsonl':
            raw_paths.append(inside(path, sampling))
        elif (path.name == 'progress.json' or path.name in NOTE_NAMES
              or relative.parts[0] in ('queue', 'sent') and path.suffix == '.json'
              or path.suffix == '.interrupted-tail'):
            metadata_paths.append(inside(path, sampling))
    for name in CONFIG_PATHS:
        path = root / name
        if path.is_file():
            metadata_paths.append(inside(path, root))
    for directory in (root, root/'agent/agent_config'):
        for name in NOTE_NAMES:
            path = directory / name
            if path.is_file():
                metadata_paths.append(inside(path, root))
    return sorted(set(raw_paths)), sorted(set(metadata_paths))


def collector_identity(sampling):
    """Use the recorder's anonymous ID; create it only if no recorder made one."""
    path = sampling/'collector_identity.json'
    inside(path, sampling)
    if not path.exists():
        try:
            with path.open('x', encoding='utf-8', newline='\n') as file:
                json.dump({'participant_id': uuid.uuid4().hex,
                           'created_at': timestamp(), 'created_by': 'offline_export_tool'},
                          file, ensure_ascii=False)
                file.write('\n')
        except FileExistsError:
            pass
    raw = path.read_bytes()
    obj = json.loads(raw.decode('utf-8-sig'))
    value = obj.get('participant_id')
    if not isinstance(value, str):
        raise ValueError('collector_identity.json 缺少有效的匿名 participant_id；原始记录未修改。')
    try:
        anonymous_id = uuid.UUID(value).hex
    except (ValueError, AttributeError) as exc:
        raise ValueError('collector_identity.json 的 participant_id 不是UUID；原始记录未修改。') from exc
    return anonymous_id, raw, path


class Counts:
    def __init__(self):
        self.event_types = Counter()
        self.shop_types = Counter()
        self.sources = Counter()
        self.schema_versions = Counter()
        self.record_ids = set()
        self.duplicate_record_id_rows = 0
        self.offer_ids = set()
        self.reward_ids = set()
        self.legacy_groups = set()
        self.ui_explorations = set()
        self.started_explorations = set()
        self.settled_explorations = set()
        self.stock_scans = set()
        self.invalid_json_lines = 0
        self.non_object_lines = 0
        self.rows = 0

    def add(self, raw_line):
        self.rows += 1
        try:
            row = json.loads(raw_line.decode('utf-8-sig'))
        except (ValueError, UnicodeError):
            self.invalid_json_lines += 1
            return False
        if not isinstance(row, dict):
            self.non_object_lines += 1
            return False
        event = str(row.get('event_type', 'missing_event_type'))
        self.event_types[event] += 1
        self.schema_versions[str(row.get('schema_version', 'missing'))] += 1
        rid = row.get('record_id')
        if isinstance(rid, str):
            if rid in self.record_ids:
                self.duplicate_record_id_rows += 1
            self.record_ids.add(rid)
        if row.get('run_id') is not None and str(row.get('schema_version', 1)) == '1':
            self.legacy_groups.add(str(row['run_id']))
        exploration = row.get('true_run_id') or row.get('exploration_id')
        if exploration is not None:
            self.ui_explorations.add(str(exploration))
        if event == 'exploration_start' and exploration is not None and row.get('full_run_start_observed', True):
            self.started_explorations.add(str(exploration))
        if event == 'exploration_end' and row.get('completion_observed') is True and exploration is not None:
            self.settled_explorations.add(str(exploration))
        if event == 'offer_observed':
            if row.get('offer_id') is not None:
                self.offer_ids.add(str(row['offer_id']))
            reward = row.get('reward_chain_id') or row.get('reward_id')
            if reward is not None:
                self.reward_ids.add(str(reward))
            self.sources[str(row.get('source_type', row.get('source', 'unknown')))] += 1
        if event == 'shop_observed':
            kind = str(row.get('shop_event_type', 'missing'))
            self.shop_types[kind] += 1
            if kind == 'stock_snapshot' and row.get('stock_scan_id') is not None:
                self.stock_scans.add(str(row['stock_scan_id']))
        return True

    def result(self):
        return {
            'raw_complete_lines': self.rows,
            'invalid_json_complete_lines': self.invalid_json_lines,
            'non_object_json_lines': self.non_object_lines,
            'event_rows_by_type': dict(sorted(self.event_types.items())),
            'schema_versions': dict(sorted(self.schema_versions.items())),
            'unique_record_ids': len(self.record_ids),
            'duplicate_record_id_rows': self.duplicate_record_id_rows,
            'unique_offer_ids': len(self.offer_ids),
            'unique_reward_chain_ids': len(self.reward_ids),
            'offer_rows_by_source': dict(sorted(self.sources.items())),
            'legacy_Maa_reset_groups_not_game_runs': len(self.legacy_groups),
            'observed_local_UI_exploration_ids': len(self.ui_explorations),
            'observed_UI_full_starts': len(self.started_explorations),
            'observed_UI_settlement_ends_including_resumed_runs': len(self.settled_explorations),
            'observed_UI_full_start_and_settlement_same_id':
                len(self.started_explorations & self.settled_explorations),
            'shop_event_rows_by_type': dict(sorted(self.shop_types.items())),
            'unique_stock_snapshot_scan_ids': len(self.stock_scans),
            'interpretation': '清点观察记录；不是独立样本数或服务器对局数，未计算概率。',
        }


def stream_digest(stream):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
        size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def copy_raw(archive, name, path, expected, counts):
    digest = hashlib.sha256()
    included = invalid = lines = 0
    remaining = expected
    with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as out:
        while remaining:
            line = source.readline(remaining)
            if not line:
                raise ValueError('Source was truncated while exporting: ' + path.name)
            remaining -= len(line)
            if not line.endswith(b'\n'):
                if remaining:
                    raise ValueError('Source was truncated while exporting: ' + path.name)
                break
            out.write(line)
            digest.update(line)
            included += len(line)
            lines += 1
            if not counts.add(line):
                invalid += 1
    return {'included_bytes': included, 'excluded_unterminated_tail_bytes': expected - included,
            'complete_lines': lines, 'invalid_or_non_object_complete_lines': invalid,
            'sha256_exported_prefix': digest.hexdigest()}


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('ZIP 内有重复文件名')
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC 完整性校验失败')
        manifest = json.loads(archive.read('MANIFEST.json').decode('utf-8'))
        for name, info in manifest['files'].items():
            with archive.open(name) as stream:
                actual = stream_digest(stream)
            if actual != info:
                raise ValueError('包内文件 SHA256 校验失败：' + name)
        for line in archive.read('SHA256SUMS.txt').decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            if manifest['files'].get(name, {}).get('sha256') != digest:
                raise ValueError('SHA256SUMS 校验失败：' + name)
        expected = set(manifest['files']) | {'MANIFEST.json', 'SHA256SUMS.txt'}
        if set(names) != expected:
            raise ValueError('ZIP 文件清单不匹配')
        return {'verified': True, 'archive_entries': len(names),
                'contains_images': any(Path(name).suffix.lower() in
                    {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'} for name in names)}


def export_bundle(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    sampling = root/'debug/tower_sampling'
    if not sampling.is_dir():
        raise ValueError('尚未找到采集目录。请先运行 Maa 的爬塔任务，再导出；原始记录未修改。')
    inside(sampling, root)
    raw_paths, metadata_paths = source_files(root, sampling)
    if not raw_paths:
        raise ValueError('尚未找到 JSONL 采样记录。请先运行爬塔任务；原始记录未修改。')
    frozen_at = timestamp()
    # Freeze each length once before reading; later appends are excluded.
    frozen_sizes = {path: path.stat().st_size for path in raw_paths}
    anonymous_id, identity_raw, identity_path = collector_identity(sampling)
    payloads = {'README.txt': README.encode('utf-8'),
                'metadata/debug/tower_sampling/collector_identity.json': identity_raw}
    schema_path = Path(__file__).with_name('observation_schema.json')
    if schema_path.is_file():
        payloads['SCHEMA.json'] = schema_path.read_bytes()
    counts = Counts()
    inputs = []
    metadata = []
    for path in metadata_paths:
        # Copy a metadata file once; an in-flight progress rewrite may be
        # observed as old or new. It never controls authoritative raw counts.
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        relative = path.relative_to(root).as_posix()
        name = 'metadata/' + relative
        payloads[name] = raw
        metadata.append({'archive_path': name, 'bytes': len(raw), 'sha256': sha256(raw)})
    output = sampling/'exports'
    inside(output, sampling)
    output.mkdir(exist_ok=True)
    stamp = datetime.now(JST).strftime('%Y%m%d_%H%M%S')
    destination = output/f'TowerSamples_{anonymous_id[:8]}_{stamp}.zip'
    if destination.exists():
        destination = output/f'TowerSamples_{anonymous_id[:8]}_{stamp}_{uuid.uuid4().hex[:6]}.zip'
    temporary = destination.with_name(destination.name + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            files = {}
            for path in raw_paths:
                expected = frozen_sizes[path]
                relative = path.relative_to(sampling).as_posix()
                name = 'raw/' + relative
                copied = copy_raw(archive, name, path, expected, counts)
                inputs.append({'source_path_relative_to_sampling': relative,
                               'archive_path': name, 'size_at_freeze_bytes': expected, **copied})
                files[name] = {'bytes': copied['included_bytes'], 'sha256': copied['sha256_exported_prefix']}
            info = {'format': 'stella-sora-tower-community-samples-v1',
                    'exporter_version': VERSION, 'participant_id': anonymous_id,
                    'frozen_at_JST': frozen_at, 'created_at_JST': timestamp(),
                    'freeze_policy': 'per_file_fixed_size_prefix_then_keep_newline_terminated_lines_only',
                    'source_root_recorded': False,
                    'raw_files': inputs, 'metadata_files': metadata,
                    'counts': counts.result(), 'includes_images': False,
                    'preset_completeness': 'schema_v2_and_v3_embed_original_full_preset_in_each_run_header; legacy_samples_may_need_external_preset',
                    'excluded': ['screenshots_and_all_images', 'Maa_account_or_instance_configuration',
                                 'full_Maa_logs', 'source_executable_files'],
                    'source_data_deleted': False}
            payloads['BUNDLE_INFO.json'] = encoded(info)
            for name, raw in sorted(payloads.items()):
                archive.writestr(name, raw)
                files[name] = {'bytes': len(raw), 'sha256': sha256(raw)}
            archive.writestr('MANIFEST.json', encoded({'format': 'sha256-manifest-v1', 'files': files}))
            archive.writestr('SHA256SUMS.txt', ''.join(
                info['sha256'] + '  ' + name + '\n' for name, info in sorted(files.items())))
        verification = verify_bundle(temporary)
        if verification['contains_images']:
            raise ValueError('导出包出现图片，拒绝交付')
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    with destination.open('rb') as stream:
        zip_digest = stream_digest(stream)
    return {'status': 'exported', 'zip_path': str(destination),
            'zip_bytes': zip_digest['bytes'], 'zip_sha256': zip_digest['sha256'],
            'participant_id': anonymous_id, 'counts': info['counts'],
            'excluded_unterminated_tail_bytes': sum(x['excluded_unterminated_tail_bytes'] for x in inputs),
            **verification, 'raw_data_deleted': False}


def main():
    parser = argparse.ArgumentParser(description='导出不含图片的星塔采样包；不删除原始数据。')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='仅供离线测试/迁移，默认自动定位 Maa 根目录')
    parser.add_argument('--verify', type=Path, help='只校验一个已导出的ZIP')
    args = parser.parse_args()
    try:
        result = verify_bundle(args.verify) if args.verify else export_bundle(args.root)
    except Exception as exc:
        print('导出/校验失败：' + str(exc), file=sys.stderr)
        print('原始采样记录未删除。修复问题后可再次导出。', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.verify:
        print('\n导出成功。新版记录已包含完整游戏预设；旧版缺少预设时请另外提供。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
