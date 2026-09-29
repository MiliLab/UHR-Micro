#!/usr/bin/env python3
"""观察门控 + 查询参照窗口证据路由 (fast-iteration implementation, 非冻结).

Design 1 (观察门控): 无补充证据的行 (no admitted deficit proposal) 跳过重观察与两段式
composer, native 阶段答案直通为最终答案 (流程在无证据工作时自我简化)。

Design 2 (查询参照窗口): 问题自带区域坐标的任务 (RD/OC/FGR/CS/RC), 无补充证据的行使用
查询区域原分辨率窗口的观察结果作为最终答案 (窗口观察 = 证据获取的查询参照候选视图)。

Usage:
  python observe_gate_runtime.py --canonical <map_full canonical> --native-results <work/native_results.jsonl> \
      --decisions <controller/deficit_decisions.jsonl> --identity <private_opaque_identity_map.jsonl> \
      --window-preds <qc predictions.jsonl> --mode {gate,gate_plus_window,window_only} --out <out.jsonl>
"""
import argparse
import json
import re
from pathlib import Path

QC_TASKS = {
    'Grounding/RegionDetection', 'Understanding/Classification',
    'Understanding/Fine-grainedRecognition', 'Understanding/ComponentSegmentation',
    'Count/RegionCount',
}


def task_of(rel: str) -> str:
    p = rel.split('/')
    return p[1] + '/' + p[2]


def convert_native(resp) -> str:
    """native 阶段 'Provisional evidence: ...' -> 'Final answer: ...'"""
    if resp is None:
        return 'Final answer: null'
    s = str(resp).strip()
    m = re.match(r'^Provisional evidence:\s*(.*)$', s, re.S)
    if not m:
        return s
    rest = m.group(1).strip()
    if not rest or rest.lower() == 'null':
        return 'Final answer: null'
    lm = re.fullmatch(r'([A-Da-d])', rest)
    if lm:
        return f'Final answer: {lm.group(1).upper()}'
    if re.fullmatch(r'[-+]?\d+(\.\d+)?', rest):
        return f'Final answer: {rest}'
    boxes = re.findall(
        r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        rest,
    )
    if boxes:
        return 'Final answer: [' + ', '.join(f'[{a},{b},{c},{d}]' for a, b, c, d in boxes) + ']'
    return f'Final answer: {rest}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', type=Path, required=True)
    ap.add_argument('--native-results', type=Path, required=True)
    ap.add_argument('--decisions', type=Path, required=True)
    ap.add_argument('--identity', type=Path, required=True)
    ap.add_argument('--window-preds', type=Path, required=True)
    ap.add_argument('--mode', choices=['gate', 'gate_plus_window', 'window_only'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    sid2rel = {}
    for l in open(args.identity):
        if not l.strip():
            continue
        d = json.loads(l)
        sid2rel[d['sample_id']] = d['qa_relpath']
    admitted = set()
    for l in open(args.decisions):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get('admitted') and d['sample_id'] in sid2rel:
            admitted.add(sid2rel[d['sample_id']])
    nat_stage = {}
    for l in open(args.native_results):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get('sample_id') in sid2rel:
            nat_stage[sid2rel[d['sample_id']]] = d.get('model_response')
    window = {}
    for l in open(args.window_preds):
        if not l.strip():
            continue
        d = json.loads(l)
        window[d['qa_relpath']] = d.get('final_response')

    stats = {'keep': 0, 'gate': 0, 'window': 0}
    with open(args.out, 'w') as out:
        for l in open(args.canonical):
            if not l.strip():
                continue
            row = json.loads(l)
            rel = row['qa_relpath']
            if rel in admitted:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                stats['keep'] += 1
                continue
            if args.mode in ('gate_plus_window', 'window_only') and task_of(rel) in QC_TASKS and rel in window:
                row['final_response'] = window[rel]
                stats['window'] += 1
            elif args.mode in ('gate', 'gate_plus_window'):
                row['final_response'] = convert_native(nat_stage.get(rel))
                stats['gate'] += 1
            else:
                stats['keep'] += 1
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'mode={args.mode}: {stats} -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
