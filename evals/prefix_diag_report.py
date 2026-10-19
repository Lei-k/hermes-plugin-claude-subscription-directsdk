"""Summarize #17 prefix diagnostics: where consecutive requests stop replaying identically.

Run: python3 evals/prefix_diag_report.py /path/to/directsdk-prefix-diag.jsonl [--json]

Records are grouped by Hermes session id (falling back to the client id) and compared with
the previous successful record of the same group. For each request the report gives the
first diverging position at each layer (host messages, replayed frames, native wire body)
and classifies a *rewrite*: cache_read at or below the system+tools prefix while the prior
request in the group had a larger prefix. Output contains positions, block types, sizes
and counts only; the input already holds no content.
"""
import argparse
import collections
import json
import sys


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) <= len(b) else len(b)


def flat_wire(wire):
    rows = [('system', j, None, h) for j, h in enumerate(wire.get('system') or [])]
    rows.append(('tools', 0, None, [wire.get('tools_all')]))
    for i, (role, blocks) in enumerate(wire.get('messages') or []):
        for j, block in enumerate(blocks):
            rows.append(('msg', i, j, block))
    return rows


def flat_frames(frames):
    rows = []
    for i, frame in enumerate(frames or []):
        if frame[0] == 'system':
            rows.append(('system', i, None, frame[1]))
        else:
            for j, block in enumerate(frame[1]):
                rows.append((frame[0], i, j, block))
    return rows


def locate(prev_rows, rows):
    k = first_diff([r[3] for r in prev_rows], [r[3] for r in rows])
    if k is None:
        return None
    here = rows[k] if k < len(rows) else None
    was = prev_rows[k] if k < len(prev_rows) else None
    def show(r):
        if r is None:
            return None
        kind = r[3][0] if isinstance(r[3], list) and r[3] and isinstance(r[3][0], str) and len(r[3]) == 4 else None
        size = r[3][-2] if isinstance(r[3], list) and len(r[3]) >= 3 else None
        return {'layer_kind': r[0], 'message': r[1], 'block': r[2], 'type': kind, 'size': size}
    return {'index': k, 'of_prev': len(prev_rows), 'now': show(here), 'was': show(was)}


def same_head(prev_block, block):
    """Same block type and same hash of the first 256 canonical characters: an edit in place."""
    return (isinstance(prev_block, list) and isinstance(block, list) and len(prev_block) == len(block) == 4
            and prev_block[0] == block[0] and prev_block[3] == block[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    for line in open(args.path, encoding='utf-8'):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        groups[record.get('sid') or record.get('client')].append(record)
    summary = collections.Counter()
    where = collections.Counter()
    events = []
    for key, records in groups.items():
        records.sort(key=lambda r: (r.get('ts') or 0, r.get('seq') or 0))
        prev = None
        for record in records:
            wire = (record.get('wire') or {}).get('forwarded') or (record.get('wire') or {}).get('native')
            usage = record.get('usage') or {}
            if record.get('error') or not wire:
                summary['skipped_failed'] += 1
                continue
            summary['requests'] += 1
            if prev is None:
                prev = record
                continue
            pwire = (prev['wire'].get('forwarded') or prev['wire'].get('native'))
            wl = locate(flat_wire(pwire), flat_wire(wire))
            fl = locate(flat_frames(prev['frames']), flat_frames(record['frames']))
            hl = first_diff([h[1] for h in prev['host']], [h[1] for h in record['host']])
            prefix = len(wire.get('system') or []) + 1
            read = usage.get('cache_read_input_tokens') or 0
            written = usage.get('cache_creation_input_tokens') or 0
            event = {'group': str(key)[-12:], 'seq': record.get('seq'), 'gap_s': round((record.get('ts') or 0) - (prev.get('done') or prev.get('ts') or 0), 1),
                     'cache_read': read, 'cache_write': written, 'wire': wl, 'frames': fl, 'host_first_diff': hl,
                     'host_len': [len(prev['host']), len(record['host'])]}
            # Only the previous request's cached span matters: through its last message breakpoint.
            prev_rows = flat_wire(pwire)
            breakpoints = [k for k, r in enumerate(prev_rows) if r[0] == 'msg'
                           and any(m[0] == 'm' and m[1] == r[1] and m[2] == r[2] for m in pwire.get('marks') or [])]
            cached_end = max(breakpoints) if breakpoints else len(prev_rows) - 1
            event['prev_breakpoint_index'] = cached_end
            if not breakpoints:
                summary['prev_without_message_breakpoint'] += 1
            if wl is None or wl['index'] > cached_end:
                summary['append_only'] += 1
            else:
                summary['prefix_changed'] += 1
                layer = 'host' if hl is not None and hl < len(prev['host']) else (
                    'frames' if fl is not None and fl['index'] < len(flat_frames(prev['frames'])) else 'native/relay')
                rows_prev, rows_now = flat_wire(pwire), flat_wire(wire)
                k = wl['index']
                edit = same_head(rows_prev[k][3], rows_now[k][3]) if k < len(rows_now) else False
                position = 'system/tools' if k < prefix else 'history'
                where[(layer, position, (wl['now'] or {}).get('type'), 'edit-in-place' if edit else 'replaced/inserted')] += 1
                event['layer'] = layer
                event['position'] = position
                events.append(event)
            prev = record
    report = {'summary': summary, 'first_divergence': [
        {'layer': k[0], 'position': k[1], 'block_type': k[2], 'shape': k[3], 'count': n}
        for k, n in where.most_common()], 'events': events[-50:]}
    if args.json:
        json.dump(report, sys.stdout, indent=1, default=str)
        print()
        return
    print('requests', summary['requests'], 'append_only', summary['append_only'],
          'prefix_changed', summary['prefix_changed'], 'skipped_failed', summary['skipped_failed'])
    for row in report['first_divergence']:
        print(f"{row['count']:5d}  first change at {row['layer']:12s} {row['position']:12s} {row['block_type']!s:12s} {row['shape']}")
    for e in events[-15:]:
        print(json.dumps(e, default=str))


if __name__ == '__main__':
    main()
