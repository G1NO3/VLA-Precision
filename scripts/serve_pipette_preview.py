#!/usr/bin/env python3
"""GUI-owned, preview-only JSON-lines worker. Writes local logs, never actions."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from vla_precision.integrations.openpi.pipette_preview import PreviewModel, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    def status(state, **extra):
        write_json(run / 'status.json', {'state': state, 'timestamp_ms': int(time.time()*1000), **extra})
    def event(value):
        with (run / 'events.jsonl').open('a') as f:
            f.write(json.dumps(value, allow_nan=False) + '\n')
    status('loading')
    try:
        model = PreviewModel(args.config, args.checkpoint)
        write_json(run / 'model.json', model.metadata)
        status('ready')
        print('Pipette pi05 preview ready; robot output disabled', flush=True)
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError('Expected a request object')
                if request.get('command') == 'stop':
                    break
                request_id = request.get('id', '')
                if len(request_id) != 16 or any(c not in '0123456789abcdef' for c in request_id):
                    raise ValueError('Invalid request ID')
                status('running', request_id=request_id)
                result = model.infer(request, run)
                write_json(run / 'result.json', result)
                event({'event': 'inference', **result})
                status('ready', completed_id=request_id)
            except Exception as error:
                traceback.print_exc()
                event({'event': 'error', 'request': request, 'error': str(error), 'timestamp_ms': int(time.time()*1000)})
                status('ready', completed_id=request.get('id') if isinstance(request, dict) else None, error=str(error))
    except Exception as error:
        status('error', error=str(error))
        raise
    finally:
        print('Pipette preview worker exited', flush=True)

if __name__ == '__main__':
    main()
