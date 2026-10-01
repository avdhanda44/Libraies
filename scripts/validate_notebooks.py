"""Execute notebooks in disposable working folders and retain reviewable results."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', help='Notebook filenames; default: all.')
    parser.add_argument('--timeout', type=int, default=600, help='Seconds allowed per cell.')
    args = parser.parse_args()
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted(ROOT.glob('*.ipynb'))
    destination = ROOT / 'outputs' / 'validation'
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        result = {'notebook': path.name, 'status': 'passed'}
        with tempfile.TemporaryDirectory(prefix='library-notebook-') as work:
            work = Path(work)
            for source in ROOT.glob('*.py'):
                shutil.copy2(source, work / source.name)
            if (ROOT / 'data').exists():
                shutil.copytree(ROOT / 'data', work / 'data')
            environment = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(work / '.matplotlib'),
                               IPYTHONDIR=str(work / '.ipython'), PYTHONHASHSEED='0')
            manager = KernelManager(kernel_name='python3')
            manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                    resources={'metadata': {'path': str(work)}},
                                    record_timing=True)
            try:
                client.execute(cwd=str(work), env=environment)
            except Exception as exc:
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                # An externally supplied manager is not owned by NotebookClient.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
        result['seconds'] = round(time.monotonic() - started, 2)
        nbformat.write(notebook, destination / path.name)
        results.append(result)
        (destination / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f"{path.name}: {result['status']} ({result['seconds']}s)", flush=True)
        if result['status'] == 'failed':
            print(result['error'][-1500:], flush=True)
    return int(any(result['status'] != 'passed' for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
