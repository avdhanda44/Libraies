"""Download the notebooks' public datasets once, then reuse local copies."""

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

CACHE_DIR = Path(__file__).resolve().parent / 'data' / 'cache'
DATASETS = {
    'winequality-red.csv': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    'winequality-white.csv': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    'co2_mm_mlo.csv': 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv',
}


def get_dataset(name, *, offline=False):
    """Return a cached path; download only when absent and offline is False."""
    url = DATASETS[name]
    path = CACHE_DIR / name
    if path.is_file() and path.stat().st_size:
        return path
    instructions = (
        f'Download {url} and save it as {path}, or run '
        f'`python dataset_cache.py {name}` with internet access.'
    )
    if offline:
        raise FileNotFoundError(instructions)
    request = Request(url, headers={'User-Agent': 'Python-Libraries-Learning/1.0'})
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
            content_type = response.headers.get('Content-Type', '')
        if not content.strip() or 'text/html' in content_type or content.lstrip().lower().startswith((b'<!doctype html', b'<html')):
            raise ValueError('The server returned an empty file or HTML instead of data.')
    except (URLError, OSError, ValueError) as exc:
        raise RuntimeError(f'Could not fetch {name}. {instructions}') from exc
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*', help='Dataset filenames; omit to download all three.')
    parser.add_argument('--offline', action='store_true', help='Check the cache without downloading.')
    args = parser.parse_args()
    for name in args.names or DATASETS:
        if name not in DATASETS:
            parser.error(f'Unknown dataset: {name}. Choose from {", ".join(DATASETS)}')
        print(get_dataset(name, offline=args.offline))
