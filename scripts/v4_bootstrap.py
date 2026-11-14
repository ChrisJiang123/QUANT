"""Preserve the input tree and record commands without overwriting old evidence."""
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'experiments/v4'
E.mkdir(parents=True, exist_ok=True)

def now():
    return datetime.now(timezone.utc).isoformat()

def run(name, argv, timeout=1800):
    folder = E / 'commands'
    folder.mkdir(exist_ok=True)
    suffix = 1
    while (folder / f'{name}-{suffix}.json').exists():
        suffix += 1
    stem = folder / f'{name}-{suffix}'
    start = now()
    begin = time.monotonic()
    with stem.with_suffix('.log').open('wb') as log:
        try:
            p = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = p.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.write(str(exc).encode())
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
    result = dict(schema_version=1, command=argv, exit_code=code, started_at=start,
                  completed_at=now(), duration_seconds=time.monotonic()-begin,
                  log=stem.with_suffix('.log').relative_to(ROOT).as_posix())
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return code

def bootstrap():
    manifest = E / 'input_manifest.json'
    if not manifest.exists():
        skip = {'.git', '.venv', '.venv-training', 'build', '__pycache__', '.pytest_cache'}
        files = [p for p in ROOT.rglob('*') if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)
                 and not p.is_relative_to(E) and not p.is_relative_to(ROOT/'scripts')]
        rows = {}
        with zipfile.ZipFile(E/'input_backup.zip', 'x', zipfile.ZIP_DEFLATED) as z:
            for p in files:
                data = p.read_bytes()
                rel = p.relative_to(ROOT).as_posix()
                rows[rel] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                z.writestr(rel, data)
        manifest.write_text(json.dumps(dict(created_at=now(), files=rows), indent=2), encoding='utf-8')
    if not (ROOT/'.git').exists():
        run('git-init', ['git','init','-b','v4-build'])
        run('git-add-input', ['git','add','.'])
        run('git-start-commit', ['git','-c','user.name=Quant Research','-c','user.email=research@localhost',
                                'commit','-m','Preserve V4.4 starting source and retained audit reports'])
        run('git-start-tag', ['git','tag','v4.4-start-20261112'])
    run('regression-before', [str(ROOT/'.venv-training/Scripts/python.exe'),'-m','pytest','-q',
                             '--junitxml=experiments/v4/regression-before.xml'])

if __name__ == '__main__':
    if len(sys.argv)>1:
        sys.exit(run(sys.argv[1], sys.argv[2:]))
    bootstrap()
