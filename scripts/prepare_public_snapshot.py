"""Prepare a reviewable source-only package, without rewriting or pushing history."""
import json
import re
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import digest_file, git_sha, now, write_json
OUT=ROOT/'experiments/next_step_20261113/public_candidate'

def main():
    files={}
    for prefix in ['src','tests','scripts','ops','freqtrade_userdir/strategies','freqtrade_userdir/receipt_strategies']:
        for p in (ROOT/prefix).rglob('*'):
            rel=p.relative_to(ROOT).as_posix()
            if p.is_file() and p.suffix in {'.py','.ps1'} and '/jobs/' not in rel and '__pycache__' not in rel:
                files[rel]=rel
    for p in ROOT.glob('requirements*.txt'): files[p.name]=p.name
    for prefix,ext in [('configs','.json'),('third_party/licenses','.txt')]:
        for p in (ROOT/prefix).rglob('*'+ext): files[p.relative_to(ROOT).as_posix()]=p.relative_to(ROOT).as_posix()
    files.update({'README.md':'README_PUBLIC.md','.gitignore':'ops/public_snapshot.gitignore',
                  'pyproject.toml':'pyproject.toml','.gitattributes':'.gitattributes',
                  'THIRD_PARTY_NOTICES.md':'THIRD_PARTY_NOTICES.md',
                  'docs/FORWARD_OPERATIONS.md':'docs/FORWARD_OPERATIONS.md',
                  'reports/V4_NEXT_STEP_REPORT.md':'reports/V4_NEXT_STEP_PUBLIC.md',
                  'reports/V4_NEXT_STEP_METRICS.json':'experiments/next_step_20261113/comparison.json'})
    checks=[]
    for target,source in sorted(files.items()):
        raw=(ROOT/source).read_bytes(); text=raw.decode('utf-8')
        if re.search(r'[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|<|\$|\{)[\w.-]+',text,re.I):
            raise ValueError('personal path in candidate: '+target)
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}',text):
            raise ValueError('possible secret in candidate: '+target)
        if re.search(r'\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b',text):
            raise ValueError('connection information in candidate: '+target)
        checks.append(dict(target=target,source=source,sha256=digest_file(ROOT/source),bytes=len(raw)))
    OUT.mkdir(parents=True,exist_ok=True)
    package=OUT/'QUANT-public-source-review.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for target,source in sorted(files.items()): z.write(ROOT/source,target)
    value=dict(generated_at=now(),local_source_head=git_sha(ROOT),file_count=len(checks),files=checks,
               package_sha256=digest_file(package),package=str(package.relative_to(ROOT)),
               public_history='NEW_CLEAN_ROOT_REQUIRES_EXPLICIT_AUTHORIZATION',
               old_history_rewritten=False,pushed=False,
               exclusions=['private Git ancestry','legacy documents','user planning documents','historical reports/logs',
                           'raw data','weights','environments','remote connection information','local manifests'])
    write_json(OUT/'manifest.json',value)
    print(json.dumps({k:value[k] for k in ['file_count','package_sha256','package','public_history']}))

if __name__=='__main__': main()
