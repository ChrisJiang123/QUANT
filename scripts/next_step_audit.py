"""Local-only preservation and public-history audit; never prints matched values."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from quant_platform.research.governance import digest_file, now, write_json, verify_freeze

OUT = ROOT / 'experiments/next_step_20261113'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / 'v4_start_manifest.json').read_text())
    verify_freeze(json.loads((ROOT / 'experiments/v4/frozen.json').read_text()))
    for path, sha in manifest['protected'].items():
        assert digest_file(ROOT / path) == sha, path
    protected = {p.relative_to(ROOT).as_posix(): digest_file(p)
                 for p in (ROOT / 'reports').glob('*V4*') if p.is_file()}
    for p in (ROOT / 'experiments/v4').rglob('metrics.json'):
        protected[p.relative_to(ROOT).as_posix()] = digest_file(p)
    for name in ['ledger.jsonl', 'report_provenance.json', 'registration.json']:
        p = ROOT / 'experiments/v4' / name
        protected[p.relative_to(ROOT).as_posix()] = digest_file(p)
    write_json(OUT / 'v4_evidence_preservation.json', protected, immutable=True)
    findings = {}
    objects = git('rev-list', '--objects', '--all').decode('utf-8').splitlines()
    blobs = 0
    for line in objects:
        oid, _, path = line.partition(' ')
        if git('cat-file', '-t', oid).strip() != b'blob':
            continue
        blobs += 1
        kinds = []
        suffix = Path(path).suffix.lower()
        if suffix in {'.pdf', '.docx', '.zip', '.parquet', '.feather', '.pt', '.npy', '.npz'}:
            kinds.append('BINARY_OR_DATA_REQUIRING_PUBLICATION_LICENSE_REVIEW')
        if re.search(r'(^|/)(\.env(?:\..*)?|[^/]+\.(?:key|pem|pfx))$', path):
            kinds.append('POTENTIAL_CREDENTIAL_FILE_NOT_READ')
        else:
            raw = git('cat-file', 'blob', oid)
            if b'\x00' not in raw and len(raw) < 5_000_000:
                txt = raw.decode('utf-8', errors='replace')
                if re.search(r'[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|<|\$|\{)[\w.-]+', txt, re.I):
                    kinds.append('PERSONAL_ACCOUNT_PATH')
                if re.search(r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b', txt):
                    kinds.append('UNREDACTED_PRIVATE_CONNECTION_INFORMATION')
                if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}', txt):
                    kinds.append('POSSIBLE_SECRET_NO_VALUE_DISCLOSED')
        for kind in kinds:
            findings.setdefault(kind, []).append({'blob': oid, 'path': path})
    result = dict(generated_at=now(), source_head=git('rev-parse', 'HEAD').decode().strip(),
                  reachable_blobs=blobs, categories={k: len(v) for k, v in findings.items()},
                  findings=findings, matched_values_saved=False,
                  public_push='BLOCKED_UNSUITABLE_EXISTING_HISTORY' if findings else 'REVIEWED',
                  history_rewritten=False)
    write_json(OUT / 'public_history_audit.json', result)
    print(json.dumps({k: result[k] for k in ['reachable_blobs','categories','public_push']}))

if __name__ == '__main__':
    main()
