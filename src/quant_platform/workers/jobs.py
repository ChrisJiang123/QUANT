import json,re
from pathlib import Path,PurePosixPath
from quant_platform.research.governance import digest_file

def scan_payload(name,data):
    if any(x in name.lower() for x in ['.env','.pem','.key','credential','id_rsa','id_ed25519']):raise PermissionError('credential filename')
    patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
              rb'(?i)["\']?(?:api[_-]?key|secret|password|access_token)["\']?\s*[:=]\s*["\'][^"\']+["\']',
              rb'gh[pousr]_[A-Za-z0-9]{20,}',rb'sk-[A-Za-z0-9]{20,}']
    if any(re.search(p,data) for p in patterns):raise PermissionError('credential-like payload')

def validate_manifest(m):
    if m.get('schema_version')!=1 or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',m.get('job_id','')):raise ValueError('job schema')
    if m.get('command')!=['python','worker_entry.py']:raise ValueError('worker command whitelist')
    for name,h in m['files'].items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or not re.fullmatch('[a-f0-9]{64}',h):raise ValueError('unsafe manifest path/hash')

def verify_return(root,manifest):
    if manifest.get('schema_version')!=1:raise ValueError('return schema')
    root=Path(root).resolve()
    for name,h in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest_file(path)!=h:raise ValueError('return checksum/path mismatch')
    return True
