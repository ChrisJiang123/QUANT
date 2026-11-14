"""Stream publisher checksummed parts to append-only storage, with resumable partials."""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from v4_bootstrap import ROOT

OUT=ROOT/'experiments/v4/archive_verification'
OUT.mkdir(parents=True,exist_ok=True)
STORE=ROOT/'data/v4_archives';STORE.mkdir(exist_ok=True)
def part(line):
    expected,name=line.split()
    target=STORE/name; partial=target.with_suffix(target.suffix+'.partial')
    result={'file':str(target.relative_to(ROOT)),'expected_sha256':expected}
    if target.exists():
        with target.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        result.update(actual_sha256=actual,status='VERIFIED' if actual==expected else 'CORRUPT')
    else:
        begin=time.monotonic()
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            url='https://assets.kraken.com/marketing/institutions/'+name
            headers={'User-Agent':'QuantSIDGTN/4.4 archive verification'}
            if offset: headers['Range']=f'bytes={offset}-'
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(urllib.request.Request(url,headers=headers),timeout=45) as r:
                if offset and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise ValueError('resume range mismatch')
                with partial.open('ab' if offset else 'xb') as f:
                    while block:=r.read(4*1024*1024):
                        f.write(block)
                        if time.monotonic()-begin>600: raise TimeoutError('bounded transfer; partial retained for resume')
            with partial.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected: raise ValueError('publisher checksum mismatch')
            partial.rename(target);os.chmod(target,0o444)
            result.update(status='VERIFIED',actual_sha256=actual,bytes=target.stat().st_size)
        except Exception as exc: result.update(status='BLOCKED',error=str(exc),partial_bytes=partial.stat().st_size if partial.exists() else 0)
        result['seconds']=time.monotonic()-begin
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2))
    print(name,result['status'],flush=True)
    return result
if __name__=='__main__':
    lines=(ROOT/'experiments/v4/public/kraken_checksums.raw').read_text().splitlines()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(part,lines))
    (OUT/'metrics.json').write_text(json.dumps({'schema_version':1,'parts':results,'publisher_verified':all(r['status']=='VERIFIED' for r in results)},indent=2))
