import hashlib,io,json,zipfile,sys
from pathlib import Path
from v4_bootstrap import ROOT
sys.path.insert(0,str(ROOT/'src'))
from quant_platform.research.governance import digest_file,write_json

class Parts(io.RawIOBase):
    def __init__(self,paths):
        self.paths=paths;self.sizes=[p.stat().st_size for p in paths];self.size=sum(self.sizes);self.pos=0
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos if whence==1 else self.size)+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,size=-1):
        size=min(size if size>=0 else self.size-self.pos,self.size-self.pos);result=[];base=0
        for path,length in zip(self.paths,self.sizes):
            if base<=self.pos<base+length and size:
                with path.open('rb') as f:
                    f.seek(self.pos-base);data=f.read(min(size,base+length-self.pos))
                result.append(data);size-=len(data);self.pos+=len(data)
            base+=length
        return b''.join(result)

def main():
    verification=json.loads((ROOT/'experiments/v4/archive_verification/metrics.json').read_text())
    if not verification['publisher_verified']:raise ValueError('publisher parts not verified')
    paths=[ROOT/r['file'] for r in verification['parts']]
    download=json.loads((ROOT/'data/manifests/download-86ac87cbc20bf78b429b08e637460fbf03061947748205ce4cfca23333aed1f8.json').read_text())
    results=[]
    with zipfile.ZipFile(Parts(paths)) as z:
        for member in download['members']:
            if member['member'].endswith('.csv'):
                data=z.read(member['member']);actual=hashlib.sha256(data).hexdigest()
                if actual!=member['sha256']:raise ValueError('official member differs from TASK001')
                results.append(dict(member=member['member'],sha256=actual,original_unchanged=True,publisher_parts_verified=True))
        manifest=json.loads(z.read(next(n for n in z.namelist() if n.endswith('MANIFEST.json'))))
    write_json(ROOT/'experiments/v4/archive_verification/members.json',dict(schema_version=1,members=results,archive_manifest=manifest),immutable=True)
    print('PUBLISHER_VERIFIED_MEMBERS_MATCH_TASK001')
if __name__=='__main__':main()
