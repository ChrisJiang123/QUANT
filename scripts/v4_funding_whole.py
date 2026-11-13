"""Bounded whole-file fallback when the official server ignores Range."""
import hashlib,json,os,urllib.request,zipfile
from pathlib import Path
from v4_bootstrap import ROOT
from v4_funding_export import URL,OUT

def main():
    result={'schema_version':1,'url':URL,'anonymous':True,'proxy_used':False,'method':'bounded whole HTTPS GET because Range ignored'}
    path=OUT/'official_funding.zip.partial'
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));sha=hashlib.sha256();size=0
        with opener.open(urllib.request.Request(URL,headers={'User-Agent':'QuantSIDGTN public research/4.4'}),timeout=45) as r:
            result['http_status']=r.status
            with path.open('xb') as f:
                while block:=r.read(1024*1024):
                    size+=len(block)
                    if size>256*1024*1024:raise ValueError('bounded funding archive limit exceeded; partial retained')
                    sha.update(block);f.write(block)
        raw=OUT/(sha.hexdigest()+'.zip');path.rename(raw);os.chmod(raw,0o444);members=[]
        with zipfile.ZipFile(raw) as z:
            result['member_names']=z.namelist()
            for name in z.namelist():
                if not name.lower().endswith('.csv') or not any(s in name.upper() for s in ['PF_XBTUSD','PI_XBTUSD','PF_ETHUSD','PI_ETHUSD']):continue
                info=z.getinfo(name)
                if info.file_size>32*1024*1024:raise ValueError('member size bound exceeded')
                data=z.read(name);h=hashlib.sha256(data).hexdigest();target=OUT/(h+'.csv')
                if not target.exists():target.write_bytes(data);os.chmod(target,0o444)
                members.append(dict(name=name,path=str(target.relative_to(ROOT)),sha256=h,bytes=len(data)))
        result.update(status='ACQUIRED' if members else 'BLOCKED_NO_MATCHING_MEMBERS',archive_sha256=sha.hexdigest(),archive_path=str(raw.relative_to(ROOT)),bytes=size,members=members)
    except Exception as exc:result.update(status='BLOCKED',error=str(exc),partial_bytes=path.stat().st_size if path.exists() else 0)
    (OUT/'whole_metrics.json').write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
