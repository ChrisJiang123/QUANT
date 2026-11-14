"""Anonymous range extraction from the ZIP linked by Kraken's official support page."""
import hashlib,io,json,os,urllib.request,zipfile
from v4_bootstrap import ROOT
URL='https://assets-cms.kraken.com/files/51n36hrp/facade/4b70936c1227e4ae5514cba8bf41a5561cf13bd7.zip?dl='
OUT=ROOT/'experiments/v4/funding_export';OUT.mkdir(exist_ok=True)
class Remote(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.receipts=[];data,headers=self.get(0,0)
        self.size=int(headers['Content-Range'].split('/')[-1]);self.etag=headers.get('ETag')
    def get(self,start,end):
        headers={'User-Agent':'QuantSIDGTN public research/4.4','Range':f'bytes={start}-{end}'}
        if getattr(self,'etag',None):headers['If-Match']=self.etag
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(URL,headers=headers),timeout=30) as response:
            data=response.read(16*1024*1024+1);h=dict(response.headers)
            if response.status!=206 or len(data)!=end-start+1 or not h.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise ValueError('exact range unavailable')
            if getattr(self,'etag',None) and h.get('ETag')!=self.etag:raise ValueError('ETag changed')
        sha=hashlib.sha256(data).hexdigest();path=OUT/(sha+'.raw')
        if not path.exists():path.write_bytes(data);os.chmod(path,0o444)
        self.receipts.append(dict(range=h['Content-Range'],sha256=sha,path=str(path.relative_to(ROOT))))
        return data,h
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos if whence==1 else self.size)+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,size=-1):
        size=min(size if size>=0 else self.size-self.pos,self.size-self.pos)
        if size<=0:return b''
        if size>16*1024*1024:raise ValueError('member/range budget exceeded')
        data,_=self.get(self.pos,self.pos+size-1);self.pos+=len(data);return data

def main():
    metric={'schema_version':1,'url':URL,'source_page':'https://support.kraken.com/articles/export-historical-funding-rates','anonymous':True,'proxy_used':False}
    try:
        remote=Remote();members=[]
        with zipfile.ZipFile(remote) as z:
            names=z.namelist();print('FUNDING_MEMBERS',names[:10],flush=True)
            for name in names:
                if not name.lower().endswith('.csv') or not any(s in name.upper() for s in ['PF_XBTUSD','PI_XBTUSD','PF_ETHUSD','PI_ETHUSD']):continue
                info=z.getinfo(name)
                if info.file_size>16*1024*1024:raise ValueError('funding member bound exceeded')
                data=z.read(name);sha=hashlib.sha256(data).hexdigest();path=OUT/(sha+'.csv')
                if not path.exists():path.write_bytes(data);os.chmod(path,0o444)
                members.append(dict(name=name,path=str(path.relative_to(ROOT)),sha256=sha,crc32=f'{info.CRC:08x}',bytes=len(data)))
        metric.update(status='ACQUIRED' if members else 'BLOCKED_NO_MATCHING_MEMBERS',members=members,archive_bytes=remote.size,requests=remote.receipts,
                      integrity='official link + exact HTTPS ranges + stable ETag + member CRC + local SHA; no publisher SHA supplied for this ZIP')
    except Exception as exc:metric.update(status='BLOCKED',error=str(exc))
    (OUT/'metrics.json').write_text(json.dumps(metric,indent=2));print(metric['status'])
if __name__=='__main__':main()
