import json
from pathlib import Path
import pytest
from quant_platform.workers.jobs import scan_payload,verify_return,validate_manifest

def test_worker_credentials_and_traversal_rejected():
    with pytest.raises(PermissionError):scan_payload('config.json',b'{"api_key":"not_allowed"}')
    with pytest.raises(PermissionError):scan_payload('.env',b'x')
    with pytest.raises(ValueError):validate_manifest({'schema_version':1,'job_id':'x','command':['python','worker_entry.py'],'files':{'../escape':'a'*64}})

def test_worker_return_hash_tampering(tmp_path):
    p=tmp_path/'metrics.json';p.write_text('{}')
    m={'schema_version':1,'files':{'metrics.json':'0'*64}}
    with pytest.raises(ValueError):verify_return(tmp_path,m)
