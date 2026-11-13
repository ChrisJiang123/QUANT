"""File-only candidates and non-LLM risk-review fallback; approval remains manual."""
import json
from pathlib import Path
from quant_platform.features.mining import FactorDSL

def import_candidate_file(path,frame,ledger):
    path=Path(path)
    if path.stat().st_size>1024*1024:raise ValueError('candidate file budget')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or not isinstance(data.get('expressions'),list) or len(data['expressions'])>30:raise ValueError('candidate schema/budget')
    return [FactorDSL().record_candidate(e,frame,ledger) for e in data['expressions']]

def risk_review_template(data_status,model_status):
    return dict(schema_version=1,source='STATIC_TEMPLATE_NOT_LLM',data_status=data_status,model_status=model_status,
                review=['Check PIT coverage','Check costs and turnover','Check independent OOS windows','Retain failures'],
                order_enabled=False,live_enabled=False,approval='RESEARCH_MATERIAL_ONLY')
