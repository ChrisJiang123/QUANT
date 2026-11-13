"""Reuse the frozen native offline guard/metadata adapter; write a new audit log."""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from quant_platform.research.governance import ensure_research, now

args = sys.argv[1:]
if '--config' not in args: raise PermissionError('explicit research config required')
config = json.loads(Path(args[args.index('--config')+1]).read_text())
ensure_research(config)
if config.get('trading_mode') != 'spot' or config.get('margin_mode'):
    raise PermissionError('spot research only')
audit = Path(os.environ['QUANT_ENGINE_AUDIT']).resolve()
if not any(audit.is_relative_to(ROOT/p) for p in ['experiments/next_step_20261113', 'experiments/rule_forward_v1']):
    raise PermissionError('separate audit path required')
namespace = runpy.run_path(str(ROOT/'ops/v4_freqtrade_entry.py'), run_name='receipt_offline_adapter')
code = 1
try:
    code = namespace['main'](args) or 0
except SystemExit as exc:
    code = exc.code or 0
finally:
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(timestamp=now(), argv=args, exit_code=code,
                                network_guard_active=True, container_network_none=False,
                                blocked_network_attempts=namespace['blocked']))+'\n')
sys.exit(code)
