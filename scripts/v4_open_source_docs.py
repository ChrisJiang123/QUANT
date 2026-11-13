import base64,json,re,tomllib
from pathlib import Path
from v4_bootstrap import ROOT

def main():
    public=ROOT/'experiments/v4/public';out=ROOT/'experiments/v4/open_source';out.mkdir(exist_ok=True)
    specs={'freqtrade':('freqtrade/freqtrade','2026.10','REUSE_DIRECT','Native isolated CLI backtesting; actual Freqtrade optimize.Backtesting and external strategy callbacks.'),
           'qlib':('microsoft/qlib','v0.9.8','ADAPTER','Actual installed distribution 0.9.7, because 0.9.8 is not published on this index; qlib.init, D.features and local binary provider.'),
           'rdagent':('microsoft/RD-Agent','v1.0.0','DEFER','No Linux/WSL runtime or local LLM; no third-party code executed. File-only DSL candidate import is our compatibility interface.'),
           'finrl':('AI4Finance-Foundation/FinRL-Trading','v1.0.0','DESIGN_REFERENCE','Read source documentation for weight-centric contracts; not installed, no broker SDK or execution code reused.'),
           'tradingagents':('TauricResearch/TradingAgents','v0.7.1','DEFER','Pinned README/metadata inspected; local Ollama endpoint unavailable. No framework or LLM call claimed; static template fallback is explicitly not LLM output.')}
    rows=[];licenses=ROOT/'third_party/licenses';licenses.mkdir(parents=True,exist_ok=True)
    for short,(repo,tag,decision,usage) in specs.items():
        info=json.loads((public/(short+'_repo.raw')).read_text(encoding='utf-8'));lic=json.loads((public/(short+'_license.raw')).read_text(encoding='utf-8'))
        command_receipts=sorted((ROOT/'experiments/v4/commands').glob(short+'-tag-resolve-*.json'))
        receipt=json.loads(command_receipts[-1].read_text());log=(ROOT/receipt['log']).read_text(encoding='utf-8')
        refs=[line.split() for line in log.splitlines() if re.match(r'^[0-9a-f]{40}\s',line)]
        peeled=[x[0] for x in refs if x[1].endswith('^{}')];sha=peeled[-1] if peeled else refs[0][0] if refs else None
        rawlicense=public/(short+'_LICENSE.raw')
        data=rawlicense.read_bytes() if rawlicense.exists() else base64.b64decode(lic['content'])
        (licenses/(short+'-LICENSE.txt')).write_bytes(data)
        notice=public/(short+'_NOTICE.raw');notice_status='PRESENT' if notice.exists() else 'No top-level NOTICE at requested tag (HTTP evidence retained); not proof repository contains no nested notices'
        if notice.exists():(licenses/(short+'-NOTICE.txt')).write_bytes(notice.read_bytes())
        metadata=public/(short+'_pyproject_toml.raw');project={};metadata_status='MISSING'
        if metadata.exists():
            parsed=tomllib.loads(metadata.read_text(encoding='utf-8'));project=parsed.get('project',{});metadata_status='ACQUIRED_PINNED_TAG'
        rows.append(dict(short=short,project=repo,url='https://github.com/'+repo,tag=tag,commit=sha,license=lic.get('license',{}).get('spdx_id'),
                         archived=info.get('archived'),last_push=info.get('pushed_at'),decision=decision,actual_usage=usage,notice=notice_status,
                         requires_python=project.get('requires-python','NOT_DECLARED_IN_PROJECT_METADATA'),dependencies=project.get('dependencies',[]),metadata_status=metadata_status,
                         license_path=f'third_party/licenses/{short}-LICENSE.txt',
                         tag_command=receipt['command'],tag_exit_code=receipt['exit_code'],source_receipts=str(public.relative_to(ROOT))))
    (out/'metrics.json').write_text(json.dumps(dict(schema_version=1,projects=rows),indent=2))
    lines=['# Official source reuse audit\n','All tags were resolved against official Git repositories during this run. Licenses/notices and raw pinned README/pyproject sources are retained; current PyPI runtime and Git tag are distinguished.\n',
      '| Project | Resolved tag / commit | License | Classification | Actual module/command |\n| --- | --- | --- | --- | --- |\n']
    for r in rows:lines.append(f"| [{r['project']}]({r['url']}) | {r['tag']} / {r['commit']} | {r['license']} | {r['decision']} | {r['actual_usage']} |\n")
    lines.append('\n| Project | Declared Python | Metadata | Archived | Last push | Runtime/data/API boundary |\n| --- | --- | --- | --- | --- | --- |\n')
    boundaries={'freqtrade':'Windows native smoke passed; feather OHLCV + PIT signal parquet; IStrategy/CLI; GPL external boundary; current public market precision applied historically',
      'qlib':'Windows isolated 0.9.7 smoke passed; custom 24/7 calendar and binary provider; D.features Ref/Mean/Std crosscheck; no stock data substituted',
      'rdagent':'Linux/Docker and chat+embedding prerequisites absent; no upstream API execution verified; only our candidate-file DSL adapter is callable',
      'finrl':'Equity/Alpaca-oriented broker dependencies and separate bt engine conflict with core scope; no runtime API was called',
      'tradingagents':'Local Ollama research route described upstream; unavailable local model; news memo remains L3 research material, no historical backtest input'}
    for r in rows:lines.append(f"| {r['project']} | {r['requires_python']} | {r['metadata_status']} | {r['archived']} | {r['last_push']} | {boundaries[r['short']]} |\n")
    lines.append('\nExact pinned declared dependency lists (including unavailable metadata, rather than guessed compatibility):\n\n```json\n'+json.dumps({r['project']:r['dependencies'] for r in rows},indent=2)+'\n```\n')
    lines.append('''\nFreqtrade requires a separate CLI/file boundary because of GPL and trading-SDK dependency isolation. Its Windows native installation worked; Docker/WSL route is unavailable without virtualization/system changes. Main backtests, lookahead and recursive analyses use actual pinned package code. The offline metadata replacement affects only market discovery, not engine execution. Smoke: ft-v2-smoke, bt-v2, final-backtests, ft-analyses-v3; exact argv/exit codes are in command receipts.

Qlib has no supplied crypto calendar; our adapter constructs the calendar/provider. Actual smoke independently evaluates registered momentum, trend and volatility factors and compares tolerance on our data. qlib-offline-smoke is network guarded and passes. Pinning an older available distribution is an explicit compatibility decision, not a claim that the inspected newer tag ran.

RD-Agent is Linux-oriented, Docker-dependent in the relevant workflows, and requires chat/embedding models. No such working runtime/model is present; do not install Windows services or guess remote endpoints. TradingAgents supports a local-model research route per the inspected documentation, but no local model was available. Their safe standalone candidate/template interfaces are tested, without representing either upstream framework as integrated.

FinRL-X targets a different market/broker and would import brokerage dependencies. Only public design documentation was read; nothing is installed or copied into core. Future asset-specific data, calendars, contracts and risk models remain separate.

Maintenance and exact notices: experiments/v4/open_source/metrics.json. Python support/dependencies are available in retained pinned pyproject/README snapshots. Compatibility is established only where an actual isolated smoke passed; documentation claims are not runtime guarantees.
''')
    (ROOT/'docs/OPEN_SOURCE_REUSE_AUDIT.md').write_text(''.join(lines),encoding='utf-8')
    notices=['# Third-party notices\n\nNo vendor trading or broker source was copied into the core package. Freqtrade strategy plugins are separate GPL-labelled files; Freqtrade itself is an external installed dependency.\n\n']
    for r in rows:notices.append(f"- [{r['project']}]({r['url']}), tag {r['tag']}, commit {r['commit']}, {r['license']}; {r['decision']}. Retained license: [{r['short']}]({r['license_path']}). NOTICE audit: {r['notice']}.\n")
    notices.append('\nQlib installed version is separately recorded in its smoke metrics. Complete installed dependency inventories are in requirements-v4-*-lock.txt; dependencies retain their own licenses. Public Kraken, Deribit, Alternative.me and Wikimedia datasets retain source URLs, collection timestamps, hashes and availability/attribution notes; redistribution rights are not assumed. GDELT/BitMEX failures are retained; no mirror, proxy or paid route was used.\n')
    (ROOT/'THIRD_PARTY_NOTICES.md').write_text(''.join(notices),encoding='utf-8')
    print('FIVE_OFFICIAL_PROJECT_AUDITS_WRITTEN')
if __name__=='__main__':main()
