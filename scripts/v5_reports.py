"""Generate the single completion report and supporting derived evidence."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quant_platform.v5.common import OUT,CFG,read,status,registered
from quant_platform.research.governance import write_json,digest_file,now,Ledger

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)

def main():
    registered();v=read(OUT/'validation.json');gpu=read(OUT/'gpu_verification.json');legacy=read(OUT/'legacy_diagnostic.json');data=read(OUT/'data/manifest.json');us=read(OUT/'us_data/coverage.json')
    report=ROOT/'reports/v5';report.mkdir(parents=True,exist_ok=True);metrics=v['all_metrics'];mapping=read(OUT/'mapping_diagnostics.json')
    xml=OUT/('tests-final-with-collection.xml' if (OUT/'tests-final-with-collection.xml').exists() else 'tests-final.xml');tree=ET.parse(xml);suites=tree.findall('.//testsuite');counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    failures=[dict(test=c.attrib.get('name'),message=f.attrib.get('message'),detail=f.text) for c in tree.findall('.//testcase') for f in list(c) if f.tag in ['failure','error']]
    write_json(OUT/'test_summary.json',dict(counts=counts,failures=failures,xml_sha256=digest_file(xml)))
    status('V5-TEST','IMPLEMENTED_AND_TESTED' if not counts['failures'] and not counts['errors'] else 'BLOCKED',[str(xml.relative_to(ROOT)),'experiments/v5/test_summary.json'],financial='SOFTWARE_TESTS_ONLY',run='COMPLETED',counts=counts)
    for rid in ['V5-CORE','V5-FACTOR','V5-GRAPH']:
        status(rid,'IMPLEMENTED_AND_TESTED',['experiments/v5/gpu_verification.json','experiments/v5/validation.json'],financial=v['families'][rid.split('-')[1].lower()]['verdict'])
    maxerr=max(x['max_cpu_gpu_absolute_error'] for x in gpu['checks'])
    coverage='# V5 数据覆盖\n\n'+table(['资产','真实US OHLC根数','OHLC起止','资金行数','等级'],[[r['pair'],r['prices'],r['price_start']+'—'+r['price_end'],r['funding'],r['evidence']] for r in us['coverage']])
    coverage+='\n真实US绝对历史保证金、日内标记价/强平规则、资金原始价格单位未完全核实。产品ID/乘数/tick/当前按张费已有官方来源原文与哈希。不能据这些短而稀疏的样本报告完整美国回测或生存能力。\n\n'
    coverage+='现货价格有长历史；BTC/ETH资金区间各自只覆盖2018-09-01至2026-01-31，共2704个完整日，不是十年真实美国永续。共同模型DEV为2022—2024，1096日；缺区间拒绝主回放，不补零。历史收盘+6h为保守推定，真实前向必须用实际接收时刻。\n\n'
    coverage+=f'模型样本{data["samples"]}，八节点mask按各节点真实首根/缺失。基础因子8、辅助8，其中基差/事件未伪造；新闻0真实有效样本。固定节点集合有幸存者选择局限，不能称历史全市场。GDELT三次429、日历vintage、同步基差的缺口见data/blocked_modalities.json。\n\n'
    coverage+='官方入口：[美国产品](https://support.kraken.com/articles/us-perpetual-futures)、[费用](https://support.kraken.com/articles/us-futures-fees)、[规格](https://support.kraken.com/articles/contract-specifications)。实际页面快照与访问时刻在experiments/v5/public_sources。\n'
    (report/'DATA_COVERAGE.md').write_text(coverage,encoding='utf-8')
    diag='# 核心诊断与接线\n\n'+f'旧权重预测复现最大误差{legacy["weight_reproduction_max_error"]:.8g}。旧loss实际含tail对收益的Huber，配置loss_weights未列出这个额外项；V5 q05归一化pinball明确注册。旧close标签不等于V5未来可执行open标签，旧结果保留探索标记。标准化只在当时训练内部拟合，未据DEV重标旧权重。\n\n'
    diag+=table(['V5变体/期限','MAE skill','Holm p','80%覆盖','q05穿越率'],[[k,f'{m["mae_skill"]:.5f}',f'{m["p_holm"]:.5f}',f'{m["coverage80"]:.4f}',f'{m["tail_exceedance05"]:.4f}'] for k,m in v['forecast'].items()])
    diag+='\n五种子及全部旧学习曲线、校准分箱/波动比例见legacy_diagnostic.json；原始方向头与校准头的Brier和幅度、15次原始头回放见mapping_diagnostics.json。中位数映射和校准方向的同账本四格结果全部保留，未选最优映射升级。预测误差诊断仅能说明关联，不能将缩放视为负结果的唯一原因。\n\n'
    cal=mapping['calibration']
    diag+=f'新V5方向原始头Brier组均值{np.mean([x["raw_brier"] for x in cal]):.6f}，内部校准后组均值{np.mean([x["calibrated_brier"] for x in cal]):.6f}。这是全部变体/种子/折/期限的描述性组均值，不能代替主期限检验。原始头回放的较好个例不能据此替换已注册主映射；未来研究需另注册。\n\n'
    diag+=f'4090真实完成60拟合：3变体×5种子×(3年度WF+1仅DEV最终拟合)。固定CPU/GPU最大误差{maxerr:.8g}，参数量/显存/实际有效样本/耗时/梯度/参数更新逐模型记录。全部sentiment真实样本为0。动态融合增量资格={v["dynamic_fusion_qualified"]}，本轮执行={v["dynamic_fusion_executed"]}。\n'
    (report/'MODEL_DIAGNOSTIC.md').write_text(diag,encoding='utf-8')
    reconciliation=read(OUT/'freqtrade_reconciliation/metrics.json')
    engine='# 引擎与对账\n\n'+f'真实Freqtrade2026.10拒绝Kraken isolated/cross futures，因此V5使用唯一事件主账本。真实零费率现货可比夹具{reconciliation["trades"]}笔交易、{reconciliation["fills"]}次成交，最终差额{reconciliation["final_error"]}，逐日最大差额{reconciliation["max_wallet_error"]}。这只是会计对账，不是Alpha或美国强平验证。\n\n'
    engine+='多空、反转/减仓、整数张、固定费/摩擦分解、持有区间资金、DST、钱包隔离、在途资金、保证金提高、维护、确实跨越强平边界、同根歧义、硬锁定、归因守恒、续接等见新增测试。日线没有真实事件顺序，保守路径/有利先行敏感性不替代日内标记价。\n\n'
    engine+='初轮调用离线上下文却未进入的问题已修正；socket注入确认网络拒绝后重跑24组并保留旧结果。正式backtests_v2不覆盖初轮backtests。所有命令失败保留，未删除或自动跳过。\n'
    (report/'ENGINE_RECONCILIATION.md').write_text(engine,encoding='utf-8')
    rows=[]
    for name,m in metrics.items():rows.append([name,f'{m["net_return"]:.4f}',f'{m["annual_return"]:.4f}',f'{m["sharpe"]:.4f}',f'{m["max_drawdown"]:.4f}',f'{m["realized_vol"]:.4f}',f'{m["average_gross"]:.4f}',f'{m["funding_cash"]:.2f}',m['liquidations'],m['hard_breakers']])
    ablation='# 全部DEV比较（SCENARIO_ONLY）\n\n'+table(['试验','净收益','年化','Sharpe','回撤','波动','平均gross','资金现金','强平','硬熔断'],rows)
    ablation+='\n每项逐日权益、成交、事件链、分仓/方向归因、换手、保证金、整数/风险拒绝、年份、UP/DOWN/SIDEWAYS和固定压力窗口位于experiments/v5/backtests_v2/<试验>/。五种子均值/中位数/最差、28日配对区间、DSR、HAC与Holm见validation.json；不是独立未来样本。\n\n'
    ablation+=table(['变体','G1','G2','G3','G4','种子Δ均值','种子Δ中位','最差Δ','结论'],[[k,m['G1'],m['G2'],m['G3'],m['G4'],f'{m["mean_seed_delta"]:.4f}',f'{m["median_seed_delta"]:.4f}',f'{m["worst_seed_delta"]:.4f}',m['verdict']] for k,m in v['families'].items()])
    ablation+='\nG2对比各成本下最强简单基线及重放的缩减风险诊断，不能只比弱规则。风险预算相同不代表实现波动相同；matched_diagnostics.json逐项列实际差距。缩减比例是DEV风险诊断，不是新调优模型或干净终测。所有主结果保留预注册成本/路径。\n\n'
    rates=[]
    for cost,c in CFG['costs'].items():
        for days in [28,30]:
            income=.08*days/365;expense=2*(c['spot_bps']+c['perp_bps'])/10000
            rates.append([cost,days,f'{income:.6f}',f'{expense:.6f}',f'{income-expense:.6f}'])
    ablation+='套利8%年化每美元名义的期望融资与综合费用敏感性（按张最低费若更高还需加大成本；30天只作敏感性，不延长主持仓）：\n\n'+table(['成本','天数','融资','双腿开平bps成本','净额上界'],rates)
    (report/'ABLATIONS.md').write_text(ablation,encoding='utf-8')
    write_json(report/'METRICS.json',dict(verdict=v['verdict'],families=v['families'],forecast=v['forecast'],gpu=dict(fits=gpu['fits'],max_cpu_gpu_error=maxerr,gpu=gpu['environment']['gpu']),test_counts=counts,scope='SCENARIO_ONLY; no clean final or U.S. survival proof'))
    state=read(OUT/'BUILD_STATE.json')
    statuses=table(['需求ID','软件状态','运行','金融证据','证据'],[[rid,x['status'],x['run'],x['financial_evidence'],'; '.join(x['evidence'])] for rid,x in state['requirements'].items()])
    commands=[]
    for p in sorted((OUT/'commands').glob('*/receipt.json')):
        r=read(p);commands.append([p.parent.name,r['exit_code'],' '.join(r['command'])])
    (report/'COMMANDS.md').write_text('# 实际命令与退出码\n\n'+table(['回执目录','退出码','命令'],commands)+'\n完整stdout和哈希保留在各commands目录；未运行项目不会列作成功。\n',encoding='utf-8')
    text=f'''# V5 统一完成报告

1. 软件：五期限SIDGTN、真实辅助因子/图分支、三仓策略和离线事件账本已实现并运行；真实美国完整账务与合格前向仍受数据限制。最终测试{counts['tests']}项，失败{counts['failures']}、错误{counts['errors']}、跳过{counts['skipped']}。
2. SIDGTN：core、factor、graph各5种子，3年度WF和1仅DEV最终拟合，共60次真实4090拟合。情绪真实历史不足，金融实验BLOCKED_DATA；动态融合未取得门槛资格时保持禁用。
3. 相对规则：{v['verdict']}。14/28日三个变体均未改善零预测MAE；没有据最佳种子、映射或次要期限改换主检验。金融比较、原始/校准/中位方向和四格替换全部保留。
4. 4090：权重、指标、日志、checkpoint和manifest已回传核验；固定CPU/GPU最大绝对差{maxerr:.8g}。训练源码为冻结作业SHA，不用后续报告提交冒充训练版本。
5. 最重要下一步：补齐带真实接收时间的美国合约报价/标记价与保证金资料，给已冻结SIDGTN建立新的可审计前向窗口；当前不升级、不进入实盘。

## 四个维度

| 维度 | 结论 | 最简单下一步 |
|---|---|---|
| 软件 | 已实现和验证；数据门保守阻断 | 按README_V5运行dry-check/验证/恢复 |
| Alpha | {v['verdict']}；独立终测INCONCLUSIVE_DATA | 保留负结果；不重置FINAL_CONSUMED |
| 集成 | 专用账本+Freqtrade可比子集对账；美国规格部分未核实 | 补齐实际美元保证金/日内标记价 |
| 远程GPU | VERIFIED，60拟合+完整回传与固定批次核验 | 复用冻结作业，无需重复训练 |

## 需求状态与证据

{statuses}
## 验收边界

共同DEV为2022—2024的1096日，只是现货价格/境外费率/假设美国规格情景。真实美国OHLC仅BTC3根、ETH6根，费率各543行；历史保证金和完整标记价未知。假设下无强平不能证明美国生存能力。所有年份、状态、压力窗口、成本分解、资金收支、止损/熔断/强平和五种子细节由metrics生成，见reports/v5/ABLATIONS.md和原始逐次产物。

G1/HAC/Holm、G2最强简单基线/28日配对90%区间/DSR、G3五种子和压力、G4独立未来窗口分别记录。DSR保守计入{v['trial_count_for_DSR']}次已记录研究比较/拟合及历史试验；没有将交叠的资产/日标签当独立样本。干净final拒绝执行，合格独立窗口0/12；三个月观察也尚未发生。

观察入口实际加载已核验最终DEV权重，输出30条真实推理；旧代理特征和稀疏US行情触发数据门，合格前向行0，BLOCKED_US_FORWARD，没有虚构交易或净值。趋势/套利/短周期归因独立，短周期仍默认不进主组合；套利净成本门不为凑交易放宽。

LocalCTO只读实查：控制器正常、既有4090节点在线，当前固定health/Ollama可用；所给文档与代码仍将QUANT研究/SSH/文件适配标未部署。没有把健康结果当训练结果，没有改权限/服务。此次QUANT真实训练经已授权RustDesk执行，续传跨两次10054中断后完成。具体通道检查在本地experiments/v5/localcto_channel_probe.json，不公开连接信息。

## 验证、保留和复现

实际命令/退出码见reports/v5/COMMANDS.md及experiments/v5/commands；最终完整回归XML为{str(xml.relative_to(ROOT))}，定向失败与修正均保留。README_V5给出本机验证过的dry-check、prepare、作业准备、回传核验、验证、观察、离线重放与恢复命令。新增公开US采集已实际执行，四时钟与闭合K线门经过测试；每次保存独立原文/哈希，不覆盖首轮快照。最新原文仍未提供足够新鲜OHLC/标记价/保证金，重复下载不能增加合格窗口。未进行真实资金操作、重启/72h远程可靠性测试、真实美国完整强平验证、情绪历史金融实验或新的独立终测，均不得称已通过。

V4冻结98文件和旧报告/消费标记逐哈希保留；旧规则轨11月15日按原代码实际运行，3个真实UTC日、0成熟独立窗口，仍有DRIFT，仅属旧轨运维证据。首次回放守卫上下文缺陷已修正并作socket注入拒绝测试；初轮结果保留，正式结果在backtests_v2。无自动跳过失败或调整收益口径。

代码小步提交保留本地私有历史。公开发布只在已核验干净main祖先上增加脱敏V5源代码/本报告，不带原始数据、权重、日志、用户Prompt、连接信息或私有祖先；具体推送/PR结果以本地experiments/v5/publication.json为准，不自动合并。
'''
    (ROOT/'reports/V5_COMPLETION_REPORT.md').write_text(text,encoding='utf-8')
    files=[ROOT/'reports/V5_COMPLETION_REPORT.md',*report.glob('*.md'),report/'METRICS.json',OUT/'validation.json',OUT/'gpu_verification.json',xml]
    write_json(OUT/'delivery_manifest.json',dict(generated_at=now(),files={str(p.relative_to(ROOT)):digest_file(p) for p in files},registered_hash=registered(),ledger_rows=Ledger(OUT/'ledger.jsonl').verify()))
    state=read(OUT/'BUILD_STATE.json');state.update(phase='IMPLEMENTED_VALIDATED_WITH_DATA_BLOCKS',verdict=v['verdict'],final_status='INCONCLUSIVE_DATA',remaining_blocks=['historical sentiment/calendar/basis','full real U.S. mark/margin/quotes','clean forward windows']);write_json(OUT/'BUILD_STATE.json',state)
    print(json.dumps(dict(verdict=v['verdict'],test_counts=counts,requirements=len(state['requirements']),report='reports/V5_COMPLETION_REPORT.md')))

if __name__=='__main__':main()
