import json,math
from pathlib import Path
from v4_bootstrap import ROOT

def load(path,default=None):
    p=ROOT/path;return json.loads(p.read_text(encoding='utf-8')) if p.exists() else ({} if default is None else default)
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def number(x):return 'N/A' if x is None else f'{x:.4f}'

def supplement():
    ext=load('experiments/v4/extended/metrics.json');score=load('experiments/v4/scorecard/metrics.json');l2=load('experiments/v4/l2/metrics.json');cal=load('experiments/v4/calibration/metrics.json')
    nextsteps=load('experiments/v4/planning/metrics.json');proj=load('experiments/v4/open_source/metrics.json')
    hy=table(['假设','本轮结论','证据'],[[k,v.get('interpretation',v.get('result','N/A')),'experiments/v4/extended/metrics.json'] for k,v in ext.get('hypotheses',{}).items()])
    ml=table(['SIDGTN','最强简单 ML','BASE Sharpe 增量','区块区间下界','STRESS 增量','G2','G3 种子及成本部分'],
       [[k,v['best_simple_ml'],number(v['base_sharpe_delta']),number(v['bootstrap']['lower90']),number(v['stress_delta']),v['G2'],v['G3_seed_and_cost_component']] for k,v in ext.get('sidgtn_vs_simple_ml',{}).items()])
    implication=table(['成本情景','DVOL 同区间 Ridge Sharpe 增量'],[[k,number(v)] for k,v in l2.get('incremental_sharpe',{}).items()])
    ten=table(['证据区段','策略','对应等仓位基准','成本','策略 Sharpe','基准 Sharpe','差值','策略实际均仓','基准实际均仓'],
       [[v['split'],v['model'],v['benchmark'],v['cost'],number(v['model_sharpe']),number(v['benchmark_sharpe']),number(v['sharpe_delta']),number(v['strategy_exposure']),number(v['benchmark_realized_exposure'])] for v in ext.get('matched_exposure',[])])
    ten+='\n恒定目标权重是策略平均实际仓位；无交易带、费用和价格变化会使基准实际平均仓位偏离目标，表中同时披露，未声称完全精确匹配。该对照为事后风险暴露匹配，不能作为预测策略。\n\n'
    ten+=table(['区段','模型','成本','市场状态','天数','状态 Sharpe','状态复合收益'],
        [[v['split'],v['model'],v['cost'],v['regime'],v['days'],number(v['sharpe']),number(v['net_return'])] for v in ext.get('regimes',[])])
    ten+=table(['压力起点','压力谷值','模型','观察日数','窗口成本后收益'],
        [[v['start'],v['end'],v['model'],v['days'],number(v['net_return'])] for v in ext.get('pressure_windows',[])])
    oos=ml+'\n逐资产预测诊断见 extended/metrics.json 的 asset_forecasts。主门仍按日期汇总相关资产损失差，不把两资产横截面 IC 当显著性证据。\n\n'
    calibration=table(['模型','期','Brier','ECE','分位数覆盖','标称覆盖'],[[v['model'],v['horizon'],number(v['brier']),number(v['ece']),number(v['q10_q90_coverage']),number(v['nominal_coverage'])] for v in cal.get('results',[])])
    hyptext='# 假设与增量检验\n\n'+hy+implication+ml+calibration+'''\n正的局部增量不等同于通过预注册门槛。DVOL/资金过滤属于另行登记的 DEV 同区间检验，未复用终测区；历史发布时点仍有推断局限。H3 的区块置信区间及是否确认由 h3/metrics.json 给出。H4 因历史 announced_at 不足保持 mask；H5 为探索性关联。不得从未显著的结果推断已经证明不存在任何效应。\n'''
    (ROOT/'reports/HYPOTHESES_V4_REPORT.md').write_text(hyptext,encoding='utf-8')
    integr=table(['项目','实查 tag','解析 commit','许可证','定位','实际调用'],[[v['project'],v['tag'],v['commit'],v['license'],v['decision'],v['actual_usage']] for v in proj.get('projects',[])])
    (ROOT/'reports/INTEGRATION_V4_REPORT.md').write_text('# 实际集成与边界\n\n'+integr+'\nQlib 实装版本、表达式交叉校验及容差来自 qlib/metrics.json；FreqAI 仅为合成数据可用性 smoke。RD-Agent、TradingAgents 未安装运行，不称为已集成。\n',encoding='utf-8')
    summary='\n逐条假设与增量：\n\n'+hy+implication+ml
    if nextsteps:
        summary+='\n最低成本下一份证据的时间量纲（只表示采样量，不会修复 DEV 失败）：\n\n'+table(['指标','值','证据'],[
            ['既有干净区独立窗口',nextsteps['existing_fresh_windows'],'planning/metrics.json'],
            ['原切分距离门槛所缺日历天',nextsteps['additional_split_days'],'planning/metrics.json'],
            ['新前向轨单独所需预测日数',nextsteps['new_forward_anchor_days'],'planning/metrics.json'],
            ['最后标签额外成熟日数',nextsteps['maturation_days'],'planning/metrics.json'],
            ['新轨最低总日历天',nextsteps['new_forward_total_days'],'planning/metrics.json'],
            ['近似月数',number(nextsteps['approx_calendar_months']),'planning/metrics.json']])
    summary+='\n最简单下一步是保留规则基线，持续按文件记录免费的价格量、已取得的 DVOL/资金费率和诊断预测。若研究新的改进模型，应先另开预注册和前向轨；无需为了本机研究先打通远程 GPU，也不应调整旧终测口径。\n'
    return {'summary':summary,'tenyear':ten,'oos':oos,'integration':integr}
