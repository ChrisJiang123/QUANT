# V4 续做统一报告

软件：**PARTIAL**（前向与规则因果性缺口已修复；历史协议偏差保留）。Alpha：**NO_GO_ALPHA**；新终测：**INCONCLUSIVE_DATA**。集成：原生 Freqtrade/Qlib 已实跑，仍为 PARTIAL_INTEGRATION。远程：RustDesk 当前会话画面已读取，未探测 GPU/Python/CUDA，未训练、未验收新产物。

最少下一步：普通账户每天运行 `powershell.exe -NoProfile -File .\ops\run_forward_daily.ps1`；调度配置已导出但未启用。公开原历史被安全门阻断，见下文。

最近真实运行完成：**2026-11-13T09:26:49.209745+00:00**；首条信号实际生成：2026-11-13 09:26:35.360378+00:00；数据截至收盘：2026-11-13 00:00:00+00:00。首次采集追加1条，第二次同键采集 appended=False，当前共1条。

成熟标签 **0**，待结算 **4**；最早14天市场到期 2026-11-28 00:00:00+00:00，最早完整日线结算 2026-11-29 00:00:00+00:00（还需实际收到）。有效独立28天窗口 **0**，距离12个尚差12个；两币同期不双计。实际连续运维 1/7 天，还需 6 天，约336天只是完整窗口的最低覆盖，不能预报通过。

监控：DRIFT，PSI={"ETHUSD": 0.5476898867958556, "XBTUSD": 0.3948183256754261}。这是分布变化报警，不是正收益证明。金融比较状态 **PENDING_EXECUTABLE_CLOSED_BARS**：尚无可成交未来完整日线，规则/BH/等平均仓位净值指标缺失，不写零收益。初始研究现金、零成交和零已发生成本仅表示尚未开始。

本轮修复并验证：真实响应接收时间；严格未来开盘；原子幂等/修订追加；部分成熟逐币逐期限结算；坏链/重复/缺口/未来值拒绝；独立组合窗口计数；每日普通用户入口。新轨首条前已有精确源码提交、代码归档、数据/配置/环境哈希。原 V4 数据、权重、报告、预注册和 final 消费标记逐项哈希核对保持不变。

外部信号实测：旧导出器截断后 gap_mask 和 target_weight 不一致，原因是下一日缺口依赖及数组共享回写。新规则导出器在真实 DEV 截断点重算一致。三档 DEV 纠正结果与原规则收益/Sharpe/成交相同；每档都对账，且成交日期有真实 K 线。其他旧模型执行指标仍带审计限制，只列历史探索值，不以修复规则替它们验收。未解锁 final，未新增训练。

修正运行 ID：CAUSAL_DEV_CORRECTION_20261113，仅 DEV。

| 成本 | 规则净收益 | Sharpe | 最大回撤 | 交易数 | 换手 | 实付模型成本 |
| --- | --- | --- | --- | --- | --- | --- |
| OPTIMISTIC | 280.94% | 0.9307 | -36.69% | 115 | 125.2589 | 4384.0598 |
| BASE | 198.61% | 0.7810 | -38.63% | 115 | 109.0540 | 9269.5870 |
| STRESS | 151.65% | 0.6753 | -40.39% | 115 | 99.0053 | 11880.6306 |

**本轮无新增训练必要。** 现有 Ridge/ElasticNet/LightGBM 与五个 SIDGTN 种子的两期限预测门没有通过；SIDGTN 成本后门也失败。规则 BASE/STRESS 相比低成本退化，曝光差异不能当预测增量；仅延长前向时间不能解除 NO_GO。没有足以预注册的新机制假设，因此未新增实验、未为证明连通而重复训练。

以下数字从原 metrics 自动提取，均为 DEV 历史探索结果，独立窗口列是历史回测覆盖，不是新轨干净前向样本。收益为引擎期末账户统计；Sharpe/回撤为引擎 MTM。NOT_EVALUATED 表示没有该模型的完整门证据。

| 模型 | 成本 | 净收益 | Sharpe | 最大回撤 | 换手 | 交易数 | 历史28天窗口 | G1 | G2 | G3 | G4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_hold | BASE | 8373.77% | 1.1211 | -77.35% | 145.3487 | 14 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| buy_hold | STRESS | 7985.22% | 1.1140 | -77.34% | 140.1742 | 14 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| sma200 | BASE | 1181.33% | 0.9281 | -54.06% | 710.3684 | 83 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| sma200 | STRESS | 859.46% | 0.8485 | -57.79% | 601.4580 | 83 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| rule | BASE | 198.61% | 0.7810 | -38.63% | 109.0540 | 115 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| rule | STRESS | 151.65% | 0.6753 | -40.39% | 99.0053 | 115 | 106 | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| ridge | BASE | 41.77% | 0.2996 | -56.88% | 90.2307 | 130 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| ridge | STRESS | 12.55% | 0.1949 | -59.75% | 80.2924 | 130 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| elasticnet | BASE | 39.56% | 0.2928 | -56.82% | 86.2265 | 127 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| elasticnet | STRESS | 11.07% | 0.1869 | -59.68% | 76.6963 | 127 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| lightgbm | BASE | 216.18% | 0.7689 | -40.12% | 118.8153 | 128 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| lightgbm | STRESS | 156.00% | 0.6485 | -42.76% | 105.2101 | 128 | 106 | False | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |
| sidgtn_seed11 | BASE | 6.90% | 0.1542 | -58.60% | 70.3669 | 127 | 106 | False | False | False | False |
| sidgtn_seed11 | STRESS | -13.53% | 0.0414 | -64.18% | 63.7630 | 127 | 106 | False | False | False | False |
| sidgtn_seed22 | BASE | 133.90% | 0.5576 | -58.22% | 94.0700 | 130 | 106 | False | False | False | False |
| sidgtn_seed22 | STRESS | 88.75% | 0.4497 | -59.11% | 83.6948 | 130 | 106 | False | False | False | False |
| sidgtn_seed33 | BASE | 167.46% | 0.6277 | -55.77% | 104.5806 | 130 | 106 | False | False | False | False |
| sidgtn_seed33 | STRESS | 116.04% | 0.5188 | -56.59% | 92.8336 | 130 | 106 | False | False | False | False |
| sidgtn_seed44 | BASE | 173.90% | 0.6756 | -37.76% | 108.0314 | 128 | 106 | False | False | False | False |
| sidgtn_seed44 | STRESS | 120.53% | 0.5555 | -38.99% | 96.2074 | 128 | 106 | False | False | False | False |
| sidgtn_seed55 | BASE | 138.07% | 0.6134 | -44.49% | 83.5060 | 128 | 106 | False | False | False | False |
| sidgtn_seed55 | STRESS | 94.57% | 0.4944 | -45.83% | 74.6409 | 128 | 106 | False | False | False | False |


公开版本仅包含源代码和脱敏研究摘要；数据、权重、私有本机证据不包含。
每日运行和安装前提见 docs/FORWARD_OPERATIONS.md；完整历史审计在本地保留，旧 Git 历史尚未获准公开。
