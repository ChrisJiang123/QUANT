# V5 统一完成报告

1. 软件：五期限SIDGTN、真实辅助因子/图分支、三仓策略和离线事件账本已实现并运行；真实美国完整账务与合格前向仍受数据限制。最终测试274项，失败0、错误0、跳过0。
2. SIDGTN：core、factor、graph各5种子，3年度WF和1仅DEV最终拟合，共60次真实4090拟合。情绪真实历史不足，金融实验BLOCKED_DATA；动态融合未取得门槛资格时保持禁用。
3. 相对规则：NO_GO_ALPHA。14/28日三个变体均未改善零预测MAE；没有据最佳种子、映射或次要期限改换主检验。金融比较、原始/校准/中位方向和四格替换全部保留。
4. 4090：权重、指标、日志、checkpoint和manifest已回传核验；固定CPU/GPU最大绝对差7.1525574e-07。训练源码为冻结作业SHA，不用后续报告提交冒充训练版本。
5. 最重要下一步：补齐带真实接收时间的美国合约报价/标记价与保证金资料，给已冻结SIDGTN建立新的可审计前向窗口；当前不升级、不进入实盘。

## 四个维度

| 维度 | 结论 | 最简单下一步 |
|---|---|---|
| 软件 | 已实现和验证；数据门保守阻断 | 按README_V5运行dry-check/验证/恢复 |
| Alpha | NO_GO_ALPHA；独立终测INCONCLUSIVE_DATA | 保留负结果；不重置FINAL_CONSUMED |
| 集成 | 专用账本+Freqtrade可比子集对账；美国规格部分未核实 | 补齐实际美元保证金/日内标记价 |
| 远程GPU | VERIFIED，60拟合+完整回传与固定批次核验 | 复用冻结作业，无需重复训练 |

## 需求状态与证据

| 需求ID | 软件状态 | 运行 | 金融证据 | 证据 |
|---|---|---|---|---|
| V5-SPEC | IMPLEMENTED_AND_TESTED | COMPLETED | UNVERIFIED_VENUE_SPEC | experiments/v5/instrument_specs.json; experiments/v5/public_sources/index.json |
| V5-DATA | IMPLEMENTED_AND_TESTED | COMPLETED | PARTIAL_COVERAGE_SCENARIO_ONLY | experiments/v5/data/manifest.json; experiments/v5/us_data/coverage.json |
| V5-GOV | IMPLEMENTED_AND_TESTED | COMPLETED | NO_CLEAN_FINAL | experiments/v5/registration.json; experiments/v5/tests-governance-ready.xml |
| V5-ENGINE | IMPLEMENTED_AND_TESTED | COMPLETED | SCENARIO_ENGINE_ONLY_NOT_US_VENUE_VALIDATION | experiments/v5/tests-engine-first.xml; experiments/v5/freqtrade_reconciliation/metrics.json |
| V5-TREND | IMPLEMENTED_AND_TESTED | COMPLETED | SCENARIO_ONLY | experiments/v5/tests-pre-finance.xml; experiments/v5/rules_plan.json |
| V5-CARRY | IMPLEMENTED_AND_TESTED | COMPLETED | SCENARIO_ONLY | experiments/v5/tests-pre-finance.xml; experiments/v5/rules_plan.json |
| V5-TACTICAL | IMPLEMENTED_AND_TESTED | COMPLETED | SCENARIO_ONLY | experiments/v5/tests-pre-finance.xml; experiments/v5/rules_plan.json |
| V5-RISK | IMPLEMENTED_AND_TESTED | COMPLETED | SCENARIO_ONLY | experiments/v5/tests-pre-finance.xml; experiments/v5/rules_plan.json |
| V5-DIAG | IMPLEMENTED_AND_TESTED | COMPLETED | NO_GO_ALPHA | experiments/v5/validation.json; experiments/v5/legacy_diagnostic.json |
| V5-CORE | IMPLEMENTED_AND_TESTED | COMPLETED | NO_GO_ALPHA | experiments/v5/gpu_verification.json; experiments/v5/validation.json |
| V5-FACTOR | IMPLEMENTED_AND_TESTED | COMPLETED | NO_GO_ALPHA | experiments/v5/gpu_verification.json; experiments/v5/validation.json |
| V5-GRAPH | IMPLEMENTED_AND_TESTED | COMPLETED | NO_GO_ALPHA | experiments/v5/gpu_verification.json; experiments/v5/validation.json |
| V5-SENTIMENT | IMPLEMENTED_AND_TESTED | ADAPTER_FALLBACK_GRADIENT_TESTS_ONLY; REAL_FINANCIAL_NOT_RUN | BLOCKED_DATA | experiments/v5/data/blocked_modalities.json; experiments/v5/tests-pre-finance.xml |
| V5-FUSION | IMPLEMENTED_AND_TESTED | GATED_DISABLED | NO_INCREMENTAL_EVIDENCE | experiments/v5/validation.json; experiments/v5/tests-pre-finance.xml |
| V5-GPU | IMPLEMENTED_AND_TESTED | COMPLETED | TRAINING_VERIFIED; NO_GO_ALPHA_IS_SEPARATE | experiments/v5/gpu_verification.json; experiments/v5/validation.json |
| V5-VAL | IMPLEMENTED_AND_TESTED | COMPLETED | NO_GO_ALPHA | experiments/v5/validation.json; experiments/v5/legacy_diagnostic.json |
| V5-OBS | IMPLEMENTED_AND_TESTED | COMPLETED_DATA_GATE_BLOCKED | BLOCKED_US_FORWARD | experiments\v5\observations\20261115T100930289185Z\result.json |
| V5-TEST | IMPLEMENTED_AND_TESTED | COMPLETED | SOFTWARE_TESTS_ONLY | experiments\v5\tests-final-with-collection.xml; experiments/v5/test_summary.json |

## 验收边界

共同DEV为2022—2024的1096日，只是现货价格/境外费率/假设美国规格情景。真实美国OHLC仅BTC3根、ETH6根，费率各543行；历史保证金和完整标记价未知。假设下无强平不能证明美国生存能力。所有年份、状态、压力窗口、成本分解、资金收支、止损/熔断/强平和五种子细节由metrics生成，见reports/v5/ABLATIONS.md和原始逐次产物。

G1/HAC/Holm、G2最强简单基线/28日配对90%区间/DSR、G3五种子和压力、G4独立未来窗口分别记录。DSR保守计入311次已记录研究比较/拟合及历史试验；没有将交叠的资产/日标签当独立样本。干净final拒绝执行，合格独立窗口0/12；三个月观察也尚未发生。

观察入口实际加载已核验最终DEV权重，输出30条真实推理；旧代理特征和稀疏US行情触发数据门，合格前向行0，BLOCKED_US_FORWARD，没有虚构交易或净值。趋势/套利/短周期归因独立，短周期仍默认不进主组合；套利净成本门不为凑交易放宽。

LocalCTO只读实查：控制器正常、既有4090节点在线，当前固定health/Ollama可用；所给文档与代码仍将QUANT研究/SSH/文件适配标未部署。没有把健康结果当训练结果，没有改权限/服务。此次QUANT真实训练经已授权RustDesk执行，续传跨两次10054中断后完成。具体通道检查在本地experiments/v5/localcto_channel_probe.json，不公开连接信息。

## 验证、保留和复现

实际命令/退出码见reports/v5/COMMANDS.md及experiments/v5/commands；最终完整回归XML为experiments\v5\tests-final-with-collection.xml，定向失败与修正均保留。README_V5给出本机验证过的dry-check、prepare、作业准备、回传核验、验证、观察、离线重放与恢复命令。新增公开US采集已实际执行，四时钟与闭合K线门经过测试；每次保存独立原文/哈希，不覆盖首轮快照。最新原文仍未提供足够新鲜OHLC/标记价/保证金，重复下载不能增加合格窗口。未进行真实资金操作、重启/72h远程可靠性测试、真实美国完整强平验证、情绪历史金融实验或新的独立终测，均不得称已通过。

V4冻结98文件和旧报告/消费标记逐哈希保留；旧规则轨11月15日按原代码实际运行，3个真实UTC日、0成熟独立窗口，仍有DRIFT，仅属旧轨运维证据。首次回放守卫上下文缺陷已修正并作socket注入拒绝测试；初轮结果保留，正式结果在backtests_v2。无自动跳过失败或调整收益口径。

代码小步提交保留本地私有历史。公开发布只在已核验干净main祖先上增加脱敏V5源代码/本报告，不带原始数据、权重、日志、用户Prompt、连接信息或私有祖先；具体推送/PR结果以本地experiments/v5/publication.json为准，不自动合并。
