# QUANT-SIDGTN V5：离线永续研究

V5实现独立事件账本、三个研究子策略及五期限SIDGTN。只允许离线研究；没有账户、交易密钥或在线交易入口。软件运行、Alpha检验、真实美国市场证据分开验收。最新数字以 `reports/V5_COMPLETION_REPORT.md`、`experiments/v5/validation.json` 和逐次命令回执为准。

在原工程根目录的普通PowerShell运行。直接使用已有 `.venv-training\Scripts\python.exe`，无需激活或改变执行策略。核心依赖锁为 `requirements-v4-core-lock.txt`；Freqtrade只在独立 `.venv-freqtrade` 中使用。远程已核验环境为Python3.12.10、PyTorch2.10.0+cu128、CUDA12.8、RTX4090。训练不会停止远程其他任务或安装系统服务。

公开Git仅提供脱敏源代码，原始档案、权重和机器日志保留在本地。以下研究复现命令要求本工程已保留的V4数据与对应哈希；公开clone没有这些私有本地产物时，不能声称已复现金融结论。原始价格档案由公开Kraken档案下载，资金原文来源与SHA见原工程V4 funding_export；真实美国资料由 `scripts/v5_public_audit.py`、`scripts/v5_us_data.py` 的匿名官方读取获得。不需要私有账户。

本轮实际执行过的基础命令：

```powershell
.venv-training\Scripts\python.exe scripts/v5.py dry-check
.venv-training\Scripts\python.exe scripts/v5.py prepare
.venv-training\Scripts\python.exe scripts/v5.py train
.venv-training\Scripts\python.exe scripts/v5_run.py old-core-diagnostic .venv-training\Scripts\python.exe scripts/v5_diagnose.py
.venv-training\Scripts\python.exe scripts/v5_reconcile_freqtrade.py
```

`prepare`核验已冻结数据，`train`准备/复用精确源码与数据作业包；这两个命令不偷偷覆盖原始快照或在本地代替4090训练。首次原工程构建已经依次运行 `v5_setup.py`、官方审计、`v5_us_data.py`、`v5_prepare.py`、`v5_build_job.py`；注册文件已有时不要为改参数重写它。新实验须新版本和新注册。

4090在既有授权会话中接收作业ZIP，先核验 `gpu_job.json` 的archive SHA，再解包至独立新目录；在该作业目录通过既有Python执行 `Run-V5.ps1`，其日志/checkpoint持久化。该启动方式本轮已实际运行，未改执行策略。断线后可以运行同目录脚本恢复；已完成拟合会先核验各文件再跳过。不要用本地GPU覆盖正式训练要求。

回传 `return.zip` 后核验命令如下。是否已执行成功见GPU验收回执；它核验所有清单项、60拟合及每个模型固定批次CPU/GPU差异，随后才生成统一预测表：

```powershell
.venv-training\Scripts\python.exe scripts/v5_run.py verify-4090 .venv-training\Scripts\python.exe scripts/v5_verify_gpu.py
.venv-training\Scripts\python.exe scripts/v5_run.py all-v5-comparisons .venv-training\Scripts\python.exe scripts/v5_backtest.py --workers 6
.venv-training\Scripts\python.exe scripts/v5_run.py matched-controls .venv-training\Scripts\python.exe scripts/v5_match_controls.py
.venv-training\Scripts\python.exe scripts/v5_run.py preregistered-validation .venv-training\Scripts\python.exe scripts/v5.py validate
.venv-training\Scripts\python.exe scripts/v5_run.py one-shot-observation .venv-training\Scripts\python.exe scripts/v5.py observe
.venv-training\Scripts\python.exe scripts/v5_run.py append-public-us-capture .venv-training\Scripts\python.exe scripts/v5_collect_us.py
```

`backtest`按完整试验清单运行，已有完整结果只核验不重做；源码/数据指纹不同会拒绝把新试验写进旧完成目录。`backtests/`保留首轮未激活进程离线上下文的24组结果；修正并测试守卫后正式结果在 `backtests_v2/`，初轮不能冒充守卫验证。没有新增网络行为，但该实现问题和重跑理由均保留。

`v5_collect_us.py`每次只读采集后退出，原文、实际接收时间、四时钟和结果哈希写入独立us_collections目录及追加账链，不覆盖第一次规格/历史快照。`observe`核验最新采集指针并分别输出真实规则控制器和SIDGTN推理；缺少闭合OHLC/标记价/保证金时继续阻断金融观察。最新采集仍仅BTC3、ETH6根闭合OHLC，不能因更新下载时间声称行情已变新。

自选有限区间的连续回放/恢复（CLI已验证；使用新的output防止覆盖）：

```powershell
.venv-training\Scripts\python.exe scripts/v5.py replay --output experiments/v5/readme_replay_guarded_v2 --start 2022-09-28 --end 2022-10-01 --mode filtered --cost BASE
.venv-training\Scripts\python.exe scripts/v5.py resume --output experiments/v5/readme_replay_guarded_v2 --start 2022-09-28 --end 2022-10-03 --mode filtered --cost BASE
```

恢复不重建钱包、不将多个末端平仓曲线拼接；最后一根不强制平仓。`observe`一次运行后退出，只记录真实模型推理与数据门；它不是bot。旧特征、稀疏美国OHLC或未知保证金不能开启合格前向金融窗口。三个月观察、12个独立28日窗口都只能随真实未来数据积累。

本轮已有定向测试回执；收尾完整回归命令和实际结果见完成报告。公开代码可先运行不依赖本地研究产物的账务/策略夹具：

```powershell
.venv-training\Scripts\python.exe -m pytest -q tests/test_v5_engine.py tests/test_v5_strategy.py
```

`tests/test_v5_governance.py`还要求实际GPU产物、完整比较和观察记录，不能在缺文件时自动跳过并写通过。合成夹具只验证软件。情绪历史/PIT事件日历/同步基差缺数据继续mask；动态融合只有真实新增分支通过增量门后才有资格另行运行。真实美国保证金与日内标记价未知时，现货/境外费率回放始终是SCENARIO_ONLY。

账务、接线和单位约定见 `docs/V5_ENGINE_AND_MODEL.md`；研究冲突决定见 `docs/V5_IMPLEMENTATION_DECISIONS.md`；交接见根目录 `HANDOFF.md`。所有失败命令保留，旧V4价格、权重、负结果和FINAL_CONSUMED不变。
