# 规则主轨每日运行

仅采集免费公共行情并记录文件；交易和订单开关为 false。不会启动在线 dry-run bot。
新轨的规则、成本及代码归档由首次运行前的 `experiments/rule_forward_v1/freeze.json`
和 `code.zip` 固定。修改冻结文件会拒绝运行，规则修改必须另立新轨。

普通账户在工程根目录手动运行，无需修改执行策略：

```powershell
powershell.exe -NoProfile -File .\ops\run_forward_daily.ps1
```

需已有 `.venv-training` 和 `.venv-freqtrade` 隔离环境。前者使用现有研究依赖，后者
使用已锁定的 Freqtrade 2026.10。依赖和许可沿用 `ENV_SETUP.md`、第三方许可文件。
不需要 Docker、WSL、4090 或本地 LLM。采集层直连 Kraken 免费公共 OHLC 接口，
禁用环境代理；处理阶段进入 offline_guard，回测另由现有进程级守卫拒绝外部网络。

日志在 `experiments/rule_forward_v1/entry_logs/`；每次结果及真实响应、快照、哈希在
`experiments/rule_forward_v1/runs/<run_id>/`。`latest.json` 仅是最新结果指针，
观察、结算、运行台账均为哈希链追加文件。每次运行先验证冻结代码和链，失败退出码非零。
同一轨和已收盘日重复运行不会重复预测；源价格有修正时另追加 correction，原记录保留，
该观察及其修订不计入干净前向证据。旧失败 SIDGTN 台账继续单独保留，不作为主轨。

每根行情在响应完整接收后才可见。较早行情仅作补录上下文，不能冒充历史前向预测。
信号在真实生成时间之后的下一个 UTC 开盘才可执行。14/28 天标签采用该开盘至
14/28 天后开盘的价格；结算还等待结束日 K 线收盘并实际收到，故比市场到期至少晚一天。
未成熟标签列 pending 和最早到期/可结算时间，没有占位零收益。

每日目标权重沿用固定趋势窗口 28/60/120、目标波动 .15、单资产上限 .5、14 天固定
日历再平衡、.05 带宽和 28 天持有上限。首个计划再平衡前持有研究现金。所有净值、
成本与规则/BH/等平均仓位比较均使用已有 Freqtrade 离线引擎；未来可成交日线尚未
收齐时报告 PENDING_EXECUTABLE_CLOSED_BARS，净值和收益指标为缺失。初始现金状态
10000、零成交、零已发生成本只是初始账务事实。每次引擎回放是封存的离线重建，
末端强制平仓是引擎约定；每日重建快照不能直接拼成一条连续可投资曲线。

只导出用户级每日 06:15 **本地时间**的配置，不启用无人值守任务：

```powershell
powershell.exe -NoProfile -File .\ops\export_forward_schedule.ps1
```

经用户以后明确授权无人值守调度后，可由该用户手动注册：

```powershell
Register-ScheduledTask -TaskName 'QUANT-Rule-Observation' -Xml (Get-Content -Raw .\experiments\rule_forward_v1\schedule.xml)
```

配置采用当前普通账户 InteractiveToken/LeastPrivilege，不保存密码，仅登录时执行；
不会提权，不创建系统服务。机器需开机、登录且联网；配置不唤醒、不补跑错过的定时触发。
休眠、断网后手动重新执行每日命令，只观察当时最新已收盘日，不回填错过的预测。
失败结果保留；修复网络后同键重跑幂等。链损坏先停止并核对保留证据，不能截断台账。
进程崩溃留下 transaction-lock/lock 时，先确认对应 PID 已终止并备份台账，再由用户
只移除确认为孤立的锁文件，不自动删除未知锁。禁止删除数据或重新冻结已有轨以消除失败。

停用或删除已由用户注册的任务：

```powershell
Disable-ScheduledTask -TaskName 'QUANT-Rule-Observation'
Unregister-ScheduledTask -TaskName 'QUANT-Rule-Observation' -Confirm:$false
```

验收：每天检查最新结果 SUCCESS、最新收盘日、接收/生成/可执行时间、链完整、无重复
新观察、待结算数和数据/漂移报警。实际连续 7 个 UTC 日运行后才完成运维验收；重复
运行和夹具不增加天数。7 天不是 Alpha 验收。12 个不重叠 28 天窗口约需 336 天，
两币同期不重复计数；完整窗口还需 28 天连续实际观察和两币成熟标签。旧已触碰历史
不计入新轨。NO_GO_ALPHA 不因观察期变长自动解除。

Kraken 当前行与最近 720 根限制见
[官方接口说明](https://docs.kraken.com/api-reference/market-data/get-ohlc-data)。
调度 XML 含本机账户与路径，仅保留本地，不公开。日志、真实行情和权重也不上传公开仓库。
