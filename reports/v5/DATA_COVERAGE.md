# V5 数据覆盖

| 资产 | 真实US OHLC根数 | OHLC起止 | 资金行数 | 等级 |
|---|---|---|---|---|
| XBTUSD | 3 | 2026-10-16—2026-10-30 | 543 | REAL_US_PERP_PARTIAL |
| ETHUSD | 6 | 2026-10-20—2026-11-10 | 543 | REAL_US_PERP_PARTIAL |

真实US绝对历史保证金、日内标记价/强平规则、资金原始价格单位未完全核实。产品ID/乘数/tick/当前按张费已有官方来源原文与哈希。不能据这些短而稀疏的样本报告完整美国回测或生存能力。

现货价格有长历史；BTC/ETH资金区间各自只覆盖2018-09-01至2026-01-31，共2704个完整日，不是十年真实美国永续。共同模型DEV为2022—2024，1096日；缺区间拒绝主回放，不补零。历史收盘+6h为保守推定，真实前向必须用实际接收时刻。

模型样本8112，八节点mask按各节点真实首根/缺失。基础因子8、辅助8，其中基差/事件未伪造；新闻0真实有效样本。固定节点集合有幸存者选择局限，不能称历史全市场。GDELT三次429、日历vintage、同步基差的缺口见data/blocked_modalities.json。

官方入口：[美国产品](https://support.kraken.com/articles/us-perpetual-futures)、[费用](https://support.kraken.com/articles/us-futures-fees)、[规格](https://support.kraken.com/articles/contract-specifications)。实际页面快照与访问时刻在experiments/v5/public_sources。
