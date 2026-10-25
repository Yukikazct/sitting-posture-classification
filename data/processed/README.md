# 处理后数据

关键点特征缓存使用 `.npz`，包含 `features`、`labels`、`splits`、`sample_ids`、`views`，不使用 pickle。提取失败的样本以 NaN 特征保留，评估时按拒识计入。

缓存旁的 JSON 记录数据清单摘要与提取信息。对原始清单或标签的变更应重新生成缓存。
