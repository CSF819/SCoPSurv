# 文件索引

- `checkpoint_sources.json`：25 个 checkpoint 的仓库内相对路径、源文件位置和原始结果摘要中的精确 C-index。源位置用于追溯文件来源，运行时通过相对路径读取。
- `checkpoint_summary.csv`：各折最优轮次、checkpoint 内保存的验证 C-index、严格参数加载结果和合成输入前向检查结果。`evaluated_validation_c_index` 留空，表示这份清单没有重新计算真实患者的验证 C-index。
- `data_summary.json`：各基因表的行列数、患者数、六组基因维度以及 25 份患者划分的交集和重叠检查。

CSV 中的 `best_epoch` 与 checkpoint 保存值及原始文件名一致。五折汇总采用 checkpoint 的历史验证分数，各癌种各折来自同一训练配置。
