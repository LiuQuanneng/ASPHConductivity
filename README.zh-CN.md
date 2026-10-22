# ASPHConductivity

本仓库提供论文《Topological descriptors for improved prediction of ionic conductivity in solid-state electrolytes》的训练数据、拓扑描述符代码、模型评价代码及模型文件。

## 结果

完整描述符CatBoost模型的MAE为0.804、R²为0.721。相对Magpie-only CatBoost，MAE降低5.7%，R²相对提高5.8%。图3的七组数据、六种算法共42组结果和1,050条折级记录位于 `publication_fixed_S1/data/`。

## 使用

使用Python 3.12，安装 `requirements-tested-reproduction.txt`。运行 `python scripts/check_fixed_S1_publication.py` 检查结果；运行 `python scripts/reproduce_paper_fixed_S1.py --execute --output-dir runs/evaluation` 进行固定参数评价。完整命令见英文README。

`src/descriptors/` 包含结构读取、扩胞、点云、Ripser持久同调和拓扑统计特征代码，以及Magpie组成特征生成代码。结构输入和计算流程见 `docs/METHODS.md`。

匹配的最终模型、标准化器和特征顺序位于 `publication_fixed_S1/model/`。候选特征及3,082条预测位于 `publication_fixed_S1/data/`。原样式图3图片和可编辑PPT位于 `publication_fixed_S1/figures/Fig3_original_style.*`。
