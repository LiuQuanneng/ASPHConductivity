# 导入必要的库
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_material_features_from_csv(input_csv, formula_column, output_csv=None):
    """
    从CSV文件读取材料数据，生成特征并保留原始所有列

    参数:
        input_csv: 输入CSV文件路径
        formula_column: 存储化学式的列名
        output_csv: 输出结果的CSV文件路径，默认为"material_features_with_all_columns.csv"
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"输入文件不存在: {input_csv}")

        # 读取CSV文件
        logging.info(f"读取CSV文件: {input_csv}")
        df = pd.read_csv(input_csv)

        # 检查化学式列是否存在
        if formula_column not in df.columns:
            raise ValueError(f"CSV文件中不存在列: {formula_column}")

        # 初始化Magpie特征生成器
        mp = ElementProperty.from_preset(preset_name="magpie")
        feature_names = mp.feature_labels()  # 获取特征名称列表

        # 为原始DataFrame添加特征列（初始化为NaN）
        for name in feature_names:
            df[name] = pd.NA

        # 遍历所有材料，生成特征并填充
        total = len(df)
        for i, row in df.iterrows():
            formula = row[formula_column]
            logging.info(f"处理材料 {i + 1}/{total}: {formula}")

            try:
                # 创建材料成分对象并生成特征
                composition = Composition(formula)
                features = mp.featurize(composition)

                # 将生成的特征填充到对应的行
                for name, value in zip(feature_names, features):
                    df.at[i, name] = value

            except Exception as e:
                logging.warning(f"处理材料 {formula} 时出错: {e}")
                continue

        # 保存结果
        if output_csv is None:
            output_csv = "material_features_with_all_columns.csv"

        df.to_csv(output_csv, index=False)
        logging.info(f"特征生成完成，结果已保存到: {output_csv}")

        return df

    except Exception as e:
        logging.error(f"处理CSV文件时出错: {e}")
        return None


# 示例使用
if __name__ == "__main__":
    # 输入CSV文件路径（使用原始字符串格式）
    input_file = "Li_SSE_metadata.csv"
    # 存储化学式的列名（根据你的CSV实际列名修改）
    formula_col = "formula"
    # 将结果保存到CSV文件
    results_csv = f'material_features_{input_file}'

    # 生成特征
    df = generate_material_features_from_csv(input_file, formula_col, results_csv)

    if df is not None:
        print(f"\n成功处理 {len(df)} 条数据")

