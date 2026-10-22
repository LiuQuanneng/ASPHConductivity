# ======= 在train调参shap11的基础上添加了保存最优模型的功能 =======
# ======= 解释器为python3.8 CRNN-for-battery-thermal-runaway-main =======

# 导入必要的库
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import os
import numpy as np
import warnings
import joblib  # 新增：用于保存模型和预处理对象

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# ===================== 核心参数配置=====================
CONFIG = {
    "shap_feature_threshold": "top_30%",  # SHAP特征筛选阈值
    "corr_threshold": 1.1,  # 特征相关性剔除阈值
    "test_size": 0.0,  # 无hold-out测试集，使用全数据进行交叉验证
    "base_random_state": 42,  # 基础随机种子（后续重复会在此基础上递增）
    "kfold_splits": 5,  # 5折（每折：80%训练、20%测试）
    "kfold_repeats": 5,  # 重复5次（每次用不同种子）
    "plot_loss_curves": True,  # 是否绘制loss曲线
    "loss_curve_epoch_step": 5  # loss曲线绘制的epoch步长
}


# ===================== Loss曲线绘制工具函数 =====================
def plot_loss_curves(train_losses, test_losses, model_name, save_path, data_basename, fold=None):
    """
    绘制训练/测试Loss曲线
    参数:
        train_losses: 训练集损失列表
        test_losses: 测试集损失列表（交叉验证中的测试折）
        model_name: 模型名称
        save_path: 保存路径
        data_basename: 数据集基础名称
        fold: 折数（可选）
    """
    if not CONFIG["plot_loss_curves"] or (len(train_losses) == 0 and len(test_losses) == 0):
        return

    plt.figure(figsize=(10, 6))

    step = CONFIG["loss_curve_epoch_step"]
    epochs = list(range(1, len(train_losses) + 1)) if len(train_losses) > 0 else list(range(1, len(test_losses) + 1))
    epochs_sampled = epochs[::step]
    train_losses_sampled = train_losses[::step] if len(train_losses) > 0 else []
    test_losses_sampled = test_losses[::step] if len(test_losses) > 0 else []

    if len(train_losses_sampled) > 0:
        plt.plot(epochs_sampled, train_losses_sampled,
                 label='Training Loss', color='#2ecc71', linewidth=2, marker='o', markersize=4)
    if len(test_losses_sampled) > 0:
        plt.plot(epochs_sampled, test_losses_sampled,
                 label='Test Loss', color='#e74c3c', linewidth=2, marker='s', markersize=4)

    fold_suffix = f"_fold{fold}" if fold is not None else ""
    plt.title(f'{model_name} Training Loss Curve{fold_suffix}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs/Iterations', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()

    loss_file = os.path.join(save_path,
                             f'loss_curve_{model_name.lower().replace(" ", "_")}{fold_suffix}_{data_basename}.png')
    plt.savefig(loss_file, dpi=300, bbox_inches='tight')
    print(f"✓ Loss曲线已保存：{loss_file}")
    plt.close()


# ===================== 统一模型训练函数 =====================
def train_model_with_loss_tracking(model, model_name, X_train, y_train, X_test=None, y_test=None):
    """
    训练模型并记录loss曲线
    参数:
        model: 模型对象
        model_name: 模型名称
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征（交叉验证中的测试折）
        y_test: 测试集标签
    返回:
        训练好的模型，训练loss列表，测试loss列表
    """
    train_losses = []
    test_losses = []

    if model_name == 'CatBoost':
        # CatBoost训练
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test) if X_test is not None else None,
            verbose=False,
            plot=False
        )

        # 获取训练/测试loss
        eval_result = model.get_evals_result()
        if 'learn' in eval_result and 'RMSE' in eval_result['learn']:
            train_losses = [x ** 2 for x in eval_result['learn']['RMSE']]
        if X_test is not None and 'validation' in eval_result and 'RMSE' in eval_result['validation']:
            test_losses = [x ** 2 for x in eval_result['validation']['RMSE']]

    elif model_name == 'XGBoost':
        # XGBoost训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)] if X_test is not None else [(X_train, y_train)],
            verbose=False
        )

        # 获取训练历史
        if hasattr(model, 'evals_result_'):
            eval_result = model.evals_result_
            if 'validation_0' in eval_result and 'rmse' in eval_result['validation_0']:
                train_losses = [x ** 2 for x in eval_result['validation_0']['rmse']]
            if X_test is not None and 'validation_1' in eval_result and 'rmse' in eval_result['validation_1']:
                test_losses = [x ** 2 for x in eval_result['validation_1']['rmse']]

    elif model_name == 'Gradient Boosting':
        # GradientBoosting训练并手动记录loss
        n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 100

        for i in range(1, n_estimators + 1, CONFIG["loss_curve_epoch_step"]):
            temp_model = GradientBoostingRegressor(
                n_estimators=i,
                max_depth=model.max_depth if hasattr(model, 'max_depth') else 3,
                learning_rate=model.learning_rate if hasattr(model, 'learning_rate') else 0.1,
                random_state=CONFIG["base_random_state"]
            )
            temp_model.fit(X_train, y_train)

            # 记录训练loss
            y_train_pred = temp_model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_losses.append(train_mse)

            # 记录测试loss
            if X_test is not None:
                y_test_pred = temp_model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_losses.append(test_mse)

        # 训练完整模型
        model.fit(X_train, y_train)

    else:
        # 其他模型直接训练
        model.fit(X_train, y_train)

    return model, train_losses, test_losses


# ===================== 1. 特征相关性分析与保存 =====================
def correlation_analysis_and_save(X, feature_names, save_path="./", data_basename="data"):
    """
    分析特征相关性，保存热图和CSV表格，并剔除高相关特征
    """
    print(f"\n{'=' * 60}")
    print("特征相关性分析")
    print(f"{'=' * 60}")

    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr()

    # 保存完整相关系数矩阵CSV
    corr_csv = os.path.join(save_path, f'correlation_matrix_{data_basename}.csv')
    corr_matrix.to_csv(corr_csv)
    print(f"✓ 相关系数矩阵已保存：{corr_csv}")

    # 绘制热图
    n_features = len(feature_names)
    if n_features > 50:
        print(f"⚠️ 特征数量较多({n_features})，热图仅显示前50个特征")
        display_size = 50
    else:
        display_size = n_features

    plt.figure(figsize=(max(12, display_size * 0.3), max(10, display_size * 0.25)))
    sns.heatmap(corr_matrix.iloc[:display_size, :display_size],
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                xticklabels=feature_names[:display_size],
                yticklabels=feature_names[:display_size])

    plt.title(f'Feature Correlation Heatmap (threshold={CONFIG["corr_threshold"]})',
              fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    heatmap_file = os.path.join(save_path, f'correlation_heatmap_{data_basename}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✓ 相关性热图已保存：{heatmap_file}")
    plt.close()

    # 找出高相关特征对
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = []
    for col in upper_triangle.columns:
        high_corr = upper_triangle[col][upper_triangle[col].abs() >= CONFIG["corr_threshold"]]
        for idx, corr_value in high_corr.items():
            high_corr_pairs.append({
                'Feature_1': idx,
                'Feature_2': col,
                'Correlation': corr_value
            })

    if len(high_corr_pairs) > 0:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            'Correlation', key=abs, ascending=False
        )
        high_corr_csv = os.path.join(save_path, f'high_correlation_pairs_{data_basename}.csv')
        high_corr_df.to_csv(high_corr_csv, index=False)
        print(f"✓ 高相关特征对已保存：{high_corr_csv} (共{len(high_corr_pairs)}对)")

    # 剔除高相关特征
    drop_features = [col for col in upper_triangle.columns
                     if any(upper_triangle[col].abs() >= CONFIG["corr_threshold"])]

    X_filtered = X_df.drop(columns=drop_features).values
    remaining_names = [name for name in feature_names if name not in drop_features]

    # 保存剔除记录
    if len(drop_features) > 0:
        drop_info = []
        for feat in drop_features:
            related = upper_triangle[feat][upper_triangle[feat].abs() >= CONFIG["corr_threshold"]]
            for rel_feat, corr_val in related.items():
                drop_info.append({
                    'Dropped_Feature': feat,
                    'Highly_Correlated_With': rel_feat,
                    'Correlation': corr_val
                })

        drop_df = pd.DataFrame(drop_info)
        drop_csv = os.path.join(save_path, f'dropped_features_{data_basename}.csv')
        drop_df.to_csv(drop_csv, index=False)
        print(f"✓ 剔除特征记录已保存：{drop_csv}")

    print(f"\n剔除结果:")
    print(f"  原始特征数: {len(feature_names)}")
    print(f"  剔除特征数: {len(drop_features)}")
    print(f"  剩余特征数: {len(remaining_names)}")

    return X_filtered, remaining_names, drop_features  # 新增：返回剔除的特征列表


# ===================== 2. SHAP特征重要性分析与筛选 =====================
def shap_feature_selection(X, y, feature_names,
                           save_path="./", data_basename="data"):
    """
    使用SHAP进行特征重要性分析和筛选
    """
    print(f"\n{'=' * 60}")
    print("SHAP特征重要性分析与筛选")
    print(f"{'=' * 60}")

    # 训练XGBoost模型
    print("正在训练XGBoost模型...")
    xgb_model = xgb.XGBRegressor(
        max_depth=5, n_estimators=100, learning_rate=0.1,
        random_state=CONFIG["base_random_state"], n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X, y)
    print("✓ 模型训练完成")

    # 计算SHAP值
    print("正在计算SHAP值（可能需要几分钟）...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    print("✓ SHAP值计算完成")

    # 计算特征重要性
    shap_importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'SHAP_Importance': shap_importance,
    }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)

    # 保存完整SHAP重要性
    shap_importance_csv = os.path.join(save_path,
                                       f'shap_importance_all_{data_basename}.csv')
    importance_df.to_csv(shap_importance_csv, index=False)
    print(f"✓ SHAP重要性已保存：{shap_importance_csv}")

    # 解析阈值并筛选特征
    threshold = CONFIG["shap_feature_threshold"]
    if isinstance(threshold, str):
        if threshold.endswith('%'):
            percent = int(threshold.replace('top_', '').replace('%', ''))
            n_features = max(1, int(len(feature_names) * percent / 100))
            selected_indices = importance_df.index[:n_features].tolist()
            threshold_desc = f"Top {percent}% ({n_features}个特征)"
        elif threshold.startswith('top_'):
            n_features = int(threshold.replace('top_', ''))
            n_features = min(n_features, len(feature_names))
            selected_indices = importance_df.index[:n_features].tolist()
            threshold_desc = f"Top {n_features}个特征"
        elif threshold == "mean":
            threshold_value = shap_importance.mean()
            selected_indices = importance_df[
                importance_df['SHAP_Importance'] >= threshold_value
                ].index.tolist()
            threshold_desc = f"≥均值 ({threshold_value:.6f})"
        elif threshold == "median":
            threshold_value = np.median(shap_importance)
            selected_indices = importance_df[
                importance_df['SHAP_Importance'] >= threshold_value
                ].index.tolist()
            threshold_desc = f"≥中位数 ({threshold_value:.6f})"
        else:
            threshold_value = float(threshold)
            selected_indices = importance_df[
                importance_df['SHAP_Importance'] >= threshold_value
                ].index.tolist()
            threshold_desc = f"≥{threshold_value:.6f}"
    else:
        threshold_value = float(threshold)
        selected_indices = importance_df[
            importance_df['SHAP_Importance'] >= threshold_value
            ].index.tolist()
        threshold_desc = f"≥{threshold_value:.6f}"

    selected_features = importance_df.iloc[selected_indices]
    selected_feature_names = selected_features['Feature_Name'].tolist()

    # 保存选中的特征
    selected_csv = os.path.join(save_path, f'shap_selected_features_{data_basename}.csv')
    selected_features.to_csv(selected_csv, index=False)
    print(f"✓ 选中特征已保存：{selected_csv}")

    # 筛选数据
    feature_mask = [name in selected_feature_names for name in feature_names]
    X_selected = X[:, feature_mask]

    print(f"\n特征筛选结果:")
    print(f"  筛选策略: {threshold_desc}")
    print(f"  原始特征数: {len(feature_names)}")
    print(f"  选中特征数: {len(selected_feature_names)}")
    print(f"  保留比例: {len(selected_feature_names) / len(feature_names) * 100:.1f}%")

    # 绘制SHAP可视化
    plot_shap_analysis(importance_df, selected_indices, shap_values,
                       X, feature_names, selected_feature_names,
                       save_path, data_basename)

    return X_selected, selected_feature_names, importance_df  # 新增：返回完整的SHAP重要性DF


def plot_shap_analysis(importance_df, selected_indices, shap_values,
                       X, feature_names, selected_feature_names,
                       save_path, data_basename):
    """绘制SHAP分析图（4张独立图片）"""

    # 1. SHAP重要性条形图（Top 30）
    plt.figure(figsize=(16, 10))
    n_show = min(30, len(importance_df))
    colors = ['#2ecc71' if i in selected_indices else '#95a5a6'
              for i in range(n_show)]

    y_pos = np.arange(n_show)
    plt.barh(y_pos, importance_df['SHAP_Importance'][:n_show], color=colors)
    plt.yticks(y_pos, importance_df['Feature_Name'][:n_show], fontsize=11)
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.title(f'Top {n_show} Features by SHAP Importance\n(Green=Selected)',
              fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plot_file1 = os.path.join(save_path, f'shap_importance_bar_{data_basename}.png')
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP重要性条形图已保存：{plot_file1}")
    plt.close()

    # 2. SHAP重要性分布
    plt.figure(figsize=(12, 8))
    plt.hist(importance_df['SHAP_Importance'], bins=50,
             color='skyblue', edgecolor='black', alpha=0.7)
    mean_val = importance_df['SHAP_Importance'].mean()
    median_val = importance_df['SHAP_Importance'].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                label=f'Median: {median_val:.4f}')
    plt.xlabel('SHAP Importance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('SHAP Importance Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_file2 = os.path.join(save_path, f'shap_importance_distribution_{data_basename}.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP重要性分布图已保存：{plot_file2}")
    plt.close()

    # 3. SHAP Summary Plot（确保排序一致）
    plt.figure(figsize=(16, 10))
    if len(selected_feature_names) <= 30:
        n_display = len(selected_feature_names)
        sorted_features = [f for f in importance_df['Feature_Name'].tolist()
                           if f in selected_feature_names][:n_display]
        title_suffix = "(Selected Features)"
    else:
        n_display = 20
        sorted_features = importance_df['Feature_Name'][:n_display].tolist()
        title_suffix = "(Top 20)"

    # 创建有序的索引映射
    ordered_indices = [feature_names.index(f) for f in sorted_features]

    # 按照条形图的排序提取SHAP值和特征值
    shap_values_ordered = shap_values[:, ordered_indices]
    X_ordered = X[:, ordered_indices]

    # 禁止SHAP内部重新排序
    shap.summary_plot(
        shap_values_ordered,
        X_ordered,
        feature_names=sorted_features,
        plot_type="dot",
        show=False,
        max_display=n_display,
        sort=False  # 禁用内部排序
    )

    plt.title(f'SHAP Summary Plot {title_suffix}\n(Ordered by Mean |SHAP Value|)',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    plot_file3 = os.path.join(save_path, f'shap_summary_plot_{data_basename}.png')
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP Summary图已保存：{plot_file3}")
    plt.close()

    # 4. 累计重要性曲线
    plt.figure(figsize=(12, 8))
    cumsum = np.cumsum(
        importance_df['SHAP_Importance'] / importance_df['SHAP_Importance'].sum()
    )
    plt.plot(range(1, len(cumsum) + 1), cumsum, linewidth=3, color='#3498db')
    plt.axhline(0.8, color='red', linestyle='--', linewidth=2, label='80%')
    plt.axhline(0.9, color='orange', linestyle='--', linewidth=2, label='90%')
    n_selected = len(selected_feature_names)
    if n_selected > 0:
        plt.axvline(n_selected, color='green', linestyle='--', linewidth=2,
                    label=f'Selected: {n_selected} ({cumsum[n_selected - 1] * 100:.1f}%)')
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Cumulative SHAP Importance', fontsize=14)
    plt.title('Cumulative Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_file4 = os.path.join(save_path, f'shap_cumulative_importance_{data_basename}.png')
    plt.savefig(plot_file4, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP累计重要性曲线已保存：{plot_file4}")
    plt.close()


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 加载数据
    data_filename = 'Can_topo_named_onedata.csv'
    data_basename = os.path.splitext(os.path.basename(data_filename))[0]
    df = pd.read_csv(data_filename)

    # 提取特征和标签
    feature_names = df.drop(columns=['Ionic Conductivity (Log)', 'ID']).columns.tolist()
    X_original = df.drop(columns=['Ionic Conductivity (Log)', 'ID']).values
    y = df['Ionic Conductivity (Log)'].values

    # 创建结果保存目录
    save_dir = f"results_{data_basename}"
    model_save_dir = os.path.join(save_dir, "best_model")  # 新增：模型保存子目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print(f"\n结果将保存到目录：{save_dir}")
    print(f"最优模型将保存到目录：{model_save_dir}")

    # ========== 步骤1：特征相关性分析 ==========
    X_corr_filtered, corr_remaining_names, dropped_features = correlation_analysis_and_save(  # 新增：接收dropped_features
        X_original, feature_names,
        save_path=save_dir,
        data_basename=data_basename
    )

    # 使用全数据
    X_full_data = X_corr_filtered
    y_full_data = y

    # 标准化
    scaler = StandardScaler()
    X_full_data_scaled = scaler.fit_transform(X_full_data)

    # ========== 步骤2：SHAP特征筛选 ==========
    X_full_data_selected, selected_feature_names, shap_importance_df = shap_feature_selection(
        # 新增：接收shap_importance_df
        X_full_data_scaled, y_full_data,
        feature_names=corr_remaining_names,
        save_path=save_dir,
        data_basename=data_basename
    )

    # ========== 步骤3：模型训练与交叉验证评估 ==========
    model_configs = [
        ('Decision Tree', DecisionTreeRegressor(random_state=CONFIG["base_random_state"])),
        ('Random Forest', RandomForestRegressor(random_state=CONFIG["base_random_state"])),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=CONFIG["base_random_state"])),
        ('Support Vector Machine', SVR()),
        ('XGBoost', xgb.XGBRegressor(
            max_depth=6, learning_rate=0.1, n_estimators=500,
            n_jobs=-1, random_state=CONFIG["base_random_state"],
            eval_metric='rmse'
        )),
        ('CatBoost', CatBoostRegressor(
            max_depth=7,
            learning_rate=0.01,
            n_estimators=2000,
            l2_leaf_reg=5,
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=50,
            eval_metric='R2',
            random_state=CONFIG["base_random_state"],
            verbose=False,
            loss_function='RMSE'
        )),
    ]

    results = []
    trained_models = []

    print("\n" + "=" * 70)
    print(f"开始模型训练与评估（{CONFIG['kfold_repeats']}次{CONFIG['kfold_splits']}折交叉验证）")
    print(f"每折划分：80%训练集 + 20%测试集")
    print(f"每次重复使用不同随机种子：{[CONFIG['base_random_state'] + i for i in range(CONFIG['kfold_repeats'])]}")
    print("=" * 70)

    for name, model_config in model_configs:
        print(f"\n{'=' * 30} {name} {'=' * 30}")

        # 训练最终模型（在全数据上）
        print("正在训练最终模型（使用全部数据）...")
        final_model, train_losses, test_losses = train_model_with_loss_tracking(
            model_config, name,
            X_full_data_selected, y_full_data,
            X_test=None, y_test=None
        )

        # 保存训练好的模型
        trained_models.append((name, final_model))

        # 绘制loss曲线（只有train_losses）
        if CONFIG["plot_loss_curves"]:
            plot_loss_curves(train_losses, test_losses, name, save_dir, data_basename)

        # 进行交叉验证评估（核心修改：不同重复使用不同随机种子）
        print(f"正在进行{CONFIG['kfold_repeats']}次{CONFIG['kfold_splits']}折交叉验证评估...")
        print(f"  每折：80%训练集 + 20%测试集")

        # 初始化存储所有折的指标
        cv_mse_scores = []
        cv_mae_scores = []
        cv_r2_scores = []
        cv_mape_scores = []

        # 核心修改：循环实现不同种子的重复交叉验证
        for repeat_idx in range(CONFIG["kfold_repeats"]):
            # 为每次重复生成不同的随机种子
            current_seed = CONFIG["base_random_state"] + repeat_idx
            print(f"  重复 {repeat_idx + 1}/{CONFIG['kfold_repeats']} | 随机种子: {current_seed}")

            # 初始化KFold（每次用不同种子）
            kf = KFold(
                n_splits=CONFIG["kfold_splits"],
                shuffle=True,  # 必须开启shuffle才能体现种子差异
                random_state=current_seed
            )

            # 遍历当前重复的所有折
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full_data_selected), 1):
                X_fold_train = X_full_data_selected[train_idx]
                y_fold_train = y_full_data[train_idx]
                X_fold_test = X_full_data_selected[test_idx]
                y_fold_test = y_full_data[test_idx]

                # 训练临时模型（克隆确保每次折都是全新模型）
                temp_model = clone(model_config)
                # 确保模型也使用当前折的随机种子（覆盖原有种子）
                if hasattr(temp_model, 'random_state'):
                    temp_model.random_state = current_seed + fold_idx  # 折级别也差异化
                temp_model.fit(X_fold_train, y_fold_train)

                # 在测试集上预测
                y_fold_test_pred = temp_model.predict(X_fold_test)

                # 计算测试集指标并存储
                cv_mse_scores.append(mean_squared_error(y_fold_test, y_fold_test_pred))
                cv_mae_scores.append(mean_absolute_error(y_fold_test, y_fold_test_pred))
                cv_r2_scores.append(r2_score(y_fold_test, y_fold_test_pred))
                cv_mape_scores.append(mean_absolute_percentage_error(y_fold_test, y_fold_test_pred))

        # 计算平均指标和标准差
        cv_mse = np.mean(cv_mse_scores)
        cv_mae = np.mean(cv_mae_scores)
        cv_r2 = np.mean(cv_r2_scores)
        cv_mape = np.mean(cv_mape_scores)

        cv_mse_std = np.std(cv_mse_scores)
        cv_r2_std = np.std(cv_r2_scores)

        print(f"交叉验证测试集结果（{CONFIG['kfold_repeats']}次{CONFIG['kfold_splits']}折）：")
        print(f"  MSE: {cv_mse:.4f} (±{cv_mse_std:.4f})")
        print(f"  MAE: {cv_mae:.4f}")
        print(f"  R²: {cv_r2:.4f} (±{cv_r2_std:.4f})")
        print(f"  MAPE: {cv_mape:.4f}")

        # 保存交叉验证详细结果
        cv_details_df = pd.DataFrame({
            'Repeat': np.repeat(range(1, CONFIG['kfold_repeats'] + 1), CONFIG['kfold_splits']),
            'Fold': list(range(1, CONFIG['kfold_splits'] + 1)) * CONFIG['kfold_repeats'],
            'Random_Seed': [CONFIG['base_random_state'] + (r - 1) for r in range(1, CONFIG['kfold_repeats'] + 1) for _
                            in range(CONFIG['kfold_splits'])],
            'Test_MSE': cv_mse_scores,
            'Test_MAE': cv_mae_scores,
            'Test_R2': cv_r2_scores,
            'Test_MAPE': cv_mape_scores
        })
        cv_details_file = os.path.join(save_dir, f'cv_details_{name.lower().replace(" ", "_")}_{data_basename}.csv')
        cv_details_df.to_csv(cv_details_file, index=False)
        print(f"✓ 交叉验证详细结果已保存：{cv_details_file}")

        # 保存汇总结果
        results.append([
            name, cv_mse, cv_mae, cv_r2, cv_mape,
            cv_mse_std, cv_r2_std
        ])

    # ========== 步骤4：筛选并保存最优模型 ==========
    print("\n" + "=" * 70)
    print("筛选最优模型（基于R²评分）")
    print("=" * 70)

    # 将结果转换为DataFrame便于筛选
    results_df = pd.DataFrame(results,
                              columns=['Model', 'CV_Test_Avg_MSE', 'CV_Test_Avg_MAE',
                                       'CV_Test_Avg_R2', 'CV_Test_Avg_MAPE',
                                       'CV_Test_MSE_Std', 'CV_Test_R2_Std'])

    # 按R²降序排序，选择R²最高的模型作为最优模型
    results_df_sorted = results_df.sort_values('CV_Test_Avg_R2', ascending=False)
    best_model_name = results_df_sorted.iloc[0]['Model']
    best_model_metrics = results_df_sorted.iloc[0]

    print(f"\n最优模型：{best_model_name}")
    print(f"最优模型指标：")
    print(f"  平均R²: {best_model_metrics['CV_Test_Avg_R2']:.4f}")
    print(f"  平均MSE: {best_model_metrics['CV_Test_Avg_MSE']:.4f}")
    print(f"  平均MAE: {best_model_metrics['CV_Test_Avg_MAE']:.4f}")

    # 找到对应的训练好的模型对象
    best_model = [model for name, model in trained_models if name == best_model_name][0]

    # 保存最优模型
    model_save_path = os.path.join(model_save_dir, f"best_model_{best_model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"\n✓ 最优模型已保存：{model_save_path}")

    # 保存标准化器（预测时需要用相同的scaler）
    scaler_save_path = os.path.join(model_save_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_save_path)
    print(f"✓ 标准化器已保存：{scaler_save_path}")

    # 保存特征相关信息（预测时需要对齐特征）
    feature_info = {
        "original_feature_names": feature_names,
        "dropped_features": dropped_features,
        "corr_remaining_names": corr_remaining_names,
        "selected_feature_names": selected_feature_names,
        "shap_feature_threshold": CONFIG["shap_feature_threshold"],
        "corr_threshold": CONFIG["corr_threshold"]
    }
    feature_info_save_path = os.path.join(model_save_dir, "feature_info.joblib")
    joblib.dump(feature_info, feature_info_save_path)
    print(f"✓ 特征信息已保存：{feature_info_save_path}")

    # 保存SHAP重要性数据（预测时特征筛选需要）
    shap_importance_save_path = os.path.join(model_save_dir, "shap_importance.csv")
    shap_importance_df.to_csv(shap_importance_save_path, index=False)
    print(f"✓ SHAP重要性数据已保存：{shap_importance_save_path}")

    # 保存最优模型的评估指标
    best_metrics_save_path = os.path.join(model_save_dir, "best_model_metrics.csv")
    best_model_metrics.to_frame().T.to_csv(best_metrics_save_path, index=False)
    print(f"✓ 最优模型指标已保存：{best_metrics_save_path}")

    # ========== 步骤5：可视化 ==========
    print("\n正在生成可视化结果...")
    print("\n正在生成交叉验证稳定性分析图...")

    # 创建交叉验证稳定性分析图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = [result[0] for result in results]
    mse_stability = [result[5] for result in results]
    r2_stability = [result[6] for result in results]
    avg_mse = [result[1] for result in results]
    avg_r2 = [result[3] for result in results]

    # 子图1：MSE稳定性
    ax1 = axes[0, 0]
    ax1.barh(model_names, mse_stability, color='lightcoral')
    ax1.set_xlabel('MSE Standard Deviation', fontsize=12)
    ax1.set_title('Cross-Validation MSE Stability\n(Lower is better)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 子图2：R2稳定性
    ax2 = axes[0, 1]
    ax2.barh(model_names, r2_stability, color='lightblue')
    ax2.set_xlabel('R² Standard Deviation', fontsize=12)
    ax2.set_title('Cross-Validation R² Stability\n(Lower is better)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 子图3：MSE vs R2散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(avg_mse, avg_r2, s=100, alpha=0.7)
    ax3.set_xlabel('Average Test MSE', fontsize=12)
    ax3.set_ylabel('Average Test R²', fontsize=12)
    ax3.set_title('Performance vs Stability', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)

    for i, name in enumerate(model_names):
        ax3.annotate(name, (avg_mse[i], avg_r2[i]), fontsize=9)

    # 子图4：性能对比条形图
    ax4 = axes[1, 1]
    x = np.arange(len(model_names))
    width = 0.35

    normalized_mse = 1 - (np.array(avg_mse) / max(avg_mse))
    normalized_r2 = np.array(avg_r2) / max(avg_r2)

    bars1 = ax4.bar(x - width / 2, normalized_mse, width,
                    label='Normalized MSE (1 - MSE/max)', color='lightcoral')
    bars2 = ax4.bar(x + width / 2, normalized_r2, width,
                    label='Normalized R²', color='lightblue')

    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Normalized Performance', fontsize=12)
    ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    stability_file = os.path.join(save_dir, f'cv_stability_analysis_{data_basename}.png')
    plt.savefig(stability_file, dpi=300, bbox_inches='tight')
    print(f"✓ 交叉验证稳定性分析图已保存：{stability_file}")
    plt.close()

    # 保存结果到CSV
    results_csv = os.path.join(save_dir, f'model_results_{data_basename}.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ 模型结果已保存：{results_csv}")

    # ========== 最终总结 ==========
    print(f"\n{'=' * 70}")
    print(f"所有分析完成！")
    print(f"交叉验证策略：{CONFIG['kfold_repeats']}次{CONFIG['kfold_splits']}折（每次重复使用不同随机种子）")
    print(f"使用的随机种子序列：{[CONFIG['base_random_state'] + i for i in range(CONFIG['kfold_repeats'])]}")
    print(f"每折划分：80%训练集（用于训练模型） + 20%测试集（用于评估性能）")
    print(f"最优模型：{best_model_name} (R²={best_model_metrics['CV_Test_Avg_R2']:.4f})")
    print(f"{'=' * 70}")
    print(f"保存目录：{save_dir}")
    print(f"模型保存目录：{model_save_dir}")
    print(f"\n包含文件：")
    print(f"  1. correlation_matrix_{data_basename}.csv - 特征相关系数矩阵")
    print(f"  2. correlation_heatmap_{data_basename}.png - 相关性热图")
    print(f"  3. high_correlation_pairs_{data_basename}.csv - 高相关特征对")
    print(f"  4. dropped_features_{data_basename}.csv - 剔除特征记录")
    print(f"  5. shap_importance_all_{data_basename}.csv - SHAP重要性（全部）")
    print(f"  6. shap_selected_features_{data_basename}.csv - SHAP筛选特征")
    print(f"  7. shap_importance_bar_{data_basename}.png - SHAP重要性条形图")
    print(f"  8. shap_importance_distribution_{data_basename}.png - SHAP重要性分布图")
    print(f"  9. shap_summary_plot_{data_basename}.png - SHAP Summary图")
    print(f"  10. shap_cumulative_importance_{data_basename}.png - SHAP累计重要性曲线")
    print(f"  11. model_results_{data_basename}.csv - 所有模型测试集评估结果")
    print(f"  12. cv_details_*.csv - 每个模型的详细测试集结果（每折，含种子信息）")
    print(f"  13. cv_stability_analysis_{data_basename}.png - 交叉验证稳定性分析图")
    print(f"  14. loss_curve_*.png - 各模型训练Loss曲线")
    print(f"  15. best_model/*.joblib - 最优模型、标准化器、特征信息")
    print(f"  16. best_model/best_model_metrics.csv - 最优模型评估指标")