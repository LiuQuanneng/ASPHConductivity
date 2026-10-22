# ============================================================================
# 最终版：6个模型（决策树、随机森林、梯度提升、SVM、XGBoost、CatBoost）
# 选择性超参数调优：XGBoost 和 CatBoost 使用 Optuna，其余使用固定参数
# 功能包括：
# 1. 特征相关性分析（剔除高相关）→ 热图、剔除记录
# 2. SHAP 特征重要性筛选（top 30%）→ 重要性条形图、summary plot（带颜色条）、累计曲线
# 3. 6个模型的交叉验证评估（5次重复×5折）
# 4. 每个模型：训练/验证 MSE 曲线（同图，基于第一个 fold）
# 5. 所有模型 Predicted vs Actual 散点图（自动适应6个模型，3×2布局）
# 6. 输出所有结果表：超参数汇总、交叉验证细节、模型性能汇总
# 7. 保存最优模型及配套文件
# 8. 固定所有随机种子，结果可重现
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
import random
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import shap
import optuna
from optuna.samplers import TPESampler

# ---------------------- 固定随机种子 ----------------------
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)
os.environ['PYTHONHASHSEED'] = str(SEED_GLOBAL)

warnings.filterwarnings('ignore')

# 绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# ---------------------- 参数配置 ----------------------
CONFIG = {
    "shap_feature_threshold": "top_30%",  # SHAP 筛选保留 top 30% 特征
    "corr_threshold": 0.95,  # 相关性剔除阈值
    "base_random_state": SEED_GLOBAL,
    "kfold_splits": 5,
    "kfold_repeats": 5,
    "plot_loss_curves": True,
    "loss_curve_epoch_step": 5
}

OPTUNA_CONFIG = {
    "n_trials": 30,  # Optuna 搜索次数（可调小至 15 以加速）
    "cv_folds": 3,  # Optuna 内部交叉验证折数
    "n_jobs": -1
}

# 只有 XGBoost 和 CatBoost 需要精细调优
FINE_TUNE_MODELS = ['XGBoost', 'CatBoost']

# 其他模型的默认参数（决策树、随机森林、梯度提升、SVM）
DEFAULT_PARAMS = {
    'Decision Tree': {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
    'Random Forest': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
    'Gradient Boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
    'Support Vector Machine': {'C': 10, 'epsilon': 0.1, 'kernel': 'rbf'}
}


# ---------------------- 辅助函数 ----------------------
def plot_loss_curves(train_losses, test_losses, model_name, save_path, data_basename, fold=None):
    """绘制训练/验证 MSE 曲线（同一张图）"""
    if not CONFIG["plot_loss_curves"] or (len(train_losses) == 0 and len(test_losses) == 0):
        return
    plt.figure(figsize=(10, 6))
    step = max(1, CONFIG["loss_curve_epoch_step"])
    if len(train_losses) > 0:
        epochs = list(range(1, len(train_losses) + 1))[::step]
        plt.plot(epochs, train_losses[::step], label='Training MSE', color='#2ecc71', marker='o', markersize=4)
    if len(test_losses) > 0:
        epochs = list(range(1, len(test_losses) + 1))[::step]
        plt.plot(epochs, test_losses[::step], label='Validation MSE', color='#e74c3c', marker='s', markersize=4)
    fold_suffix = f"_fold{fold}" if fold else ""
    plt.title(f'{model_name} Training & Validation MSE Curve{fold_suffix}')
    plt.xlabel('Epochs/Iterations')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_path, f'loss_curve_{model_name.lower().replace(" ", "_")}{fold_suffix}_{data_basename}.png'),
        dpi=300)
    plt.close()


def train_model_with_loss_tracking(model, model_name, X_train, y_train, X_test=None, y_test=None):
    """训练模型并记录训练/验证 MSE 曲线（支持 CatBoost, XGBoost, GBDT）"""
    train_losses, test_losses = [], []
    if model_name == 'CatBoost':
        model.fit(X_train, y_train, eval_set=(X_test, y_test) if X_test is not None else None, verbose=False,
                  plot=False)
        ev = model.get_evals_result()
        if 'learn' in ev and 'RMSE' in ev['learn']:
            train_losses = [x ** 2 for x in ev['learn']['RMSE']]
        if X_test is not None and 'validation' in ev and 'RMSE' in ev['validation']:
            test_losses = [x ** 2 for x in ev['validation']['RMSE']]
    elif model_name == 'XGBoost':
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_test, y_test)] if X_test is not None else [(X_train, y_train)],
                  verbose=False)
        if hasattr(model, 'evals_result_'):
            ev = model.evals_result_
            if 'validation_0' in ev and 'rmse' in ev['validation_0']:
                train_losses = [x ** 2 for x in ev['validation_0']['rmse']]
            if X_test is not None and 'validation_1' in ev and 'rmse' in ev['validation_1']:
                test_losses = [x ** 2 for x in ev['validation_1']['rmse']]
    elif model_name == 'Gradient Boosting':
        n_est = model.n_estimators if hasattr(model, 'n_estimators') else 100
        for i in range(1, n_est + 1, CONFIG["loss_curve_epoch_step"]):
            tmp = GradientBoostingRegressor(n_estimators=i, max_depth=model.max_depth,
                                            learning_rate=model.learning_rate, random_state=CONFIG["base_random_state"])
            tmp.fit(X_train, y_train)
            train_losses.append(mean_squared_error(y_train, tmp.predict(X_train)))
            if X_test is not None:
                test_losses.append(mean_squared_error(y_test, tmp.predict(X_test)))
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
        if X_test is not None:
            train_losses = [mean_squared_error(y_train, model.predict(X_train))]
            test_losses = [mean_squared_error(y_test, model.predict(X_test))]
    return model, train_losses, test_losses


def correlation_analysis_and_save(X, feature_names, save_path, data_basename):
    """特征相关性分析、剔除高相关特征、保存热图和剔除记录"""
    print("\n=== 特征相关性分析与剔除 ===")
    X_df = pd.DataFrame(X, columns=feature_names)
    corr = X_df.corr()
    corr.to_csv(os.path.join(save_path, f'correlation_matrix_{data_basename}.csv'))
    # 热图
    n_display = min(50, len(feature_names))
    plt.figure(figsize=(max(12, n_display * 0.3), max(10, n_display * 0.25)))
    sns.heatmap(corr.iloc[:n_display, :n_display], cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title(f'Feature Correlations (threshold={CONFIG["corr_threshold"]})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'correlation_heatmap_{data_basename}.png'), dpi=300)
    plt.close()
    # 剔除高相关特征
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col].abs() >= CONFIG["corr_threshold"])]
    X_filtered = X_df.drop(columns=drop_cols).values
    keep_names = [c for c in feature_names if c not in drop_cols]
    if drop_cols:
        drop_info = []
        for col in drop_cols:
            related = upper[col][upper[col].abs() >= CONFIG["corr_threshold"]]
            drop_info.append({'Dropped_Feature': col, 'Highly_Correlated_With': ', '.join(related.index),
                              'Correlation': ', '.join(map(str, related.values))})
        pd.DataFrame(drop_info).to_csv(os.path.join(save_path, f'dropped_features_{data_basename}.csv'), index=False)
    print(f"原始特征数: {len(feature_names)} -> 剔除 {len(drop_cols)} 个 -> 剩余 {len(keep_names)} 个")
    return X_filtered, keep_names, drop_cols


def shap_feature_selection(X, y, feature_names, save_path, data_basename, top_percent=30):
    """基于 SHAP 的特征重要性筛选，生成多种 SHAP 图"""
    print("\n=== SHAP 特征重要性分析与筛选 ===")
    model = xgb.XGBRegressor(max_depth=5, n_estimators=100, learning_rate=0.1, random_state=CONFIG["base_random_state"],
                             n_jobs=-1, verbosity=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({'Feature': feature_names, 'SHAP_importance': importance}).sort_values('SHAP_importance',
                                                                                                 ascending=False)
    imp_df.to_csv(os.path.join(save_path, f'shap_importance_all_{data_basename}.csv'), index=False)
    # 筛选 top% 特征
    n_selected = max(1, int(len(feature_names) * top_percent / 100))
    selected_names = imp_df.head(n_selected)['Feature'].tolist()
    selected_idx = [feature_names.index(f) for f in selected_names]
    X_selected = X[:, selected_idx]
    imp_df_selected = imp_df[imp_df['Feature'].isin(selected_names)]
    imp_df_selected.to_csv(os.path.join(save_path, f'shap_selected_features_{data_basename}.csv'), index=False)
    print(f"SHAP 筛选后保留 {len(selected_names)} 个特征（top {top_percent}%）")
    pd.Series(selected_names, name='Selected_Feature').to_csv(
        os.path.join(save_path, 'selected_features_after_shap.csv'), index=False)

    # 1. 绘制水平条形图（平均绝对值）
    plot_shap_importance_bar(imp_df, save_path, data_basename, top_n=20)
    # 2. 绘制 SHAP summary plot（带颜色条的经典图）
    plot_shap_summary(shap_values, X, feature_names, save_path, data_basename, max_display=20)
    # 3. 绘制累计重要性曲线
    plot_cumulative_importance(imp_df, selected_names, save_path, data_basename)

    return X_selected, selected_names, imp_df


def plot_shap_importance_bar(imp_df, save_path, data_basename, top_n=20):
    """水平条形图：平均 |SHAP| 值"""
    top_features = imp_df.head(top_n).copy()
    top_features = top_features.iloc[::-1]  # 反转使最重要的在上方
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['SHAP_importance'], color='#1f77b4')
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
    plt.xlabel('Mean |SHAP value| (impact on model output)', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Absolute)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'shap_importance_bar_absolute_{data_basename}.png'), dpi=300)
    plt.close()


def plot_shap_summary(shap_values, X, feature_names, save_path, data_basename, max_display=20):
    """绘制带颜色条的 SHAP summary plot（类似您提供的参考图）"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      max_display=max_display, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'shap_summary_plot_{data_basename}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary plot 已保存")


def plot_cumulative_importance(imp_df, selected_names, save_path, data_basename):
    """累计 SHAP 重要性曲线"""
    cumsum = np.cumsum(imp_df['SHAP_importance'].values / imp_df['SHAP_importance'].sum())
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, linewidth=2, color='#3498db')
    plt.axhline(0.8, color='red', linestyle='--', label='80%')
    plt.axhline(0.9, color='orange', linestyle='--', label='90%')
    n_selected = len(selected_names)
    if n_selected > 0:
        plt.axvline(n_selected, color='green', linestyle='--',
                    label=f'Selected {n_selected} ({cumsum[n_selected - 1] * 100:.1f}%)')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative SHAP Importance')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'shap_cumulative_importance_{data_basename}.png'), dpi=300)
    plt.close()


# ---------------------- Optuna 目标函数 ----------------------
def objective_catboost(trial, X, y, cv_folds, random_state):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'eval_metric': 'R2',
        'verbose': 0,
        'random_seed': random_state
    }
    model = CatBoostRegressor(**params)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        scores.append(r2_score(y_val, model.predict(X_val)))
    return np.mean(scores)


def objective_xgboost(trial, X, y, cv_folds, random_state):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'eval_metric': 'rmse',
        'verbosity': 0,
        'random_state': random_state,
        'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        scores.append(r2_score(y_val, model.predict(X_val)))
    return np.mean(scores)


def optimize_with_optuna(model_name, X, y, cv_folds, n_trials, random_state):
    sampler = TPESampler(seed=random_state)
    if model_name == 'CatBoost':
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda t: objective_catboost(t, X, y, cv_folds, random_state), n_trials=n_trials,
                       show_progress_bar=True)
    else:
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(lambda t: objective_xgboost(t, X, y, cv_folds, random_state), n_trials=n_trials,
                       show_progress_bar=True)
    print(f"{model_name} Optuna best params: {study.best_params}, best R²: {study.best_value:.4f}")
    return study.best_params


# ---------------------- 多模型 Predicted vs Actual 散点图（自动适应模型数量） ----------------------
def plot_pred_vs_actual_all_models(pred_dict, true_dict, save_path, data_basename):
    """绘制所有模型的预测 vs 真实散点图，自动适应子图布局"""
    n_models = len(pred_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    for idx, (model_name, preds) in enumerate(pred_dict.items()):
        trues = true_dict[model_name]
        ax = axes[idx]
        ax.scatter(trues, preds, alpha=0.5, s=10)
        min_val = min(min(trues), min(preds))
        max_val = max(max(trues), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        r2 = r2_score(trues, preds)
        ax.set_title(f'{model_name}\nR² = {r2:.3f}', fontsize=10)
        ax.set_xlabel('Actual log(σ)', fontsize=8)
        ax.set_ylabel('Predicted log(σ)', fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(len(pred_dict), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'pred_vs_actual_all_models_{data_basename}.png'), dpi=300)
    plt.close()
    print("✓ 多模型 Predicted vs Actual 散点图已保存")


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 1. 加载数据
    data_file = 'material_features_topo_ionic_conductivity_database_named_onedata.csv'
    data_basename = os.path.splitext(os.path.basename(data_file))[0]
    df = pd.read_csv(data_file)
    feature_names = df.drop(columns=['Ionic Conductivity (Log)', 'ID']).columns.tolist()
    X_original = df.drop(columns=['Ionic Conductivity (Log)', 'ID']).values
    y = df['Ionic Conductivity (Log)'].values

    save_dir = f"results_{data_basename}"
    model_save_dir = os.path.join(save_dir, "best_model")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # 2. 相关性剔除
    X_corr, corr_names, dropped_feats = correlation_analysis_and_save(X_original, feature_names, save_dir,
                                                                      data_basename)
    pd.Series(corr_names, name='Remaining_After_Correlation').to_csv(
        os.path.join(save_dir, 'remaining_features_after_correlation.csv'), index=False)

    # 3. 标准化 + SHAP 特征筛选
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_corr)
    X_selected, selected_names, shap_imp_df = shap_feature_selection(X_scaled, y, corr_names, save_dir, data_basename,
                                                                     top_percent=30)

    # 4. 定义 6 个模型（去掉 KNN）
    all_models = [
        ('Decision Tree', DecisionTreeRegressor(random_state=CONFIG["base_random_state"])),
        ('Random Forest', RandomForestRegressor(random_state=CONFIG["base_random_state"])),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=CONFIG["base_random_state"])),
        ('Support Vector Machine', SVR()),
        ('XGBoost', xgb.XGBRegressor(random_state=CONFIG["base_random_state"], eval_metric='rmse', verbosity=0)),
        ('CatBoost', CatBoostRegressor(verbose=0, random_state=CONFIG["base_random_state"]))
    ]

    tuned_models = []
    hyperparam_records = []

    print("\n=== 超参数调优阶段 ===")
    for name, base_model in all_models:
        if name in FINE_TUNE_MODELS:
            print(f"\n>>> 使用 Optuna 对 {name} 进行精细调优...")
            best_params = optimize_with_optuna(name, X_selected, y, OPTUNA_CONFIG["cv_folds"],
                                               OPTUNA_CONFIG["n_trials"], CONFIG["base_random_state"])
            if name == 'CatBoost':
                best_model = CatBoostRegressor(**best_params, verbose=0, random_state=CONFIG["base_random_state"])
            else:
                best_model = xgb.XGBRegressor(**best_params, eval_metric='rmse', verbosity=0,
                                              random_state=CONFIG["base_random_state"])
            hyperparam_records.append({'Model': name, 'Tuning_Method': 'Optuna', 'Hyperparameters': str(best_params)})
        else:
            print(f"\n>>> 使用默认参数构建 {name}")
            default_params = DEFAULT_PARAMS.get(name, {})
            best_model = base_model.set_params(**default_params)
            hyperparam_records.append({'Model': name, 'Tuning_Method': 'Fixed', 'Hyperparameters': str(default_params)})
        tuned_models.append((name, best_model))

    pd.DataFrame(hyperparam_records).to_csv(os.path.join(save_dir, 'all_models_hyperparameters.csv'), index=False)

    # 5. 交叉验证评估 + 收集预测值 + 绘制 loss 曲线（在第一个 fold 中）
    results = []
    all_preds = {}
    all_trues = {}
    trained_final_models = []

    print("\n=== 模型交叉验证评估 ===")
    for name, best_model in tuned_models:
        print(f"\n--- {name} ---")
        # 在全量数据上训练最终模型（保存用）
        final_model, _, _ = train_model_with_loss_tracking(best_model, name, X_selected, y)
        trained_final_models.append((name, final_model))

        cv_mse_scores, cv_mae_scores, cv_r2_scores, cv_mape_scores = [], [], [], []
        pred_list, true_list = [], []
        loss_plotted = False

        for repeat in range(CONFIG["kfold_repeats"]):
            current_seed = CONFIG["base_random_state"] + repeat
            kf = KFold(n_splits=CONFIG["kfold_splits"], shuffle=True, random_state=current_seed)
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
                X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                temp = clone(best_model)
                if hasattr(temp, 'random_state'):
                    temp.random_state = current_seed + fold_idx

                # 对支持迭代的模型（CatBoost, XGBoost）在第一个重复的第一折中记录 loss
                if name in ['CatBoost', 'XGBoost'] and not loss_plotted and repeat == 0 and fold_idx == 1:
                    temp, train_losses, test_losses = train_model_with_loss_tracking(temp, name, X_tr, y_tr, X_te, y_te)
                    if CONFIG["plot_loss_curves"]:
                        plot_loss_curves(train_losses, test_losses, name, save_dir, data_basename, fold=f"rep1_fold1")
                    loss_plotted = True
                else:
                    # 正常训练（不记录 loss 曲线）
                    if name == 'CatBoost':
                        temp.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
                    elif name == 'XGBoost':
                        temp.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                    else:
                        temp.fit(X_tr, y_tr)

                pred = temp.predict(X_te)
                cv_mse_scores.append(mean_squared_error(y_te, pred))
                cv_mae_scores.append(mean_absolute_error(y_te, pred))
                cv_r2_scores.append(r2_score(y_te, pred))
                cv_mape_scores.append(mean_absolute_percentage_error(y_te, pred))
                pred_list.extend(pred)
                true_list.extend(y_te)

        all_preds[name] = pred_list
        all_trues[name] = true_list

        avg_mse = np.mean(cv_mse_scores)
        avg_mae = np.mean(cv_mae_scores)
        avg_r2 = np.mean(cv_r2_scores)
        avg_mape = np.mean(cv_mape_scores)
        std_mse = np.std(cv_mse_scores)
        std_r2 = np.std(cv_r2_scores)
        results.append([name, avg_mse, avg_mae, avg_r2, avg_mape, std_mse, std_r2])
        print(f"  CV 结果: MSE={avg_mse:.4f}±{std_mse:.4f}, R²={avg_r2:.4f}±{std_r2:.4f}")

        # 保存每折详细结果
        cv_details = pd.DataFrame({
            'Repeat': np.repeat(range(1, CONFIG['kfold_repeats'] + 1), CONFIG['kfold_splits']),
            'Fold': list(range(1, CONFIG['kfold_splits'] + 1)) * CONFIG['kfold_repeats'],
            'Test_MSE': cv_mse_scores, 'Test_MAE': cv_mae_scores,
            'Test_R2': cv_r2_scores, 'Test_MAPE': cv_mape_scores
        })
        cv_details.to_csv(os.path.join(save_dir, f'cv_details_{name.lower().replace(" ", "_")}_{data_basename}.csv'),
                          index=False)

    # 6. 多模型散点图
    plot_pred_vs_actual_all_models(all_preds, all_trues, save_dir, data_basename)

    # 7. 汇总结果并选最优
    results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'R2', 'MAPE', 'MSE_Std', 'R2_Std'])
    results_df_sorted = results_df.sort_values('R2', ascending=False)
    best_name = results_df_sorted.iloc[0]['Model']
    best_metrics = results_df_sorted.iloc[0]

    results_df.to_csv(os.path.join(save_dir, f'model_results_{data_basename}.csv'), index=False)

    # 8. 保存最优模型及附属文件
    best_model_obj = [m for n, m in trained_final_models if n == best_name][0]
    joblib.dump(best_model_obj,
                os.path.join(model_save_dir, f'best_model_{best_name.lower().replace(" ", "_")}.joblib'))
    joblib.dump(scaler, os.path.join(model_save_dir, 'scaler.joblib'))
    feature_info = {
        'original_feature_names': feature_names,
        'dropped_features': dropped_feats,
        'corr_remaining_names': corr_names,
        'selected_feature_names': selected_names,
        'shap_threshold': CONFIG["shap_feature_threshold"],
        'corr_threshold': CONFIG["corr_threshold"]
    }
    joblib.dump(feature_info, os.path.join(model_save_dir, 'feature_info.joblib'))
    shap_imp_df.to_csv(os.path.join(model_save_dir, 'shap_importance.csv'), index=False)
    best_metrics.to_frame().T.to_csv(os.path.join(model_save_dir, 'best_model_metrics.csv'), index=False)

    print(f"\n所有结果已保存至：{save_dir}")
    print(f"最优模型：{best_name} (平均 R² = {best_metrics['R2']:.4f})")