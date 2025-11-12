import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from typing import Dict, Any, Callable
from datetime import datetime
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def optimize_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    params_base: Dict[str, Any],
    define_params_func: Callable[[optuna.trial.Trial], Dict[str, Any]],
    n_folds: int = 5,
    n_trials: int = 50,
    early_stopping_rounds: int = 50,
    thresholds: np.ndarray = np.arange(0.1, 0.5, 0.01),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    指定されたモデルのハイパーパラメータをOptunaで最適化する汎用関数。

    Args:
        X_train (pd.DataFrame): 学習データの特徴量
        y_train (pd.Series): 学習データの目的変数
        model_name (str): モデル名 ('lightgbm', 'xgboost', 'catboost'のいずれか)
        params_base (Dict[str, Any]): 全ての試行で固定する基本パラメータ
        define_params_func (Callable): Optunaのtrialオブジェクトを引数とし、
                                     探索するパラメータ範囲を定義して辞書を返す関数
        n_folds (int): 交差検証の分割数
        n_trials (int): Optunaの試行回数
        early_stopping_rounds (int):早期終了のラウンド数
        thresholds (np.ndarray): F1スコアを計算するための閾値の範囲
        random_state (int): 乱数シード

    Returns:
        Dict[str, Any]: 最適化結果を含む辞書
                        {'best_params': dict, 'best_score': float, 'study': optuna.study.Study}
    """
    print(f"--- Optimizing {model_name} ---")

    folds = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial) -> float:
        # trialごとに探索するパラメータを動的に生成する
        if model_name == "xgboost":
            params_base['early_stopping_rounds'] = early_stopping_rounds
        params_opt = define_params_func(trial)
        params = params_base | params_opt

        oof_preds = np.zeros(X_train.shape[0])

        for _, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_valid_fold, y_valid_fold = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

            if model_name == "lightgbm":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_valid_fold, y_valid_fold)],
                          callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            elif model_name == "xgboost":
                model = xgb.XGBClassifier(**params)
                model.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_valid_fold, y_valid_fold)],
                          verbose=False)
            elif model_name == "catboost":
                model = CatBoostClassifier(**params)
                model.fit(X_train_fold, y_train_fold,
                          eval_set=[(X_valid_fold, y_valid_fold)],
                          use_best_model=True,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose=False)
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]

        # 各foldでF1スコアを計算し、平均を返す
        f1_scores_per_threshold = []
        for t in thresholds:
            fold_f1_scores = []
            for _, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
                y_valid_fold = y_train.iloc[valid_idx]
                y_pred_fold = (oof_preds[valid_idx] > t).astype(int)
                fold_f1_scores.append(f1_score(y_valid_fold, y_pred_fold))
            f1_scores_per_threshold.append(np.mean(fold_f1_scores))
        best_f1_index = np.argmax(f1_scores_per_threshold)
        best_f1_score = f1_scores_per_threshold[best_f1_index]
        best_threshold = thresholds[best_f1_index]

        # OptunaのtrialにOOF予測値とベスト閾値を保存
        # (JSONシリアライズ可能な形式に変換)
        trial.set_user_attr("oof_preds", oof_preds.tolist())
        trial.set_user_attr("best_threshold", best_threshold)

        return best_f1_score  # 最大のF1スコアを返す

    # Optunaによる最適化実行
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Optimization Finished ---")
    print(f"Best trial for {model_name}:")
    best_trial = study.best_trial
    print(f"  Value (Best F1 Score): {best_trial.value:.5f}")
    print("  Best Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # ベストトライアルのOOF予測値と閾値を取得
    try:
        # user_attrsから値を取得
        oof_preds_best = np.array(best_trial.user_attrs['oof_preds'])
        best_threshold = best_trial.user_attrs['best_threshold']

        # 最適な閾値で予測値を0/1に変換
        y_pred_best = (oof_preds_best > best_threshold).astype(int)

        # 混合行列を計算
        cm = confusion_matrix(y_train, y_pred_best)

        # 精度、再現率、F1スコアを計算
        precision = precision_score(y_train, y_pred_best)
        recall = recall_score(y_train, y_pred_best)
        f1_oof = f1_score(y_train, y_pred_best)

        # 混合行列を整形して表示
        print(
            f"\n--- Confusion Matrix (Best Trial at Threshold: {best_threshold:.4f}) ---")
        print("                          |Predicted Label              |")
        print("                          |-----------------------------|")
        print("                          | Negative (0) | Positive (1) |")
        print("-----------|--------------|--------------|--------------|")
        # cm[0, 0] = TN, cm[0, 1] = FP
        print(
            f"True Label | Negative (0) | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN, FP)")
        print("           |--------------|--------------|--------------|")
        # cm[1, 0] = FN, cm[1, 1] = TP
        print(
            f"           | Positive (1) | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN, TP)")
        print("-----------|--------------|--------------|--------------|")
        print("\n--- OOF Scores (at Best Threshold) ---")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1 Score:  {f1_oof:.5f} (F1 score on total OOF predictions)")
        print("  (Note: 'Best F1 Score' above is the CV mean optimized by Optuna.)")

    except KeyError:
        print("\n--- Confusion Matrix ---")
        print("Could not retrieve 'oof_preds' or 'best_threshold' from user_attrs.")
    except Exception as e:
        print(f"\nError displaying confusion matrix: {e}")

    # 最適パラメータと結果を返す
    best_params = params_base | best_trial.params

    # ベストパラメータでモデルを全データで再学習し、特徴量重要度を表示
    print("\n--- Feature Importance (Best Model) ---")
    final_params = best_params.copy()
    if model_name == 'xgboost' and 'early_stopping_rounds' in final_params:
        final_params.pop('early_stopping_rounds')  # fit時にeval_setがないため削除
    if model_name == "lightgbm":
        model_final = lgb.LGBMClassifier(**final_params)
        model_final.fit(X_train, y_train)
    elif model_name == "xgboost":
        model_final = xgb.XGBClassifier(**final_params)
        model_final.fit(X_train, y_train)
    elif model_name == "catboost":
        model_final = CatBoostClassifier(**final_params)
        model_final.fit(X_train, y_train, verbose=0)
    # 特徴量重要度をDataFrameに格納
    df_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model_final.feature_importances_
    })
    # 重要度の高い順にソートして表示
    df_importance = df_importance.sort_values(
        'Importance', ascending=False).reset_index(drop=True)
    print(df_importance.head(20))

    return {
        'best_params': best_params,
        'best_score': best_trial.value,
        'study': study,
        'feature_importance': df_importance
    }


def train_ensemble_models(
    X_train_df: pd.DataFrame,
    y_train_df: pd.Series,
    X_test_df: pd.DataFrame,
    lgb_best_params: dict,
    xgb_best_params: dict,
    cat_best_params: dict,
    sample_submit: pd.DataFrame,
    n_folds: int = 5,
    early_stopping_rounds: int = 50,
    thresholds: np.ndarray = np.arange(0.1, 0.5, 0.01),
    random_state: int = 42
) -> dict:
    """
    LightGBM, XGBoost, CatBoostの最適化済みパラメータを用いてアンサンブル学習を行う関数。
    OOF予測、テスト予測、F1スコア評価、提出ファイル生成を実施。

    Returns:
        dict: {
            'ensemble_f1': float,
            'ensemble_threshold': float,
            'individual_scores': dict,
            'submission_files': list
        }
    """
    folds = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state)
    n_trains = X_train_df.shape[0]
    n_tests = X_test_df.shape[0]

    # OOF予測値とテスト予測値の初期化
    oof_preds_lgb = np.zeros(n_trains)
    oof_preds_xgb = np.zeros(n_trains)
    oof_preds_cat = np.zeros(n_trains)
    test_preds_lgb = np.zeros(n_tests)
    test_preds_xgb = np.zeros(n_tests)
    test_preds_cat = np.zeros(n_tests)

    print("--- Start Ensemble Training ---")
    all_valid_indices = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X_train_df, y_train_df)):
        print(f"Fold {fold+1}/{folds.n_splits} started...")
        X_train_fold, y_train_fold = X_train_df.iloc[train_idx], y_train_df.iloc[train_idx]
        X_valid_fold, y_valid_fold = X_train_df.iloc[valid_idx], y_train_df.iloc[valid_idx]
        all_valid_indices.append(valid_idx)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_best_params)
        lgb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(
                early_stopping_rounds, verbose=False)]
        )
        oof_preds_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(
            X_test_df)[:, 1] / folds.n_splits

        # XGBoost
        xgb_best_params['early_stopping_rounds'] = early_stopping_rounds
        xgb_model = xgb.XGBClassifier(**xgb_best_params)
        xgb_model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_valid_fold, y_valid_fold)],
                      verbose=False)
        oof_preds_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(
            X_test_df)[:, 1] / folds.n_splits

        # CatBoost
        cat_model = CatBoostClassifier(**cat_best_params)
        cat_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        oof_preds_cat[valid_idx] = cat_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_cat += cat_model.predict_proba(
            X_test_df)[:, 1] / folds.n_splits

    print("--- Ensemble Training Finished ---")

    # アンサンブル予測(ブレンディング)
    oof_preds = np.stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat], axis=1)
    def objective_function(weights):
        # 重み付き平均を計算
        weighted_oof_preds = np.dot(oof_preds, weights)
        # 最適な閾値を見つける
        f1_scores_list = [
            f1_score(y_train_df, (weighted_oof_preds > t).astype(int)) for t in thresholds
        ]
        best_f1 = np.max(f1_scores_list)
        # 最小化するため、負のF1スコアを返す
        return -best_f1
    # 制約条件: 重みの合計が1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # 境界条件: 各重みは0から1の間
    bounds = [(0, 1) for _ in range(3)]
    # 初期値: 均等な重み
    initial_weights = np.array([1/3, 1/3, 1/3])
    # 最適化の実行
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP', # 制約付き最適化のための標準的な手法
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        print("Warning: Weight optimization failed. Using equal weights.")
        best_weights = initial_weights
    else:
        best_weights = result.x
    print(f"Optimal Weights - LGB: {best_weights[0]:.4f}, XGB: {best_weights[1]:.4f}, CAT: {best_weights[2]:.4f}")
    # 重み付き平均でアンサンブル予測
    oof_preds_ensemble = (
        best_weights[0] * oof_preds_lgb +
        best_weights[1] * oof_preds_xgb +
        best_weights[2] * oof_preds_cat
    )

    # 最適な閾値の探索とF1スコア計算
    f1_scores = [
        f1_score(y_train_df, (oof_preds_ensemble > t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)

    # 各モデル単体のスコア
    f1_scores_lgb = [
        f1_score(y_train_df, (oof_preds_lgb > t).astype(int)) for t in thresholds]
    best_threshold_lgb = thresholds[np.argmax(f1_scores_lgb)]
    best_f1_lgb = np.max(f1_scores_lgb)
    f1_scores_xgb = [
        f1_score(y_train_df, (oof_preds_xgb > t).astype(int)) for t in thresholds]
    best_threshold_xgb = thresholds[np.argmax(f1_scores_xgb)]
    best_f1_xgb = np.max(f1_scores_xgb)
    f1_scores_cat = [
        f1_score(y_train_df, (oof_preds_cat > t).astype(int)) for t in thresholds]
    best_threshold_cat = thresholds[np.argmax(f1_scores_cat)]
    best_f1_cat = np.max(f1_scores_cat)

    print("\n--- Evaluation ---")
    print(f"Ensemble Best F1: {best_f1_score:.5f} (Threshold: {best_threshold:.2f})")
    print(f"LightGBM: {best_f1_lgb:.5f} (Threshold: {best_threshold_lgb:.2f})")
    print(f"XGBoost:  {best_f1_xgb:.5f} (Threshold: {best_threshold_xgb:.2f})")
    print(f"CatBoost: {best_f1_cat:.5f} (Threshold: {best_threshold_cat:.2f})")

    # Foldごとの安定性評価
    print("\n--- Fold-wise Stability Evaluation (using global thresholds) ---")
    fold_results = []
    for i, valid_idx in enumerate(all_valid_indices):
        y_valid_fold = y_train_df.iloc[valid_idx]
        # このFoldのOOF予測値
        oof_lgb_fold = oof_preds_lgb[valid_idx]
        oof_xgb_fold = oof_preds_xgb[valid_idx]
        oof_cat_fold = oof_preds_cat[valid_idx]
        # グローバルな重みでアンサンブル
        oof_ensemble_fold = (
            best_weights[0] * oof_lgb_fold +
            best_weights[1] * oof_xgb_fold +
            best_weights[2] * oof_cat_fold
        )
        # グローバルな閾値を使ってF1を計算
        f1_ensemble_fold = f1_score(y_valid_fold, (oof_ensemble_fold > best_threshold).astype(int))
        f1_lgb_fold = f1_score(y_valid_fold, (oof_lgb_fold > best_threshold_lgb).astype(int))
        f1_xgb_fold = f1_score(y_valid_fold, (oof_xgb_fold > best_threshold_xgb).astype(int))
        f1_cat_fold = f1_score(y_valid_fold, (oof_cat_fold > best_threshold_cat).astype(int))
        print(f"  Fold {i+1} F1 -> Ensemble: {f1_ensemble_fold:.5f} | LGB: {f1_lgb_fold:.5f} | XGB: {f1_xgb_fold:.5f} | CAT: {f1_cat_fold:.5f}")
        fold_results.append({
            'fold': i + 1,
            'ensemble_f1': f1_ensemble_fold,
            'lgb_f1': f1_lgb_fold,
            'xgb_f1': f1_xgb_fold,
            'cat_f1': f1_cat_fold
        })
    # 平均と標準偏差を計算
    all_fold_f1s = [r['ensemble_f1'] for r in fold_results]
    mean_f1 = np.mean(all_fold_f1s)
    std_f1 = np.std(all_fold_f1s)
    print(f"Ensemble Fold F1 Mean: {mean_f1:.5f}, Std: {std_f1:.5f}")

    # 提出ファイル生成
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    submission_files = []

    # 重み付き平均でテストデータの予測値を計算
    test_preds_ensemble = (
        best_weights[0] * test_preds_lgb +
        best_weights[1] * test_preds_xgb +
        best_weights[2] * test_preds_cat
    )
    final_predictions = (test_preds_ensemble > best_threshold).astype(int)
    submit_df_ensemble = sample_submit.copy()
    submit_df_ensemble[1] = final_predictions
    ensemble_file = f'submission_ensemble_{timestamp}.csv'
    submit_df_ensemble.to_csv(ensemble_file, index=False, header=False)
    submission_files.append(ensemble_file)

    # 単体モデル（アンサンブルより良ければ）
    if best_f1_lgb > best_f1_score:
        final_predictions_lgb = (
            test_preds_lgb > best_threshold_lgb).astype(int)
        submit_df_lgb = sample_submit.copy()
        submit_df_lgb[1] = final_predictions_lgb
        lgb_file = f'submission_lgb_{timestamp}.csv'
        submit_df_lgb.to_csv(lgb_file, index=False, header=False)
        submission_files.append(lgb_file)

    if best_f1_xgb > best_f1_score:
        final_predictions_xgb = (
            test_preds_xgb > best_threshold_xgb).astype(int)
        submit_df_xgb = sample_submit.copy()
        submit_df_xgb[1] = final_predictions_xgb
        xgb_file = f'submission_xgb_{timestamp}.csv'
        submit_df_xgb.to_csv(xgb_file, index=False, header=False)
        submission_files.append(xgb_file)

    if best_f1_cat > best_f1_score:
        final_predictions_cat = (
            test_preds_cat > best_threshold_cat).astype(int)
        submit_df_cat = sample_submit.copy()
        submit_df_cat[1] = final_predictions_cat
        cat_file = f'submission_cat_{timestamp}.csv'
        submit_df_cat.to_csv(cat_file, index=False, header=False)
        submission_files.append(cat_file)

    return {
        'ensemble_f1': best_f1_score,
        'ensemble_threshold': best_threshold,
        'individual_scores': {
            'LightGBM': (best_f1_lgb, best_threshold_lgb),
            'XGBoost': (best_f1_xgb, best_threshold_xgb),
            'CatBoost': (best_f1_cat, best_threshold_cat)
        },
        'submission_files': submission_files,
        'optimal_weights': best_weights.tolist(),
        'fold_results': fold_results,
        'fold_f1_mean_std': (mean_f1, std_f1)
    }


# 改良版アンサンブル学習関数（ブレンディング＋スタッキング）
def train_ensemble_models_(
    X_train_df: pd.DataFrame,
    y_train_df: pd.Series,
    X_test_df: pd.DataFrame,
    lgb_best_params: dict,
    xgb_best_params: dict,
    cat_best_params: dict,
    sample_submit: pd.DataFrame,
    n_folds: int = 5,
    early_stopping_rounds: int = 50,
    thresholds: np.ndarray = np.arange(0.1, 0.5, 0.01),
    random_state: int = 42
) -> dict:
    """
    LightGBM, XGBoost, CatBoostの最適化済みパラメータを用いてアンサンブル学習（Blending & Stacking）を行う関数。
    """
    
    # --- 1. Base Models Training (Level 1) ---
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n_trains = X_train_df.shape[0]
    n_tests = X_test_df.shape[0]

    # OOF予測値とテスト予測値の初期化
    oof_preds_lgb = np.zeros(n_trains)
    oof_preds_xgb = np.zeros(n_trains)
    oof_preds_cat = np.zeros(n_trains)
    
    test_preds_lgb = np.zeros(n_tests)
    test_preds_xgb = np.zeros(n_tests)
    test_preds_cat = np.zeros(n_tests)

    print("--- Start Base Models Training ---")
    all_valid_indices = []
    
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X_train_df, y_train_df)):
        print(f"Fold {fold+1}/{folds.n_splits} started...")
        X_train_fold, y_train_fold = X_train_df.iloc[train_idx], y_train_df.iloc[train_idx]
        X_valid_fold, y_valid_fold = X_train_df.iloc[valid_idx], y_train_df.iloc[valid_idx]
        all_valid_indices.append(valid_idx)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_best_params)
        lgb_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        oof_preds_lgb[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_lgb += lgb_model.predict_proba(X_test_df)[:, 1] / folds.n_splits

        # XGBoost
        xgb_cp_params = xgb_best_params.copy()
        xgb_cp_params['early_stopping_rounds'] = early_stopping_rounds
        xgb_model = xgb.XGBClassifier(**xgb_cp_params)
        xgb_model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_valid_fold, y_valid_fold)],
                      verbose=False)
        oof_preds_xgb[valid_idx] = xgb_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_xgb += xgb_model.predict_proba(X_test_df)[:, 1] / folds.n_splits

        # CatBoost
        cat_model = CatBoostClassifier(**cat_best_params)
        cat_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        oof_preds_cat[valid_idx] = cat_model.predict_proba(X_valid_fold)[:, 1]
        test_preds_cat += cat_model.predict_proba(X_test_df)[:, 1] / folds.n_splits

    print("--- Base Models Training Finished ---")

    # メタ特徴量の作成（Level 1の予測値を結合）
    train_meta = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat])
    test_meta = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])

    # --- 2. Blending Implementation (Optimization) ---
    print("\n--- Start Blending Optimization ---")
    def objective_function(weights):
        weighted_oof_preds = np.dot(train_meta, weights)
        f1_scores_list = [
            f1_score(y_train_df, (weighted_oof_preds > t).astype(int)) for t in thresholds
        ]
        return -np.max(f1_scores_list)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(3)]
    initial_weights = np.array([1/3, 1/3, 1/3])
    
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    best_weights = result.x if result.success else initial_weights
    print(f"Blending Weights - LGB: {best_weights[0]:.4f}, XGB: {best_weights[1]:.4f}, CAT: {best_weights[2]:.4f}")
    
    oof_preds_blending = np.dot(train_meta, best_weights)
    test_preds_blending = np.dot(test_meta, best_weights)

    # --- 3. Stacking Implementation (Meta Model) ---
    print("\n--- Start Stacking (Logistic Regression) ---")
    # メタモデルにはロジスティック回帰を使用（過学習しにくいため）
    meta_model = LogisticRegression(random_state=random_state)
    meta_model.fit(train_meta, y_train_df)
    
    oof_preds_stacking = meta_model.predict_proba(train_meta)[:, 1]
    test_preds_stacking = meta_model.predict_proba(test_meta)[:, 1]
    
    # スタッキングモデルの係数を確認（どのモデルを重要視したか）
    print(f"Stacking Coefs - LGB: {meta_model.coef_[0][0]:.4f}, XGB: {meta_model.coef_[0][1]:.4f}, CAT: {meta_model.coef_[0][2]:.4f}")

    # --- 4. Threshold Optimization & Scoring ---
    def get_best_metrics(y_true, y_pred_proba, thresholds):
        scores = [f1_score(y_true, (y_pred_proba > t).astype(int)) for t in thresholds]
        best_idx = np.argmax(scores)
        return np.max(scores), thresholds[best_idx]

    # 各モデルのスコア算出
    best_f1_blend, best_th_blend = get_best_metrics(y_train_df, oof_preds_blending, thresholds)
    best_f1_stack, best_th_stack = get_best_metrics(y_train_df, oof_preds_stacking, thresholds)
    best_f1_lgb, best_th_lgb = get_best_metrics(y_train_df, oof_preds_lgb, thresholds)
    best_f1_xgb, best_th_xgb = get_best_metrics(y_train_df, oof_preds_xgb, thresholds)
    best_f1_cat, best_th_cat = get_best_metrics(y_train_df, oof_preds_cat, thresholds)

    print("\n--- Evaluation Results ---")
    print(f"Blending F1: {best_f1_blend:.5f} (Th: {best_th_blend:.2f})")
    print(f"Stacking F1: {best_f1_stack:.5f} (Th: {best_th_stack:.2f})")
    print(f"LightGBM F1: {best_f1_lgb:.5f}")
    print(f"XGBoost  F1: {best_f1_xgb:.5f}")
    print(f"CatBoost F1: {best_f1_cat:.5f}")

    # --- 5. Submission Files Generation ---
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    submission_files = []

    def save_submission(preds, threshold, suffix):
        final_preds = (preds > threshold).astype(int)
        submit_df = sample_submit.copy()
        submit_df[1] = final_preds
        filename = f'submission_{suffix}_{timestamp}.csv'
        submit_df.to_csv(filename, index=False, header=False)
        submission_files.append(filename)

    # Blending Submission
    save_submission(test_preds_blending, best_th_blend, "blending")
    
    # Stacking Submission
    save_submission(test_preds_stacking, best_th_stack, "stacking")

    # 単体モデル（Blendingより良ければ出力）
    global_best_f1 = max(best_f1_blend, best_f1_stack)
    if best_f1_lgb > global_best_f1: save_submission(test_preds_lgb, best_th_lgb, "lgb")
    if best_f1_xgb > global_best_f1: save_submission(test_preds_xgb, best_th_xgb, "xgb")
    if best_f1_cat > global_best_f1: save_submission(test_preds_cat, best_th_cat, "cat")

    return {
        'scores': {
            'Blending': best_f1_blend,
            'Stacking': best_f1_stack,
            'LGB': best_f1_lgb,
            'XGB': best_f1_xgb,
            'CAT': best_f1_cat
        },
        'thresholds': {
            'Blending': best_th_blend,
            'Stacking': best_th_stack
        },
        'blending_weights': best_weights.tolist(),
        'stacking_model': meta_model,
        'submission_files': submission_files
    }
