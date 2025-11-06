import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from typing import Dict, Any, Callable
from datetime import datetime

# --- 添削・改善した最適化関数 ---


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

        # 混合行列を整形して表示
        print(
            f"\n--- Confusion Matrix (Best Trial at Threshold: {best_threshold:.4f}) ---")
        print("--------------------------|Predicted Label              |")
        print("--------------------------|-----------------------------|")
        print("--------------------------| Negative (0) | Positive (1) |")
        print("-----------|--------------|--------------|--------------|")
        # cm[0, 0] = TN, cm[0, 1] = FP
        print(
            f"True Label | Negative (0) | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN, FP)")
        print("           |--------------|--------------|--------------|")
        # cm[1, 0] = FN, cm[1, 1] = TP
        print(
            f"           | Positive (1) | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN, TP)")
        print("-----------|--------------|--------------|--------------|")

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
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X_train_df, y_train_df)):
        print(f"Fold {fold+1}/{folds.n_splits} started...")
        X_train_fold, y_train_fold = X_train_df.iloc[train_idx], y_train_df.iloc[train_idx]
        X_valid_fold, y_valid_fold = X_train_df.iloc[valid_idx], y_train_df.iloc[valid_idx]

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

    # アンサンブル予測
    oof_preds_ensemble = (oof_preds_lgb + oof_preds_xgb + oof_preds_cat) / 3
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
    print(
        f"Ensemble Best F1: {best_f1_score:.5f} (Threshold: {best_threshold:.2f})")
    print(f"LightGBM: {best_f1_lgb:.5f} (Threshold: {best_threshold_lgb:.2f})")
    print(f"XGBoost:  {best_f1_xgb:.5f} (Threshold: {best_threshold_xgb:.2f})")
    print(f"CatBoost: {best_f1_cat:.5f} (Threshold: {best_threshold_cat:.2f})")

    # 提出ファイル生成
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    submission_files = []

    # アンサンブル
    test_preds_ensemble = (
        test_preds_lgb + test_preds_xgb + test_preds_cat) / 3
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
        'submission_files': submission_files
    }
