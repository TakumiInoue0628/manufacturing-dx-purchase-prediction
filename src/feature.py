import MeCab
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


# --- MeCabトークナイザー ---
def tokenize_japanese(text):
    mecab = MeCab.Tagger()
    node = mecab.parseToNode(text)
    words = []
    while node:
        features = node.feature.split(',')
        pos = features[0]
        surface = node.surface
        if surface:
            token = surface.lower() if re.match(
                r'[A-Za-z]+', surface) else surface
            # 助詞・助動詞・記号を除外 + 正規表現で記号除去 + ストップワード除外 + 1文字以下除外
            if pos not in ['助詞', '助動詞', '記号']:
                if (
                    re.match(r'^[ぁ-んァ-ン一-龥a-zA-Z0-9]+$', token)
                    and len(token) > 1
                ):
                    words.append(token)
        node = node.next
    return words


# 購入フラグに基づくポジ・ネガワード定義関数
def analyze_tfidf_diff(
    df_text, df_flag, top_N=30, figsize=(5, 10)
):
    # --- TF-IDFベクトル化 ---
    vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
    X = vectorizer.fit_transform(df_text)
    feature_names = vectorizer.get_feature_names_out()
    # --- TF-IDFデータフレーム ---
    tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)
    tfidf_df["flag"] = df_flag
    # --- グループ別平均スコア ---
    group_mean = tfidf_df.groupby("flag").mean().T
    # --- 差分スコア計算 ---
    diff_scores = group_mean[1] - group_mean[0]
    # --- TOP Nプラス・マイナス抽出 ---
    top_positive = diff_scores.sort_values(ascending=False).head(top_N)
    top_negative = diff_scores.sort_values(ascending=True).head(top_N)
    # --- 結合（プラス→マイナス） ---
    combined_scores = pd.concat([top_positive, top_negative[::-1]])
    # --- グラフ化 ---
    colors = ['C1' if v > 0 else 'C0' for v in combined_scores.values]
    combined_scores[::-1].plot(kind="barh",
                               figsize=figsize, color=colors[::-1])
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("TF-IDF差分スコア")
    plt.tight_layout()
    plt.show()
    return top_positive, top_negative


# 特徴量エンジニアリング関数
def feature_engineering(df):
    df['自己資本比率'] = df['自己資本'] / (df['総資産'] + 1e-6)
    df['売上高営業利益率'] = df['営業利益'] / (df['売上'] + 1e-6)
    df['総資産回転率'] = df['売上'] / (df['総資産'] + 1e-6)
    df['負債比率'] = df['負債'] / (df['自己資本'] + 1e-6)
    df['規模収益性'] = df['従業員数'] * df['売上高営業利益率']
    return df


# カテゴリ変数のOne-Hotエンコード関数
def onehot_encode_categorical(df, categorical_cols, sep=', '):
    for col in categorical_cols:
        df_encode = df[col].str.get_dummies(sep=sep).add_prefix(col+'_')
        df = pd.concat([df, df_encode], axis=1)
    df = df.drop(columns=categorical_cols)
    return df


# バイナリ（２択）変数のエンコード関数
def encode_binary(df, binary_cols, true=1):
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == true else 0)
    return df


# 組織図を特徴量化する関数
def org_chart_features(df, keyword_features={
    'dx_it': r'DX|デジタル|IT|情報システム',
    'planning': r'経営企画|経営戦略|事業企画',
    'quality': r'品質管理|品質保証|プロセス改善|業務改革',
    'rd': r'R&D|研究開発|新技術',
    'risk': r'リスク管理|内部監査|コンプライアンス',
    'mfg': r'製造|工場|生産技術'
}):
    df = df.fillna('')
    org_df = pd.DataFrame()
    # 組織の規模（行数）
    org_df[df.name + '_line_count'] = df.apply(lambda x: len(x.split('\n')))
    # 組織単位数を（├, └, ┌の数）
    org_df[df.name + '_unit_count'] = df.str.count(r'[├└┌]')
    # 特定部門の有無
    for col_name, pattern in keyword_features.items():
        org_df[df.name + "_has_" +
               col_name] = df.str.contains(pattern, case=False, na=False).astype(int)
    return org_df


# テキストデータの特徴量化関数（文章量）
def text_length_features(df):
    df = df.fillna('')
    length = [len(text) for text in df]
    length_df = pd.DataFrame(length, index=df.index,
                             columns=[df.name + "_length"])
    return length_df


# テキストデータの特徴量化関数（ポジネガ出現率）
def pos_neg_ratio(df, positive_words, negative_words):
    df = df.fillna('')
    pos_neg_features = []
    for text in df:
        pos_count = sum(text.count(word) for word in positive_words)
        neg_count = sum(text.count(word) for word in negative_words)
        total_count = len(text.split())
        pos_ratio = pos_count / total_count if total_count > 0 else 0
        neg_ratio = neg_count / total_count if total_count > 0 else 0
        pos_neg_features.append([pos_ratio, neg_ratio])
    pos_neg_df = pd.DataFrame(pos_neg_features, index=df.index, columns=[
                              df.name + "_pos_ratio", df.name + "_neg_ratio"])
    return pos_neg_df
