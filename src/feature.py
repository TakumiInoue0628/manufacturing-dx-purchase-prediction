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


# 欠損値補完関数
def impute_missing_values(df, 
                          zero_fill_cols=['工場数', '店舗数'], 
                          median_fill_cols=['資本金', '短期借入金', '長期借入金', '営業利益', '経常利益']):
    for col in zero_fill_cols:
        df[col] = df[col].fillna(0)
    for col in median_fill_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    return df


# 特徴量エンジニアリング関数
def feature_engineering(df):
    epsilon = 1e-6 # ゼロ除算を避けるための微小量

    df['営業利益率'] = df['営業利益'] / (df['売上'] + epsilon)
    df['総資産回転率'] = df['売上'] / (df['総資産'] + epsilon)
    df['ROA'] = df['当期純利益'] / (df['総資産'] + epsilon)
    df['従業員一人当たり営業利益'] = df['営業利益'] / (df['従業員数'] + epsilon)

    df['流動比率'] = df['流動資産'] / (df['負債'] + epsilon)
    df['無形固定資産変動率'] = df['無形固定資産変動(ソフトウェア関連)'] / (df['総資産'] + epsilon)

    df['営業CFマージン'] = df['営業CF'] / (df['売上'] + epsilon)
    df['投資効率対CF'] = df['無形固定資産変動(ソフトウェア関連)'] / (df['営業CF'] + epsilon)

    #df['自己資本比率'] = df['自己資本'] / (df['総資産'] + epsilon)
    #df['負債比率'] = df['負債'] / (df['自己資本'] + epsilon)
    #df['規模収益性'] = df['従業員数'] * df['営業利益率']
    #df['有形固定資産変動'] = df['有形固定資産変動'] / (df['総資産'] + epsilon)
    #df['従業員一人当たり売上'] = df['売上'] / (df['従業員数'] + epsilon)
    #df['負債純資産倍率'] = df['負債'] / (df['純資産'] + epsilon)
    #df['投資効率対利益'] = df['無形固定資産変動(ソフトウェア関連)'] / (df['営業利益'] + epsilon)
    #df['新規投資度合い'] = df['無形固定資産変動(ソフトウェア関連)'] / (df['減価償却費'] + epsilon)
    drop_cols = []
    df = df.drop(columns=drop_cols)
    return df


# アンケートデータの特徴量化関数
def survey_features(df, question_cols,
                    pos_q_cols=['アンケート１', 'アンケート２', 'アンケート３', 'アンケート５', 'アンケート８', 'アンケート９', 'アンケート１１'],
                    neg_q_cols=['アンケート４']):
    # ポジネガ
    survey_pos_mean = df[pos_q_cols].mean(axis=1)
    survey_neg_mean = df[neg_q_cols].mean(axis=1)
    df["アンケート_pos_mean"] = survey_pos_mean
    df["アンケート_neg_mean"] = survey_neg_mean
    df["アンケート_pos_minus_neg"] = survey_pos_mean - survey_neg_mean
    # 全体平均
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
    'dx_it': r'DX|デジタル|IT|情報システム|システム開発|DX推進',
    #'planning': r'経営企画|経営戦略|事業企画|経営管理|戦略企画',
    'quality': r'品質管理|品質保証|プロセス改善|業務改革|試験|検査',
    'rd': r'R&D|研究開発|新技術|技術開発|製品開発|設計|開発',
    'risk': r'リスク管理|内部監査|コンプライアンス|法務|監査',
    'mfg': r'製造|工場|生産技術|生産管理|製造管理|工場運営|施工管理',
    'admin': r'管理本部|管理部門|人事|総務|財務|経理|会計|購買|バックオフィス',
    #'sales': r'営業|販売|国内営業|海外営業|国際営業|営業推進|販売促進',
    #'marketing': r'マーケティング|広告|プロモーション|市場調査|広報|PR',
    #'support': r'サポート|顧客対応|保守|サービス|カスタマーサポート|CS',
    'logistics': r'輸送|物流|海運|運輸|鉄道|バス|貨物|ロジスティクス|サプライチェーン|SCM'
}):
    df = df.fillna('')
    org_df = pd.DataFrame()
    # 組織の規模（行数）
    #org_df[df.name + '_line_count'] = df.apply(lambda x: len(x.split('\n')))
    # 組織単位数を（├, └, ┌の数）
    #org_df[df.name + '_unit_count'] = df.str.count(r'[├└┌]')
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


# テキストデータの特徴量化関数（TF-IDF）
def tfidf_vectorization(df, max_features=100, ngram_range=(1, 2)):
    df = df.fillna('')
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(df).toarray()
    df_idfs = pd.DataFrame(X_tfidf, index=df.index)
    df_idfs.columns = df.name + "_" + vectorizer.get_feature_names_out()
    return df_idfs


# テキストデータの埋め込み関数（Transformer）
def transformer_embedding(df, model_name='intfloat/multilingual-e5-base', normalize_embeddings=False):
    from sentence_transformers import SentenceTransformer
    df = df.fillna('')
    model = SentenceTransformer(
        model_name_or_path=model_name, 
        device='cpu',
    )
    embeddings = model.encode(
        sentences=df.tolist(), 
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    df_embeddings = pd.DataFrame(embeddings, index=df.index)
    df_embeddings.columns = df.name + "_emb_" + df_embeddings.columns.astype(str)
    return df_embeddings


# 次元削減関数（PCA）
def pca_reduction(df, n_components=10, df_name="data", random_state=42):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_data = pca.fit_transform(df)
    df_reduced = pd.DataFrame(reduced_data, index=df.index)
    df_reduced.columns = df_name + "_pca_" + df_reduced.columns.astype(str)
    return df_reduced


# テキストデータの埋め込み&次元削減関数（Transformer + PCA）
def transformer_pca_features(
    df, 
    model_name='intfloat/multilingual-e5-base', 
    n_components=10, 
    random_state=42
):
    df_embeddings = transformer_embedding(
        df, 
        model_name=model_name, 
    )
    df_reduced = pca_reduction(
        df_embeddings, 
        n_components=n_components, 
        df_name=df.name, 
        random_state=random_state
    )
    return df_reduced
