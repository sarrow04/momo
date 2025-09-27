import streamlit as st
import pandas as pd
from transformers import pipeline
from collections import Counter
from janome.tokenizer import Tokenizer
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import numpy as np # エラー回避のために追加

# --- データ読み込み関数（キャッシュで高速化）---
# アップロードされたファイルを直接読み込むように変更
@st.cache_data(ttl=600) # 10分間キャッシュを保持
def load_data(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)
        st.success("データの読み込みが完了しました！")
        return df
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return pd.DataFrame()

# --- データ分析関数 ---

# 感情分析モデルの読み込み（キャッシュで高速化）
@st.cache_resource
def get_sentiment_analyzer():
    # モデルが存在しない場合のエラーハンドリングを追加
    try:
        return pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")
    except Exception as e:
        st.error(f"感情分析モデルの読み込みに失敗しました。インターネット接続を確認してください。: {e}")
        return None

# 1. 感情分析
def analyze_sentiment(df, column):
    analyzer = get_sentiment_analyzer()
    if analyzer is None:
        return df # モデルが読み込めなかった場合は何もしない
        
    # モデルの最大入力長に合わせてテキストを切り詰める
    # .locを使って安全に列を更新
    df_copy = df.copy()
    # テキストがNoneや数値の場合を考慮して、安全に文字列に変換
    sentiment_results = df_copy[column].fillna("").astype(str).str[:512].apply(lambda x: analyzer(x)[0] if x else {'label': 'NEUTRAL', 'score': 0.0})
    df_copy['sentiment_label'] = [res['label'] for res in sentiment_results]
    df_copy['sentiment_score'] = [res['score'] for res in sentiment_results]
    return df_copy

# 2. 単語頻度分析
def word_frequency_analysis(df, column, stop_words):
    t = Tokenizer()
    words = []
    # 名詞、動詞、形容詞のみを抽出
    for text in df[column].fillna("").astype(str):
        tokens = t.tokenize(text)
        words.extend([token.base_form for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']])
    
    # 1文字の単語や不要な記号を除外
    words = [word for word in words if len(word) > 1 and not re.match(r'^[!-~]+$', word)]
    # ストップワードを除外
    words = [word for word in words if word not in stop_words]
    
    word_counts = Counter(words)
    freq_df = pd.DataFrame(word_counts.most_common(20), columns=['単語', '出現回数'])
    return freq_df

# 3. グループ分け（クラスタリング）
def clustering_texts(df, column, num_clusters=3, stop_words=None):
    df_copy = df.copy() # 元のデータフレームを変更しないようにコピー
    t = Tokenizer()
    def tokenize(text):
        return [token.base_form for token in t.tokenize(text) if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(df_copy[column].fillna("").astype(str))
    
    # ★★★ 修正箇所 ★★★
    # n_init='auto' から n_init=10 に変更
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) 
    df_copy['cluster'] = kmeans.fit_predict(tfidf_matrix)
    return df_copy, vectorizer, kmeans

# --- Streamlit UI ---

st.set_page_config(page_title="テキスト分析アプリ", layout="wide")
st.title('📝 テキストデータ分析アプリ')
st.markdown("お手元のCSVファイルをアップロードして、テキストデータを分析します。")

# --- サイドバー (入力エリア) ---
with st.sidebar:
    st.header("⚙️ 設定")
    # CSVアップロード
    uploaded_file = st.file_uploader("① CSVファイルをアップロード", type=['csv'])
    
    # ストップワード入力
    stop_words_input = st.text_area("③ ストップワード（カンマ区切り）", "こと,もの,ため,よう,さん,これ,それ,いる,する,ある,ない,いう,思う")
    
    df_original = pd.DataFrame()
    if uploaded_file:
        df_original = load_data(uploaded_file)
    
    if not df_original.empty:
        # object型の列（テキスト列）のみを選択肢に表示
        text_columns = df_original.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            text_column = st.selectbox("② 分析対象のテキスト列を選択", text_columns)
            analyze_button = st.button("📈 分析実行", type="primary")
        else:
            st.warning("ファイルにテキスト形式の列が見つかりません。")
            analyze_button = False
    else:
        st.info("CSVファイルをアップロードしてください。")
        analyze_button = False

# --- メイン画面 (結果表示エリア) ---
if analyze_button:
    st.header("📊 分析結果")

    # テキストエリアからストップワードのリストを作成
    stop_words = [word.strip() for word in stop_words_input.split(',') if word.strip()]
    with st.expander("除外するストップワード一覧"):
        st.write(stop_words)
    
    col1, col2 = st.columns(2)

    # --- 1. 感情分析 ---
    with col1:
        with st.spinner("1. 感情分析を実行中...（初回は時間がかかります）"):
            df_sentiment = analyze_sentiment(df_original.copy(), text_column)
        st.subheader("😃 ポジティブ/ネガティブ分析")
        if 'sentiment_label' in df_sentiment.columns:
            fig_pie = px.pie(df_sentiment, names='sentiment_label', title='感情の割合', 
                             color='sentiment_label', color_discrete_map={'POSITIVE':'#2ca02c', 'NEGATIVE':'#d62728', 'NEUTRAL':'#7f7f7f'})
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(df_sentiment[['sentiment_label', 'sentiment_score', text_column]].head())
        else:
            st.error("感情分析を実行できませんでした。")


    # --- 2. 単語の出現頻度 ---
    with col2:
        with st.spinner("2. 単語の出現頻度を計算中..."):
            df_freq = word_frequency_analysis(df_original, text_column, stop_words=stop_words)
        st.subheader("🗣️ 頻出単語トップ20")
        if not df_freq.empty:
            fig_bar = px.bar(df_freq, x='単語', y='出現回数', title='単語の出現回数ランキング', color_discrete_sequence=["#1f77b4"])
            fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("有効な単語が抽出できませんでした。")

    st.divider()

    # --- 3. グループ分け（クラスタリング） ---
    st.subheader("👨‍👩‍👧‍👦 テキストのグループ分け")
    num_clusters = st.slider("グループ数を選択", min_value=2, max_value=10, value=3)
    
    with st.spinner(f"3. {num_clusters}個のグループに分類中..."):
        df_clustered, vectorizer, kmeans = clustering_texts(df_original.copy(), text_column, num_clusters, stop_words=stop_words)
    
    # 各クラスターの特徴的な単語を表示
    try:
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        cluster_cols = st.columns(num_clusters)
        for i in range(num_clusters):
            with cluster_cols[i]:
                st.markdown(f"**グループ {i+1}**")
                # 特徴的な単語を表示
                top_words = [terms[ind] for ind in order_centroids[i, :5]]
                st.info(f"キーワード: {', '.join(top_words)}")
                # 各クラスターのサンプルテキストを表示
                st.dataframe(df_clustered[df_clustered['cluster'] == i][[text_column]], height=200, hide_index=True)
    except Exception as e:
        st.error(f"クラスタリング結果の表示中にエラーが発生しました: {e}")

else:
    st.info("サイドバーで設定を行い、「分析実行」ボタンを押してください。")
