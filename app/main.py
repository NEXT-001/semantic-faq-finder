# ディレクトリ構成の前提:
# project/
# ├── app/
# │   ├── main.py          ← Streamlit UI
# │   ├── search.py        ← 類似度検索処理
# │   ├── embedding.py     ← OpenAI埋め込み生成
# │   ├── faq_loader.py    ← JSON読み込みとデータ管理
# │   └── config.py        ← モデル・設定情報
# ├── data/
# │   └── company_x_faq.json
# └── .env

# 以下は app/main.py の内容

import streamlit as st
from app.search import search_faq
from app.faq_loader import load_faq_data
from app.config import COMPANIES, DEFAULT_COMPANY

st.set_page_config(page_title="📚 FAQ検索システム", layout="centered")
st.title("📚 FAQ 自動応答ボット")
st.caption("OpenAI Embedding + FAISS + JSONデータ")

# --- 企業選択 ---
company = st.selectbox("企業を選択してください：", COMPANIES, index=COMPANIES.index(DEFAULT_COMPANY))

# --- 入力 ---
query = st.text_input("質問を入力してください：")
threshold = st.slider("類似度しきい値", 0.0, 1.0, 0.7, 0.01)

# --- FAQ検索 ---
if query:
    with st.spinner("検索中..."):
        df, index, vectors, lang = load_faq_data(company)
        results = search_faq(query, df, index, vectors, lang, threshold=threshold)

    if results:
        st.subheader("🔍 最も近いFAQ：")
        for r in results:
            st.markdown(f"**Q: {r['question']}**")
            st.markdown(f"A: {r['answer']}")
            st.caption(f"類似度スコア: {r['score']}")
            st.markdown("---")
    else:
        st.warning("該当するFAQが見つかりませんでした。")
