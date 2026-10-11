"""Streamlit chat UI: streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag.generate import answer  # noqa: E402

st.set_page_config(page_title="SQuAD RAG", page_icon="🔎")
st.title("🔎 SQuAD v2 RAG (Chroma + Ollama)")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Ask about a Wikipedia topic in SQuAD..."):
    st.chat_message("user").markdown(question)
    with st.spinner("Searching and answering..."):
        text, chunks = answer(question)
    with st.chat_message("assistant"):
        st.markdown(text)
        with st.expander("Sources"):
            for n, c in enumerate(chunks, 1):
                st.markdown(f"**[{n}] {c.title}** (score {c.score:.2f})\n\n{c.text}")
    st.session_state.history += [("user", question), ("assistant", text)]
