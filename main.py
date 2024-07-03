from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
import time
from typing import Set

st.header("Langchain Udemy Course - Documentation Helper Bot")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_prompt_history" not in st.session_state:
    st.session_state["chat_prompt_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

prompt = st.text_input(label="Prompt", placeholder="Enter your prompt here...")


def print_sources_string(docs: Set[str]) -> str:
    if not docs:
        return ""
    print("docs", docs)
    list_sources = list(docs)
    list_sources.sort()
    print("list_sources:", list_sources)
    sources_string = "sources for this answer: "
    for i, source in enumerate(list_sources):
        sources_string += f"\n{i+1} - {source}"

    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        print("sources:", sources)
        formatted_response = (
            f"{generated_response['answer']}\n\n{print_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_prompt_history"].append(formatted_response)

        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if st.session_state["chat_prompt_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_prompt_history"], st.session_state["user_prompt_history"]
    ):
        # if the user sent the prompt
        message(user_query, is_user=True)
        message(generated_response)
