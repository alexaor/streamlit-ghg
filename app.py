import streamlit as st
import os
import pinecone
from llama_index import (
    VectorStoreIndex,
    set_global_tokenizer,
    ServiceContext,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import Replicate
from transformers import AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding


st.title("GHG Protocol Assistant")
st.write("Connecting to relevant data sources...")


os.environ["REPLICATE_API_TOKEN"] = st.secrets["replicate_api_token"]

llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)
set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

pinecone.init(api_key=st.secrets["pinecone_api_key"], environment="gcp-starter")
pinecone_index = pinecone.Index("ghg-protocol")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, service_context=service_context
)
query_engine = index.as_query_engine()

st.write("Pinecone connected")
with st.chat_message("assistant"):
    st.write("Hi! I am your GHG consultant. Ask me about anything related to GHG scope 2 or 3")

prompt = st.chat_input("What is your query?")

if prompt:
    st.write("Thinking...")
    response = query_engine.query(prompt)
    st.write(response.response)
    # Consider spitting out context as well




# res = query_engine.query("What is the scope 2 emissions of a car?")
