import os
import pinecone
from llama_index.storage.storage_context import StorageContext
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    set_global_tokenizer,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import Replicate
from transformers import AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding
from dotenv import load_dotenv

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
SCOPE_2_PDF = os.path.join(DATA_PATH, "scope2.pdf")
SCOPE_3_PDF = os.path.join(DATA_PATH, "scope3.pdf")

load_dotenv()
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


def init_db():
    print("Initializing vector db")
    print("Connecting to pinecone")
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

    print("Loading models...")
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

    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    pinecone_index = pinecone.Index("ghg-protocol")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        service_context=service_context,
    )


if __name__ == "__main__":
    init_db()
