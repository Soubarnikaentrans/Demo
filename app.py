# import streamlit as st
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from pinecone import Pinecone
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set up the API keys and other credentials
# api_key = os.getenv("PINECONE_API_KEY")
# GOOGLE_AI_STUDIO_API_KEY = "AIzaSyAti_D-4tknChQpLf9f4XtZL_hI8xEEYRk"

# # Initialize embeddings and vector store
# embeddings = HuggingFaceEmbeddings()
# index_name = "keyword-accuracy"

# pc = Pinecone(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment="us-east-1-aws"
# )
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         # api_key="44f15e14-2dfb-420a-92b4-9aedc73ed0a9",
#         name=index_name,
#         dimension=768,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )
# docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# # Set up the retriever
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# # Initialize the LLM (Google Generative AI)
# llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO_API_KEY)

# # Define the prompt template
# prompt_template = hub.pull("rlm/rag-prompt")

# # Create the RAG chain
# rag_chain = (
#     {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
#      "question": RunnablePassthrough()}
#     | prompt_template
#     | llm
#     | StrOutputParser()
# )

# # Streamlit UI
# st.title("PDF Question Answering")

# query = st.text_input("Enter your query:", placeholder="What is the Abbreviation of ISO?")

# if st.button("Get Answer"):
#     if query:
#         st.write("Processing your query...")

#         # Get the answer using the RAG chain
#         answer = ""
#         for chunk in rag_chain.stream(query):
#             answer += chunk

#         st.write("### Answer:")
#         st.write(answer)
#     else:
#         st.write("Please enter a query to get an answer.")


import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import Pinecone,ServerlessSpec
import os
from dotenv import load_dotenv
from pdfminer.high_level import extract_text  # Assuming you use pdfminer to extract text

# Load environment variables
load_dotenv()

# Set up the API keys and other credentials
api_key = os.getenv("PINECONE_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
# Retrieve the LangSmith API key from the environment variables
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Check if the API key is available
if not LANGSMITH_API_KEY:
    raise ValueError("LangSmith API key not found. Please set the LANGSMITH_API_KEY environment variable.")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
index_name = "keyword-accuracy"

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1-aws"
)
# if index_name not in pc.list_indexes():
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )
# else:
#     print(f"Index '{index_name}' already exists. Skipping creation.")
# Example function to chunk documents
embeddings = HuggingFaceEmbeddings()
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def chunk_document(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

# Function to add documents with smaller metadata
def add_documents_with_chunking(documents):
    for doc in documents:
        text_chunks = chunk_document(doc.page_content)
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = {
                'source': doc.metadata.get('source'),
                'chunk': i
            }
            # Ensure metadata size is within limits
            if len(str(chunk_metadata)) > 40960:
                chunk_metadata = {'source': doc.metadata.get('source')}
            new_doc = Document(page_content=chunk, metadata=chunk_metadata)
            docsearch.add_documents([new_doc])

# Example documents list
docs = [Document(page_content="Very large document content...", metadata={"source": "C:/Users/User/Documents/Chatbot_New/pdf/report.pdf"})]

# Add documents to Pinecone with chunking
add_documents_with_chunking(docs)
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Populate the Pinecone index with PDF content
# pdf_path = "C:/Users/User/Documents/Chatbot_New/pdf/report.pdf"
# pdf_text = extract_text(pdf_path)  # Extract text from the PDF
# docs = [Document(page_content=pdf_text)]  # Create a list of Document objects
# docsearch.add_documents(docs)  # Add documents to Pinecone

# Set up the retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Initialize the LLM (Google Generative AI)
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO_API_KEY)

# Define the prompt template
prompt_template = hub.pull("rlm/rag-prompt",api_key=LANGSMITH_API_KEY)

# Create the RAG chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("PDF Question Answering")

query = st.text_input("Enter your query:", placeholder="Ask...")

if st.button("Get Answer"):
    if query:
        st.write("Processing your query...")

        # Get the answer using the RAG chain
        answer = ""
        for chunk in rag_chain.stream(query):
            answer += chunk

        st.write("### Answer:")
        st.write(answer)
    else:
        st.write("Please enter a query to get an answer.")
