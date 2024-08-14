import pinecone
from langchain.vectorstores import PineconeVectorStore

# Initialize Pinecone
api_key = "0d2cdf9a-22c7-4051-a247-118c258f316f"
environment = "us-east-1"
index_name = "ai-knowledge-base"

pinecone.init(api_key=api_key, environment=environment)

# Check if the index exists, and create it if it doesn't
if index_name not in pinecone.list_indexes():
    print(f"Creating index: {index_name}")
    pinecone.create_index(index_name, dimension=128)

# Connect to the new index
index = pinecone.Index(index_name)

# Your existing code to create embeddings and list of documents
list_of_all_docs = [...]  # Your list of documents
embeddings = ...  # Your embeddings instance

# Create PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(list_of_all_docs, embedding=embeddings, index_name=index_name)

# Proceed with the rest of your code
print(f"Document search vector store created with index: {index_name}")
