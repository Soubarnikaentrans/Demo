from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pdf_extractor import Extract_TextTable_Info_With_Tables_Renditions_From_PDF
import glob
import os
import zipfile
from datetime import datetime
import pandas as pd
import os
import re
import json
import logging
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pinecone
import os
import sys
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)

load_dotenv()

CLIENT_ID = os.getenv("ADOBE_CLIENT_ID")
CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET")


if not CLIENT_ID and not CLIENT_SECRET:
    raise ValueError("ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET environment variables must be set")


input_pdf_path = "C:/Users/User/Documents/Chatbot_New/pdf/report.pdf"

input_pdf_files = glob.glob('C:/Users/User/Documents/Chatbot_New/pdf/*.pdf')
# glob.glob('C:/Users/User/Documents/Chatbot_New/pdf/*.pdf')
make_output_path = "output/Extract-TextTableInfo"


output_paths = []
for input_pdf_path in input_pdf_files:
    extractor = Extract_TextTable_Info_With_Tables_Renditions_From_PDF(input_pdf_path,
                                                                        make_output_path,
                                                                        CLIENT_ID,
                                                                        CLIENT_SECRET)

    zip_files = extractor.output_path()
    output_paths.append(zip_files)

    
    print('zip_file--------->',zip_files)
    print(output_paths)



json_dict = {
    "output_paths":output_paths
}

json_file_path = "output/output_paths.json"

os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

with open(json_file_path, "w") as json_file:
    json.dump(json_dict, json_file, indent=4)

    


# SECOND TASK
# UNZIP

output_zipped_files = glob.glob('C:/Users/User/Documents/Chatbot_New/pdf/*.zip')
extract_dir = 'C:/Users/User/Documents/Chatbot_New/unzip/'


with open("output/output_paths.json", "r") as json_file:
    json_dict = json.load(json_file)
    output_zipped_files = json_dict["output_paths"]
    
    # output_paths.extend(json_dict.get("output_paths", []))

filenames = [os.path.splitext(os.path.basename(path))[0] for path in output_zipped_files]

for zip_file, filename in zip(output_zipped_files, filenames):
    extract_path = os.path.join(extract_dir, filename)
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:

        # Extract all the contents into the extraction directory
        zip_ref.extractall(extract_path)
        print(f'Extracted {zip_file} to {extract_path}')


# THIRD TASK


# #CHUNK


def parse_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        content = json.loads(json_file.read())

    pdf_element = content["elements"]
    return pdf_element


def chunk_parse_pdf(unzip_dir, chunked_dir, op_json_file_path):

    try:
        elements = parse_json(op_json_file_path)

        file_split = 0
        # Define the header flag. If first time header no need to cut a new file
        FIRST_TIME_HEADER = True
        file_name = os.path.join(chunked_dir, f"file_{file_split}.txt")
        parsed_file = open(file_name, "a", encoding="utf-8")

        print("processing........................................................")
        chunck_no = 1
        for element in elements:

            chunck_no += 1
            print("1 processing........................................................")
            if "//Document/H2" in element["Path"]:
                print("2 processing........................................................")
                hdr_txt = element["Text"]
                if FIRST_TIME_HEADER:
                    print("3 processing........................................................")
                    FIRST_TIME_HEADER = False

                    parsed_file.write(hdr_txt)
                    parsed_file.write("/n")
                else:
                    print("3 processing........................................................")
                    parsed_file.close()
                    file_split = file_split + 1
                    file_name = os.path.join(chunked_dir, f"file_{file_split}.txt")
                    parsed_file = open(file_name, "a", encoding="utf-8")
                    parsed_file.write(hdr_txt)
                    parsed_file.write("/n")

            else:
                if "filePaths" in element:
                    if "Table" in element['Path']:
                        # print('end---------------------------------'*5,element['filePaths'][0])
                        element["filePaths"]
                        print("4 processing........................................................")
                        csv_file_name = element["filePaths"][0]
                        print("xlsx_file_name", csv_file_name)
                        csv_file = os.path.join(unzip_dir, csv_file_name)
                        df = pd.DataFrame(pd.read_csv(csv_file))
                        table_content = df.to_json().replace("_x000D_", " ")
                        parsed_file.write(table_content)
                        parsed_file.write("/n")
                        print("5 processing........................................................")

                elif "Table" in element["Path"]:
                    continue

                else:
                    try:
                        print("6 processing........................................................")
                        text_content = element["Text"]
                        parsed_file.write(text_content)
                        parsed_file.write("/n")
                    except KeyError as ke:
                        pass

        parsed_file.close()

    except Exception as e:
        print(e)
        logging.exception("Exception encountered while executing operation")


# # # # Step - 1 : Run this step to chunk the PDF into contextual subsections

unzip_dir = "C:/Users/User/Documents/Chatbot_New/unzip/"
chunked_dir_base = "C:/Users/User/Documents/Chatbot_New/output/chunks"

for folder in os.listdir(unzip_dir):

    chunked_dir = os.path.join(chunked_dir_base, folder)

    json_file_path = os.path.join(unzip_dir, folder, "structuredData.json")
    isExist = os.path.exists(chunked_dir)

    if not isExist:
        os.makedirs(chunked_dir)

    unzip_dir_folder = os.path.join(unzip_dir, folder)
    chunk_parse_pdf(unzip_dir_folder, chunked_dir, op_json_file_path=json_file_path)



# FOURTH Task
# #TEXT_LOADER

def get_files_from_dir(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    return files


def load_docs(file_path):
    loader = TextLoader(file_path, encoding='UTF-8')
    docs = loader.load()

    return docs


# # Step - 2 : use a TextLoader to get all the chunks in a list of Dcoments

list_of_all_docs = []

for folder in os.listdir(unzip_dir):

    chunked_dir = os.path.join(chunked_dir_base, folder)
    files = [os.path.join(chunked_dir, f) for f in os.listdir(chunked_dir) if
             os.path.isfile(os.path.join(chunked_dir, f))]
    # print("files----->", files)

    for file in files:
        document = load_docs(file)
        # print(document)
        list_of_all_docs.append(document[0])

# print("list_of_all_docs------>",list_of_all_docs)

# embeddings=GooglePalmEmbeddings(google_api_key="AIzaSyAti_D-4tknChQpLf9f4XtZL_hI8xEEYRk")


# FIFTH TASK
# #UPLOAD PINECOne

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


os.environ["PINECONE_API_KEY"] = "44f15e14-2dfb-420a-92b4-9aedc73ed0a9"



index_name = "keyword-accuracy"

import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key="44f15e14-2dfb-420a-92b4-9aedc73ed0a9"
)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        # api_key="44f15e14-2dfb-420a-92b4-9aedc73ed0a9",
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
# Function to reduce metadata size
def optimize_metadata(metadata):
    # Example: Keep only essential keys
    essential_keys = ['title', 'author', 'date']
    optimized_metadata = {key: metadata[key] for key in essential_keys if key in metadata}
    return optimized_metadata

optimized_docs = []
for doc in list_of_all_docs:
    optimized_metadata = optimize_metadata(doc.metadata)  # Access metadata attribute
    optimized_docs.append(Document(page_content=doc.page_content, metadata=optimized_metadata))

docsearch = PineconeVectorStore.from_documents(optimized_docs, embedding=embeddings, index_name=index_name)



# SIXTH TASK
# # #PASSING THE RETRIVALS TO LLM

doc = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "What is the Abbreviation of ISO?"
retriever = doc.as_retriever(search_type="similarity", 
                            #  search_kwargs={"k": 2}
                             #MMR Maximum Marginal Retrival
                              search_kwargs={'k': 2}
                             )

print('====================')
print(retriever.get_relevant_documents(query))


retrievers = doc.as_retriever(search_type="similarity", search_kwargs={"k": 2}
                             #MMR Maximum Marginal Retrival
                             #  search_kwargs={'k': 2, 'lambda_mult': 0.25}
                             )

# print('docs ====>',docs)

GOOGLE_AI_STUDIO_API_KEY = "AIzaSyAti_D-4tknChQpLf9f4XtZL_hI8xEEYRk"

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO_API_KEY)


def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)


prompt = """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
Question: {question} 

Context: {context} 

Answer:""" """


prompt = hub.pull("rlm/rag-prompt")

# print(prompt,'<=====Prompt/n/n====>type',type(prompt))

rag_chain = (
        {"context": retrievers | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)



query = "What are the activities that can require segregation?"
# rag_chain.invoke(query)
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)

