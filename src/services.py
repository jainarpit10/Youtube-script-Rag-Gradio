import os
from datetime import datetime
from pinecone import Pinecone
from dotenv import load_dotenv,find_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI


load_dotenv(find_dotenv())
MODEL =  os.getenv("MODEL")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
ENVIRNOMENT = os.getenv("ENVIRNOMENT")
PINECONE_DIMENSION = os.getenv("PINECONE_DIMENSION")

AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION=os.getenv("AZURE_API_VERSION")
AZURE_API_KEY=os.getenv("AZURE_API_KEY")
AZURE_MODEL=os.getenv("AZURE_MODEL")
AZURE_DEVELOPMENT=os.getenv("AZURE_DEVELOPMENT")

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    model=AZURE_MODEL
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    azure_deployment=AZURE_DEVELOPMENT,
    temperature=0.0
)

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        full_transcript = " ".join([item.text for item in transcript_list])
        save_transcript_into_pinecone(full_transcript,video_id)
        return True
    except Exception as e:
        print("Error is :",str(e))
        return False
    

def save_transcript_into_pinecone(transcript_data,video_id):
    chunks = load_and_chunk_pdf(transcript_data)
    save_to_pinecone(chunks,video_id)

def load_and_chunk_pdf(transcript_data, chunk_size=800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=['\n\n','\n','. ','? ','! ',';'],
        # is_separator_regex=False
    )
    print(f"File splits into chunks.")
    return text_splitter.create_documents([transcript_data])


#Function to Save to Pinecone
def save_to_pinecone(chunks,video_id):
    docs = []

    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "video_id": video_id,
            "page": i + 1,
            "timestamp":datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        docs.append(chunk)


    index_name =INDEX_NAME
    try:
        PineconeVectorStore.from_documents(
            docs,
            embedding=embeddings,
            index_name=index_name
        )
        print(f"Chunks saved into pinecone index.")
    except Exception as e:
        print("Error is",str(e))


template = """
        You are answering a question based only on the YouTube transcript context provided below and remember our previous conversations.

        ---BEGIN CONTEXT---
        {context}
        ---END CONTEXT---

        User Question:
        {question}

        Instructions:
        - Answer strictly and only from the context above.
        - If the answer is not found in the context, say:
        "The answer is not available in the provided video transcript."
        - Provide a clear and concise answer.
"""


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Query Answering
def user_query_response(query,video_id,top_k=5):
    vectordb = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k":top_k,
            "filter":{"video_id": {"$in":[video_id]}},
            "score_threshold": 0.3
        })

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT
            }
    )
    result = qa_chain.invoke(query)
    clean_result = result["result"].strip()

    source_docs = result.get("source_documents", [])
    if not source_docs or not clean_result:
        return {
            "query": query,
            "result": "No data found"
        }
    
    if "not available" in clean_result.lower():
        return {
            "query": query,
            "result": clean_result,
            "pages": [],
            "sources": []
        }

    return {
        "query": query,
        "result": clean_result,
        "pages": list(set(doc.metadata['page'] for doc in source_docs)),
        "sources": [
                # {
                #     "page": doc.metadata["page"],
                #     "start_time": doc.metadata["start_time"],
                #     "end_time": doc.metadata["end_time"]
                # }
                # for doc in source_docs
            ]
    }

    # https://www.youtube.com/watch?v=vJOGC8QJZJQ&t=47s