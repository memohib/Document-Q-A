import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage


## Adding the Chat History
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_Token'] = os.getenv('HF_Token')
groq_api_key=os.getenv('GROQ_API')  


llm = ChatGroq(groq_api_key=os.getenv("Groq_api")  ,model_name="Llama3-8b-8192")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

prompt = ChatPromptTemplate.from_template(
    
    """
    Answer the questions based on the provided context only.
    Provide the most accurate response based on the question
    
    <context>
    {context}
    <context>

    Question:{input}

"""
    

)


## Chat History Prompt


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (prompt),
        MessagesPlaceholder("chat_history"),
                           ("human", "{input}"),
      
                          ]
                                        )


def create_vector_embedding():
       
       if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
st.title("RAG Document Q&A With Groq And Lama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    

if user_prompt:
     retriever=st.session_state.vectors.as_retriever()
     history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
     question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
     rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
     
     response=rag_chain.invoke({"input":user_prompt, "chat_history":st.session_state.chat_history})
     
     st.session_state.chat_history.extend(
          [
                HumanMessage(content=user_prompt),
                AIMessage(content=response["answer"])
          ]
     )       
     st.write(response['answer'])
     
     
