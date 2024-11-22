import os  # For accessing os level directories
from langchain_community.document_loaders import PyPDFLoader  # For loading the PDF
from langchain.prompts import PromptTemplate  # Prompt template
from langchain_pinecone import PineconeVectorStore  # Vector Database
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For Chunking
from langchain.chains import RetrievalQA  # For Retrieval
from langchain_groq import ChatGroq  # Inference Engine
from dotenv import load_dotenv  # For detecting env variables
from langchain_community.embeddings import OllamaEmbeddings  # Updated import for vector embeddings
import chainlit as cl  # For user interface

load_dotenv()  # Detecting environment variables

# Defining prompt
prompt_template = """
You are an assistant for the Enigma website, a personalized energy hub. Provide accurate and helpful responses to questions about Enigma's features, plans, and services. Base your answers solely on the information provided in the context. If the information isn't available in the context, say so instead of making up information.

Context: {context}
Question: {question}

Helpful answer:
"""

# Function to interact with the prompt template
def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return prompt

# Function to perform retrieval
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Function to define the LLM model
def load_llm():
    groqllm = ChatGroq(
        model="llama3-8b-8192", temperature=0.1
    )
    return groqllm

# Function for loading PDF, chunking, vector embeddings, and storing embeddings in Pinecone vector database
def qa_bot():
    data = PyPDFLoader(r'C:\Users\Diyotrim Maitra\Downloads\chatbot.pdf')
    loader = data.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    splitdocs = chunk.split_documents(loader)
    index_name = "energy"

    # Set Pinecone API key from environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is not set. Please set the PINECONE_API_KEY environment variable.")

    db = PineconeVectorStore.from_documents(
        splitdocs, 
        OllamaEmbeddings(model="mxbai-embed-large"), 
        index_name=index_name
    )

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Chainlit decorator for starting the app
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Welcome to EnigmaBot!⚡🤖 Hello! I'm EnigmaBot, your personal energy assistant. How can I help you with your energy-related questions today? Feel free to ask about our services, energy-saving tips, or any other energy queries you may have."
    await msg.update()

    cl.user_session.set("chain", chain)

# Functionality to handle user messages
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    res = await chain.acall({'query': message.content})
    answer = res['result']
    await cl.Message(content=answer).send()