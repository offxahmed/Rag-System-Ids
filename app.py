import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Global variable to store the retrieval chain
rag_chain = None
chat_history = []

def process_pdf(file):
    global rag_chain, chat_history
    if not file:
        return "Please upload a PDF file."
    
    try:
        # Load PDF
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        # Using HuggingFace Embeddings (runs locally)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Set up RAG chain with Grok
        # Grok is compatible with OpenAI API
        llm = ChatOpenAI(
            openai_api_key=os.getenv("GROK_API_KEY"),
            openai_api_base="https://api.x.ai/v1",
            model_name="grok-beta",
            temperature=0
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        chat_history = [] # Reset history on new file
        return "PDF processed successfully! You can now ask questions."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def answer_question(message, history):
    global rag_chain
    if not rag_chain:
        return "Please upload and process a PDF file first."
    
    try:
        response = rag_chain.invoke({"input": message})
        return response["answer"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 RAG System with Gradio (Powered by Grok)")
    gr.Markdown("Upload a PDF document and ask questions about its content.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=answer_question,
                additional_inputs=[],
                title="Chat with your Document",
                description="Ask questions based on the uploaded PDF.",
            )

    process_btn.click(process_pdf, inputs=[file_input], outputs=[status_output])

if __name__ == "__main__":
    demo.launch()
