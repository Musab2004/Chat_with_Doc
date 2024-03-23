import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
flag=False
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def process_pdf(pdf_path):
        if not pdf_path:
            raise gr.Error("Error ! Please Upload a Pdf File")
            return "Error ! Please Upload a Pdf File"
        if not pdf_path.endswith(".pdf"):
             
             raise gr.Error("Error ! only Pdf are Allowed")
             return "Error ! only Pdf are Allowed"
        start_time = datetime.now()
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        print('pdf_path', pdf_path)
        texts = text_splitter.split_documents(pages)
        # print("texts", texts)
        vectorstore = Chroma.from_documents(documents=texts, embedding = HuggingFaceEmbeddings(),persist_directory="./chroma_db")
        vectorstore.persist()
        print("stored inn chroma db")
        retriever = vectorstore.as_retriever()
        gr.Info("Your File is Uploaded Successfully! You can chat with the bot now.")
        return "Your File is Uploaded Successfully!"
        # processing_time = datetime.now() - start_time
        # return True, f"Upload and processing completed in {processing_time.seconds} seconds."
chat_history = []
def rag_inference(chat_history1,question):
    new_question="According to this document, "+question 
    print("rag chain created",question)
    vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(api_key=api_key,model_name="gpt-3.5-turbo", temperature=0)
    def format_docs(docs):
            # print("docs",docs)
            return "\n\n".join(doc.page_content for doc in docs)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        if the no context provided, you can just say i don't know. don't answer the question.\
        if the context provided is not enough, you can ask for more context.\
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )


    def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]
    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )
    print(chat_history)
    ai_msg = rag_chain.invoke({"question": new_question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg])
    yield chat_history1 + [(question, ai_msg)]

def app(pdf, question):
    retriever = process_pdf(pdf.name)
    answer = rag_inference(retriever, question)
    return answer
def reset_database():
    vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=HuggingFaceEmbeddings())
    if vectorstore._collection.count() > 0:
        vectorstore._collection.delete(ids=vectorstore._collection.get()["ids"])
        gr.Info("Your Database is Deleted Successfully! You can upload a new document now.")
        return "Deleted the database Succesfully"
    return "Already Empty! You can upload a new document now."
with gr.Blocks() as demo:
    gr.Markdown('# Welcome to retrieval augmented generation (RAG) Chatbot!')
    gr.Markdown("""
## Tutorial: How to Use This App

1. **Upload a Document**: Click on the 'Upload Document' tab. You will see an option to upload a file. Click on it and select the PDF document you want to upload.

2. **Build the Chat Bot**: After you've selected a document, click on the 'Build the Chat Bot !' button. The app will process your document and use it to build a chat bot.

3. **Ask Questions**: Once the chat bot is built, you can ask it questions about the document you uploaded. Just type your question into the textbox and press Enter.

4. **Reset the ChatBot**: If you want to upload a new document and build a new chat bot, you can reset the current chat bot by clicking on the 'Reset the Chat Bot !' button. This will delete the current chat bot and allow you to upload a new document.

Remember, the chat bot uses the document you uploaded to answer your questions. So, the quality of the answers will depend on the quality of the document you upload.

Happy chatting!
""")
    with gr.Tab("Upload Document"):
        file = gr.File(type="filepath")
        # file_input = gr.File(label="Upload PDF")
        process_button = gr.Button("Build the Chat Bot !")
        reset_button = gr.Button("Reset the Chat Bot !")
        output_text = gr.Textbox()

        # Define the layout
        interface = gr.Interface(
            fn=None,  # Initially, no function is called.
            inputs=[file, process_button, reset_button],
            outputs=output_text,
            live=True
        )

        # Bind the buttons to their respective functions
        process_button.click(fn=process_pdf, inputs=file, outputs=output_text)
        reset_button.click(reset_database,outputs=output_text)

    with gr.Tab("Chat Bot"):
        chatbot = gr.Chatbot()
        message = gr.Textbox ("What is this document about?")
        message.submit(rag_inference, [chatbot,message], chatbot)
    
    # Your existing "Chat Now!" tab code here

demo.launch(debug=True,share=True)