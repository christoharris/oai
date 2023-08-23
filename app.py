import os
import time
import argparse
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI, HuggingFaceHub, CTransformers
from langchain.llms import GPT4All, LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import TextLoader, PyPDFLoader

load_dotenv()

# page configuration
st.set_page_config(page_title='Offline, Ai', page_icon=":zap:")
st.title('Offline, Ai')

# load environment variables
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
model_path2 = os.environ.get('MODEL2_PATH')

# Where PDFs are uploaded
pdf_docs = st.file_uploader(
        "Upload PDFs here", accept_multiple_files=True)

if pdf_docs:    
    if st.button('Process'):
        # global vectorstore
        st.session_state['button'] += 1
        with st.spinner('Processing'):
            # create the pdf text
            time.sleep(3)
            
if 'button' not in st.session_state:
    st.session_state.button = 0

if st.session_state['button']:

    def parse_arguments():
        parser = argparse.ArgumentParser(description='Offline, Ai: Ask questions to your documents without an internet connection, '
                                                    'using the power of LLMs.')
        parser.add_argument("--hide-source", "-S", action='store_true',
                            help='Use this flag to disable printing of source documents used for answers.')

        parser.add_argument("--mute-stream", "-M",
                            action='store_true',
                            help='Use this flag to disable the streaming StdOut callback for LLMs.')

        return parser.parse_args()

    # PDF into a string variable
    def get_pdf_text(pdf_docs):
        text = ''
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    # String broken into chunks
    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    
    def get_vectorstore(text_chuncks):
        embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
        vectorstore = FAISS.from_texts(texts=text_chuncks, embedding=embeddings)
        return vectorstore


    
    raw_text = get_pdf_text(pdf_docs)
    # create the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks) 

    llm = OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
    # args = parse_arguments()
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm2 = GPT4All(model=model_path2, max_tokens=1000, backend='gptj', n_batch=model_n_batch, verbose=False)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    qa = RetrievalQA.from_chain_type(llm=llm2, chain_type="stuff", retriever=retriever, return_source_documents=True)    

    # intialize session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I assist you?'}]

    for message in st.session_state.messages:
        st.chat_message(message['role']).write(message['content'])


    # Conversation Dynamics
    if prompt := st.chat_input(placeholder='Ask a question'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        
        with st.chat_message("assistant"):
            # st.write("🧠 thinking...")
            st_callback = StreamlitCallbackHandler(st.container())
            # response = agent.run(prompt) # respond to user with agent
            response = qa(prompt)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.markdown(response['result'])









