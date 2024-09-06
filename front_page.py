import os
import json
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_chat import message
from streamlit_lottie import st_lottie_spinner

# Load environment variables
load_dotenv()

def render_animation():
    path = "assets/typing_animation.json"
    with open(path, "r") as file: 
        animation_json = json.load(file)
        return animation_json

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

st.set_page_config(
    page_title="Softsquare AI",
    page_icon="🤖",
)
openai_api_key = st.secrets["OPENAI_API_KEY"]
# Function to read PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split documents
def text_split(raw_text, chunk_size=1000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(raw_text)
    return docs

# Function to initialize vector store
def initialize_vector_store(texts):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
 
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi there. How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'initialPageLoad' not in st.session_state:
    st.session_state['initialPageLoad'] = False


if 'prevent_loading' not in st.session_state:
    st.session_state['prevent_loading'] = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'File Upload'

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Setup for chat interface and styles
typing_animation_json = render_animation()  # Ensure this function is defined elsewhere

# Hide default Streamlit styles
hide_st_style = """ 
    <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
    </style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)



# Sidebar for navigation
with st.sidebar:
    st.title("Page Views")
    
    # Show "Chat" option only after the file is uploaded
    if st.session_state.file_uploaded:
        st.session_state.page_selection = st.radio("", ["Upload File", "ChatBot"], index=1)
    else:
        st.session_state.page_selection = st.radio("", ["Upload File"], index=0)

# Content based on sidebar selection
if st.session_state.page_selection == "Upload File":
    st.header("Upload Your File 🗃️")
    docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])
    submit = st.button("Upload")

    # File upload logic
    if docx_file is not None and submit:
      
        st.write("Please Wait file is uploading")

        raw_text = ""
        if docx_file.type == "text/plain":
            raw_text = str(docx_file.read(), "utf-8")
        elif docx_file.type == "application/pdf":
            raw_text = read_pdf(docx_file)

        if raw_text:
            splited_text = text_split(raw_text)
            try:
                st.session_state.vector_store = initialize_vector_store(splited_text)
                st.success('File uploaded and processed successfully! Vector store initialized.')
                
                # Set file uploaded flag and navigate to Chat
                st.session_state.file_uploaded = True
                st.session_state.page_selection = "ChatBot"
                st.rerun()  # Re-run the app to update UI and show Chat
                
            except Exception as e:
                st.error(f'Error initializing vector store: {e}')
        else:
            st.error('Failed to process the file.')

elif st.session_state.page_selection == "ChatBot":
    st.header("Chat with the Bot 🤖")

    # Container for chat history
    response_container = st.container()
    textcontainer = st.container()

    # Display previous chat history
    with textcontainer:
        st.session_state.initialPageLoad = False
        query = st.chat_input(placeholder="Say something ... ", key="input")
        if query and query != "Menu":
            conversation_string = get_conversation_string()
            with st_lottie_spinner(typing_animation_json, height=50, width=50, speed=3, reverse=True):
                if st.session_state.vector_store:
                    try:
                        # Retrieve relevant documents from vector store
                        relevant_chunks = st.session_state.vector_store.similarity_search(query, k=3)

                        if relevant_chunks:
                            chunk_texts = [doc.page_content.strip() for doc in relevant_chunks if len(doc.page_content) > 50]
                            combined_text = "\n".join(chunk_texts[:3])

                            # Modify prompt to ensure the chatbot only uses document content
                            prompt = f"You are an assistant that only answers questions based on the information from the provided document. Do not use any external knowledge. Answer the question only using the following relevant information and the answer should be precise and should not confuse the user:\n\nRelevant Information:\n{combined_text}\n\nQuestion: {query}\nAnswer:"

                           
                            llm = OpenAI(api_key=openai_api_key)
                            response = llm.generate(prompts=[prompt], temperature=0.7, max_tokens=200)
                            generated_text = response.generations[0][0].text

                            st.session_state.requests.append(query)
                            st.session_state.responses.append(generated_text)
                        else:
                            st.error("No relevant chunks found for your query.")
                    except Exception as e:
                        st.error(f'Error during similarity search or generation: {e}')
                else:
                    st.error('Vector store is not initialized. Please upload a file first.')

            st.session_state.prevent_loading = True

    # Display chat history
    with response_container:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.session_state.initialPageLoad = False
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                response = f"<div style='font-size:0.875rem;line-height:1.75;white-space:normal;'>{st.session_state['responses'][i]}</div>"
                message(response, allow_html=True, key=str(i), logo=('https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/SS512X512.png'))
                if i < len(st.session_state['requests']):
                    request = f"<meta name='viewport' content='width=device-width, initial-scale=1.0'><div style='font-size:.875rem'>{st.session_state['requests'][i]}</div>"
                    message(request, allow_html=True, is_user=True, key=str(i)+'_user', logo='https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/generic-user-icon-13.jpg')
