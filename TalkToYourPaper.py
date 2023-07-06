import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import ArxivLoader, PyPDFLoader, UnstructuredFileLoader, OnlinePDFLoader, UnstructuredPDFLoader
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent, Tool
from langchain.agents import AgentType
from langchain import SerpAPIWrapper


def generate_response(pdf_url, openai_api_key, query_paper, query_general):
    if pdf_url is not None:
        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        # Select LLM Model
        llm = ChatOpenAI(temperature = 0.0)
        # Load YouTube video transcript from the given url
        loader = OnlinePDFLoader(pdf_url)
        docs = loader.load()
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa_system = RetrievalQA.from_chain_type(llm=llm, 
                                                chain_type="stuff", 
                                                retriever=retriever, 
                                                verbose=False)
        # Initialize the SerpAPIWrapper for search functionality
        search = SerpAPIWrapper()
        tools = [
    		Tool(
        		name="Search",
        		func=search.run,
        		description="Useful when you need to answer questions about current events or past events related to the scientic paper, which are not explicit contents of the paper."
    		),
    		Tool(
        	name="QA system",
        	func=qa_system.run,
        	description="Useful when you need to answer questions from the contents of the given scientific paper."
    		),]
        agent = initialize_agent(
    		tools,
    		llm,
    		agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    		handle_parsing_errors=True,
    		verbose=False)
        return qa_system.run(query_paper), agent.run(query_general)





# Page title
st.set_page_config(page_title='🎖️🔗 Talk to your scientific paper')
st.title('🎖️🔗 Talk to your scientific paper')

# URL Text
pdf_url = st.text_input('Enter your paper URL:', placeholder = 'Paper URL.')
# Query paper
query_paper = st.text_input('As a question specific to your paper:', placeholder = 'Please provide a short summary.', disabled=not pdf_url)
# Query general
query_general = st.text_input('As a general question related to your paper:', placeholder = 'Please provide a short summary.', disabled=not (pdf_url and query_paper))

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (pdf_url and query_paper and query_general))
    submitted = st.form_submit_button('Submit', disabled=not(pdf_url and query_paper and query_general))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(pdf_url, openai_api_key, query_paper, query_general)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
