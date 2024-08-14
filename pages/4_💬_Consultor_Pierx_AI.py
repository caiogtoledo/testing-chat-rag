import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

st.set_page_config(page_title="PierX AI", page_icon="📄")
st.image("https://dev.pierxinovacao.com.br/assets/img/logo.svg", width=120)
st.header('Fale com o Consultor PierX AI')
st.write('Pergunte o que quiser sobre o edital "Conhecimento Brasil" da FINEP: ')

class CustomDataChatbot:

    def __init__(self):
        self.openai_model = utils.configure_openai()
        self.directory = "tmp"
        # self.save_files()

    # def save_files(self):
    #     tmp_dir = "tmp"
        
    #     if not os.path.exists(tmp_dir):
    #         print(f"A pasta '{tmp_dir}' não existe.")
    #         return
        
    #     for file_name in os.listdir(tmp_dir):
    #         if file_name.endswith(".pdf"):
    #             file_path = os.path.join(tmp_dir, file_name)
    #             with open(file_path, 'rb') as file:
    #                 self.save_file(file)

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # file_path = os.path.join(folder, file.name)
        with open(file.name, 'wb') as f:
            f.write(file.read())
        return file.name

    @st.spinner('Processando...')
    def setup_qa_chain(self, uploaded_files):
        if len(uploaded_files) <= 0:
            files = [file_name for file_name in os.listdir(self.directory) if file_name.endswith(".pdf")]
        else:
            files = uploaded_files

        # Load documents
        docs = []
        for file in files:
            if len(uploaded_files) <= 0:
                file_path = f"./{self.directory}/{file}"
            else:
                file_path = self.save_file(file)
            print(file_path)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        # User Inputs
        # uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        # if not uploaded_files:
        #     st.error("Please upload PDF documents to continue!")
        #     st.stop()

        user_query = st.chat_input(placeholder="O que deseja saber sobre o edital?")

        if user_query:
            uploaded_files = []
            qa_chain = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = qa_chain.invoke(
                    {"question":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})

                # to show references
                for idx, doc in enumerate(result['source_documents'],1):
                    filename = os.path.basename(doc.metadata['source'])
                    page_num = doc.metadata['page']
                    ref_title = f":blue[Referência {idx}: *{filename} - pag.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()