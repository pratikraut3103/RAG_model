import os
from deep_translator import GoogleTranslator
import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

class RAG_pipeline:
    def __init__(self):

        with open('creds.json') as data_file:
            data = json.load(data_file)
        os.environ["GOOGLE_API_KEY"] = data["GOOGLE_API_KEY"]
        self.groq_api_key = data["groq_api_key"]
        self.data = None
        self.translated_data = None
        self.cleaned_chunks = None
        self.vectors = None
        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name="Llama3-8b-8192")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def translate_text(self, text,source,target):
        translated_text = GoogleTranslator(source=source, target=target).translate(text)
        return translated_text

    def read_pdf(self,path):
        pdf_loader = PyMuPDFLoader(path)
        self.data = pdf_loader.load()

    def get_remove_stopwords(self,chunk):
        stop_words = list(set(stopwords.words('german')))
        tokens = word_tokenize(chunk)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text


    def chunking_and_preprocessing(self,seperators, chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=seperators,
            chunk_size=chunk_size,
            chunk_overlap=200,
        )
        data_splitted = text_splitter.split_documents(self.data)
        self.cleaned_chunks = [self.translate_text(text.page_content,"de","en") for text in data_splitted]

    def generating_embeddings_storing(self,folder_name):
        self.vectors = FAISS.from_texts(self.cleaned_chunks, self.embeddings)
        self.vectors.save_local(folder_name)

    def retrive_answer(self, embedding_folder):

        prompt_template = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}

        """)
        prompt1 = self.translate_text(input("Geben Sie Ihre Frage aus den Dokumenten ein: "),"en","de")
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        vector_index = FAISS.load_local(embedding_folder, self.embeddings, allow_dangerous_deserialization=True)
        retriever = vector_index.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})
        print(self.translate_text(response["answer"],"en","de"))

if __name__ == '__main__':
    question = input("Have you already extracted data from pdf? (y/n)")
    if question == "y":
        input_folder = input("Which pdf would you like to query? \n 1. Basispaket+WeitBlick.pdf, 2. pa_d_1006_iii_11_211.pdf. \n Press 1 or 2")
        if input_folder == "1":
            RAG_pipeline().retrive_answer(embedding_folder="Basispaket+WeitBlick")
        if input_folder == "2":
            RAG_pipeline().retrive_answer(embedding_folder="pa_d_1006_iii_11_211")

    else:
        input_folder = input(
            "Which pdf would you like generate data? \n 1. Basispaket+WeitBlick.pdf, 2. pa_d_1006_iii_11_211.pdf. \n Press 1 or 2")
        my_rag = RAG_pipeline()
        if input_folder == "1":
            my_rag.read_pdf("Basispaket+WeitBlick.pdf")
            my_rag.chunking_and_preprocessing(seperators=["\n\n","\n"],chunk_size=1000)
            my_rag.generating_embeddings_storing("Basispaket+WeitBlick")

        if input_folder == "2":
            my_rag.read_pdf("pa_d_1006_iii_11_211.pdf")
            my_rag.chunking_and_preprocessing(seperators=["\n\n","\n"],chunk_size=1000)
            my_rag.generating_embeddings_storing("pa_d_1006_iii_11_211")







