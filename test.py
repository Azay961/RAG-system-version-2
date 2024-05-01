from flask import Flask, render_template, request

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import NLTKTextSplitters

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


app = Flask(__name__)
api_key = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_in_docs():
    global api_key = request.form["api_key"]

    url = request.form.get("url")
    if not url:
        file = request.form.get("file")
    if not file:
        return render_template("index.html", error="Select File or Input url first")
                                
    files = read_file(file=file)
    db_chroma = create_embedding_store(files)

    query = request.form.get("user_input")
    answers = rag_chain(db=db_chroma, query=query)

    return render_template("search.html", answers=answers)


def read_file(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    return pages

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                               model="models/embeddings-001")
def create_embedding_store(files):
    text_spltter = NLTKTextSplitters(chunk_size=500, chunk_overlap=100)
    chunks = text_spltter.split_documents(files)

    db = Chroma.from_documents(chunks, embedding_model, "/chroma")

    return db

def rag_chain(db, query):
    chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
    ])

    chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, 
                                   model="gemini-1.5-pro-latest")

    output_parser = StrOutputParser()

    retriever = db.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
    )

    answers = rag_chain.invoke(query)
    return answers
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    app.run(debug=True)