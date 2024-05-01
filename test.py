from flask import Flask, render_template, request

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import NLTKTextSplitters

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("search.html")

@app.route("/search", methods=["POST"])
def search_in_docs():
    api_key = request.form["api_key"]

    url = request.form.get("url")
    if not url:
        file = request.form.get("file")
    if not file:
        return render_template("index.html", error="Select File or Input url first")
                                
    files = read_file(file=file)
    db_chroma = create_embedding_store(files)

    return render_template("search.html", api_key=api_key, doc=file)


def read_file(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    return pages

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key,
                                               model="models/embeddings-001")
def create_embedding_store(files, key):
    text_spltter = NLTKTextSplitters(chunk_size=500, chunk_overlap=100)
    chunks = text_spltter.split_documents(files)

    db = Chroma.from_documents(chunks, embedding_model, "/chroma")


if __name__ == "__main__":
    app.run(debug=True)