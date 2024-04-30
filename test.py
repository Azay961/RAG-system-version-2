from flask import Flask, render_template, request


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
                                
    return render_template("search.html", api_key=api_key, doc=file)


def create_embedding_store(file, key):
    


if __name__ == "__main__":
    app.run(debug=True)