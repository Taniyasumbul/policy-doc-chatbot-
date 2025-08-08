from flask import Flask, request, jsonify
from chatbot import get_answer_from_documents


app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "Query is missing"}), 400

    answer = get_answer_from_documents(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
