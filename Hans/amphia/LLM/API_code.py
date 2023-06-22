from flask import Flask, request, abort
from flask_cors import CORS
import os
from llama_index import StorageContext, load_index_from_storage

os.environ["OPENAI_API_KEY"] = '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'
storage_context = StorageContext.from_defaults(persist_dir='./storage')
# load index
index = load_index_from_storage(storage_context).as_query_engine()

app = Flask(__name__)
cors = CORS(app)




@app.route("/amphia", methods=['GET', 'POST'])
def chat():
    api_key = request.headers.get("API-KEY")

    API_KEY = "amphia-22768"
    if api_key != API_KEY:
        abort(401, "Invalid API key")

    request_json = request.get_json()
    user_input = request_json['message']

    bot_response = index.query(user_input)

    # Process user input and generate response
    response = {'response': bot_response.response}

    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
