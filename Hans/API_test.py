from flask import Flask, request, abort
from flask_cors import CORS,cross_origin
import os
from llama_index import StorageContext, load_index_from_storage

os.environ["OPENAI_API_KEY"] = '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'
# '/home/mohameedmtmn/amphia/amphia/LLM/storage'
# '/home/mohameedmtmn/Miljuschka/LLM/storage'
data = {
    'amphia' : {
        'model' : load_index_from_storage(StorageContext.from_defaults(persist_dir=r"/home/mohameedmtmn/Miljuschka/LLM/storage")).as_query_engine(),
        'API_KEY' : "amphia-22768"
    },
    'miljuschka' : {
        'model' : load_index_from_storage(StorageContext.from_defaults(persist_dir=r'/home/mohameedmtmn/Miljuschka/LLM/storage')).as_query_engine(),
        'API_KEY' : "miljuschka-23068"
    },
    'brabantia' : {
        'model' : load_index_from_storage(StorageContext.from_defaults(persist_dir=r"/home/mohameedmtmn/brabantia/LLM/storage")).as_query_engine(),
        'API_KEY' : "brabantia-232006"
    },
}


app = Flask(__name__)
app.config['CORS_HEADERS'] = ['Content-Type', 'API-KEY']
cors = CORS(app,origins=['https://sayhaito.com','http://sayhaito.com'])


@app.route("/amphia", methods=['GET', 'POST'])
@cross_origin(origin=['https://sayhaito.com','http://sayhaito.com'],headers=['Content-Type','API-KEY'])
def amphia_chat():
        global data

        api_key = request.headers.get("API-KEY")
        
        if api_key != data["amphia"]['API_KEY']:
            abort(401, "Invalid API key")

        request_json = request.get_json()
        question = request_json['message']

        
        user_input = f"""Answer this question:
                        {question}
                        Providing a response that doesn't exceed 4096 tokens based on your existing knowledge and understanding,
                        Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                        Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                        Use Dutch as your default language for your responses and answers, if the user asks or requests for a response in another language use the indicated language accordingly.
                        """
        print(user_input)
        
        bot_response = data["amphia"]['model'].query(user_input).response
        response = {'response': bot_response}

        return response


@app.route("/miljuschka", methods=['GET',"POST"])
@cross_origin(origin=['https://sayhaito.com','http://sayhaito.com'],headers=['Content-Type','API-KEY'])
def miljuschka_chat():
    try:
        global data

        api_key = request.headers.get("API-KEY")
        
        if api_key != data["miljuschka"]['API_KEY']:
            abort(401, "Invalid API key")

        request_json = request.get_json()
        question = request_json['message']
            

        user_input = f"""Answer this question:
                        {question}
                        Providing a response that doesn't exceed 4096 tokens based on your existing knowledge and understanding,
                        Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                        Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                        Use Dutch as your default language for your responses and answers, if the user asks or requests for a response in another language use the indicated language accordingly.
                        Answer the question and provide up to three URLs as sources to support your answer, Format the answer as follows: 
                        'Source 1: [URL 1], Source 2: [URL 2], Source 3: [URL 3]'.
                        The URLs should come from credible sources within the current existing knowledge and should pertain to the topic discussed, They should not be taken from the internet.
                        """
        print(user_input)
        
        bot_response = data["miljuschka"]['model'].query(user_input).response

        while not bot_response.strip().endswith('.'):

            try:
                complete_command = f'''I\'ve asked you this question:
                                    "{question}"
                                    But you answered me with this uncompleted answer:
                                    "{bot_response[int(-4096*4/2):]}"
                                    Please write just the rest of your uncompleted answer to be merged to your uncompleted answer, and make sure you follow these instruction:
                                        1- Ensure that all responses are based solely on the content learned from the available sources.

                                        2- Avoid generating fictional or made-up information. Stick to factual and relevant details from the provided knowledge.

                                        3- Do not search the internet for additional information or references. Rely exclusively on the pre-existing knowledge within the model.

                                        4- Use Dutch as the default language for responses unless specifically requested otherwise by the user. However, if the user asks for another language, feel free to respond accordingly.

                                        5- Answer the question and provide up to three URLs as sources to support your answer,the URLs should come from sources within the current existing knowlage and should pertain to the topic discussed, they should not be taken from the internet.
                                '''
                bot_response = bot_response.strip('\n') + ' ' + data["miljuschka"]['model'].query(complete_command).response.strip('\n')
            
            except:
                complete_command = f'''I\'ve asked you this question:
                                    "{question}"
                                    But you answered me with this uncompleted answer:
                                    "{bot_response}"
                                    Please write just the rest of your uncompleted answer to be merged to your uncompleted answer, and make sure you follow these instruction:
                                        1- Ensure that all responses are based solely on the content learned from the available sources.

                                        2- Avoid generating fictional or made-up information. Stick to factual and relevant details from the provided knowledge.

                                        3- Do not search the internet for additional information or references. Rely exclusively on the pre-existing knowledge within the model.

                                        4- Use Dutch as the default language for responses unless specifically requested otherwise by the user. However, if the user asks for another language, feel free to respond accordingly.

                                        5- Answer the question and provide up to three URLs as sources to support your answer,the URLs should come from sources within the current existing knowlage and should pertain to the topic discussed, they should not be taken from the internet.
                                '''
                bot_response = bot_response.strip('\n') + ' ' + data["miljuschka"]['model'].query(complete_command).response.strip('\n')
    
        # Process user input and generate response
        response = {'response': bot_response}

        return response
    
    except Exception as e:
        return {'response': 'ampiha\n'+e }


@app.route("/brabantia", methods=['GET',"POST"])
@cross_origin(origin=['https://sayhaito.com','http://sayhaito.com'],headers=['Content-Type','API-KEY'])
def brabantia_chat():
    try:
        global data

        api_key = request.headers.get("API-KEY")
        
        if api_key != data["brabantia"]['API_KEY']:
            abort(401, "Invalid API key")

        request_json = request.get_json()
        question = request_json['message']


        user_input = f"""Answer this question:
                        {question}
                        Providing a response that doesn't exceed 4096 tokens based on your existing knowledge and understanding,
                        Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                        Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                        Use Dutch as your default language for your responses and answers, if the user asks or requests for a response in another language use the indicated language accordingly.
                        Answer the question and provide up to three URLs as sources to support your answer, Format the answer as follows: 
                        'Source 1: [URL 1], Source 2: [URL 2], Source 3: [URL 3]'.
                        The URLs should come from credible sources within the current existing knowledge and should pertain to the topic discussed, They should not be taken from the internet.
                        """
        print(user_input)

        bot_response = data["brabantia"]['model'].query(user_input).response

        while not bot_response.strip().endswith('.'):

            try:
                complete_command = f'''I\'ve asked you this question:
                                    "{question}"
                                    But you answered me with this uncompleted answer:
                                    "{bot_response[int(-4096*4/2):]}"
                                    Please write just the rest of your uncompleted answer to be merged to your uncompleted answer, and make sure you follow these instruction:
                                        1- Ensure that all responses are based solely on the content learned from the available sources.

                                        2- Avoid generating fictional or made-up information. Stick to factual and relevant details from the provided knowledge.

                                        3- Do not search the internet for additional information or references. Rely exclusively on the pre-existing knowledge within the model.

                                        4- Use Dutch as the default language for responses unless specifically requested otherwise by the user. However, if the user asks for another language, feel free to respond accordingly.

                                        5- Answer the question and provide up to three URLs as sources to support your answer,the URLs should come from sources within the current existing knowlage and should pertain to the topic discussed, they should not be taken from the internet.
                                '''
                bot_response = bot_response.strip('\n') + ' ' + data["brabantia"]['model'].query(complete_command).response.strip('\n')
            
            except:
                complete_command = f'''I\'ve asked you this question:
                                    "{question}"
                                    But you answered me with this uncompleted answer:
                                    "{bot_response}"
                                    Please write just the rest of your uncompleted answer to be merged to your uncompleted answer, and make sure you follow these instruction:
                                        1- Ensure that all responses are based solely on the content learned from the available sources.

                                        2- Avoid generating fictional or made-up information. Stick to factual and relevant details from the provided knowledge.

                                        3- Do not search the internet for additional information or references. Rely exclusively on the pre-existing knowledge within the model.

                                        4- Use Dutch as the default language for responses unless specifically requested otherwise by the user. However, if the user asks for another language, feel free to respond accordingly.

                                        5- Answer the question and provide up to three URLs as sources to support your answer,the URLs should come from sources within the current existing knowlage and should pertain to the topic discussed, they should not be taken from the internet.
                                '''
                bot_response = bot_response.strip('\n') + ' ' + data["brabantia"]['model'].query(complete_command).response.strip('\n')
        # Process user input and generate response
        response = {'response': bot_response}

        return response
    
    except Exception as e:
        return {'response': 'ampiha\n'+e }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
