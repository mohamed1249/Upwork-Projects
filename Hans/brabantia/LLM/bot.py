# import tensorflow as tf
# from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain.chat_models import ChatOpenAI
# import os
# import gradio as gr

# os.environ["OPENAI_API_KEY"] = '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'

# def construct_index(directory_path='../data/'):
#     max_input_size = 4096
#     num_outputs = 1028
#     max_chunk_overlap = 20
#     chunk_size_limit = 600

#     prompt_helper = PromptHelper(max_input_size,num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="EleutherAI/gpt-neo-2.7B", max_tokens=num_outputs))

#     documents = SimpleDirectoryReader(directory_path).load_data()

#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#     index.save_to_disk('index.json')

#     # Export the LLM model as a TensorFlow SavedModel
#     # model = index.llm_predictor.()
#     tf.saved_model.save(index, 'saved_model')
    

# construct_index()


# def build_the_bot():

#     global index

#     index = GPTSimpleVectorIndex.load_from_disk('index.json')

# def chat_(chat_history, input):
  
#   bot_response = index.query(user_input)
#   response = ""
#   for letter in ''.join(bot_response.response):
#       response += letter + ""
#       yield chat_history + [(user_input, response)]

# with gr.Blocks() as demo:
#     gr.Markdown('# Zoeken op de website')
#     with gr.Tab("Amfi"):
#           build_the_bot()
#           chatbot = gr.Chatbot()
#           message = gr.Textbox(placeholder='Type hier je vraag',label='Waar ben je naar op zoek')
#           message.submit(chat_, [chatbot, message], chatbot)

# demo.queue().launch(debug = True, share=True)

import os
os.environ["OPENAI_API_KEY"] = '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'

import gradio as gr

# %%
from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='./storage')
# load index
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

chat_history_ = ''

def chat_(chat_history, question):
  global chat_history_

  if chat_history_:

    try:
            input = f"""Based on this chat history that you and I have been discussing:
                            '{chat_history_[int(-4096*4/2):]}'
                            Answer this question:
                            {question}
                            Providing a detailed response based on your existing knowledge and understanding,
                            Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                            Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                           """
            print(input)
        
    except:
            input = f"""Based on this chat history that you and I have been discussing:
                            '{chat_history_}'
                            Answer this question:
                            {question}
                            Providing a detailed response based on your existing knowledge and understanding,
                            Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                            Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                           """
            print(input)
  else:
        input = f"""Answer this question:
                            {question}
                            Providing a detailed response based on your existing knowledge and understanding,
                            Don't generate fictional or made-up information, Stick to factual and relevant details from the provided knowledge,
                            Don't search the internet for information or include any internet website URLs that are not in the existing knowledge 
                           """
  
  bot_response = query_engine.query(input).response

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

                                '''
      bot_response = bot_response.strip('\n') + ' ' + query_engine.query(complete_command).response.strip('\n')
    
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

                                    '''
      bot_response = bot_response.strip('\n') + ' ' + query_engine.query(complete_command).response.strip('\n')
      
  response = ""
  chat_history_ += f"My question: {question} \n Your answer: {bot_response}"

  for letter in ''.join(bot_response):
      response += letter + ""
      yield chat_history + [(question, response)]




with gr.Blocks() as demo:
    gr.Markdown('# 1')
    with gr.Tab("2"):

          chatbot = gr.Chatbot()
          message = gr.Textbox(placeholder='3', label='4')
          message.submit(chat_, [chatbot, message], chatbot)

demo.queue().launch(debug = True, share=True)