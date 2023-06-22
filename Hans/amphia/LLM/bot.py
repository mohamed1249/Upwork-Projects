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

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader,LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr

max_input_size = 4096
num_outputs = 4096
max_chunk_overlap = 20
chunk_size_limit = 900
directory_path='../data/'

# prompt_helper = PromptHelper(max_input_size,num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

# llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="davinci-instruct-beta", max_tokens=num_outputs))

# documents = SimpleDirectoryReader(directory_path).load_data()
# index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# # # %%
# index.storage_context.persist()

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
      input = 'Based on this chat history we\'ve been discussing:\n'+ chat_history_[-4096*4/2:] +'\n Answer this question in a summrized way from only the documments you\'ve learned and never genrate fake or made up information:\n' + question
      print(input)
      print(len(input)/4)
    
    except:
        input = 'Based on this chat history we\'ve been discussing:\n'+ chat_history_ +'\n Answer this question in a summrized way from only the documments you\'ve learned and never genrate fake or made up information:\n' + question
        print(input)
        print(len(input)/4)
  else:
    input = question
  
  bot_response = query_engine.query(input).response

  while not bot_response.strip().endswith('.'):

    try:
      complete_command = f'I\'ve sent you this question:\n"{question}"\n but you provided me with this uncompleted answer:\n "{bot_response[-4096*4/2:]}"\n Please write just the rest of your uncompleted answer to be merged to your uncompleted answer.'
      bot_response = bot_response.strip('\n') + ' ' + query_engine.query(complete_command).response.strip('\n')
    
    except:
      complete_command = f'I\'ve sent you this question:\n"{question}"\n but you provided me with this uncompleted answer:\n "{bot_response}"\n Please write just the rest of your uncompleted answer to be merged to your uncompleted answer.'
      bot_response = bot_response.strip('\n') + ' ' + query_engine.query(complete_command).response.strip('\n')
      


  response = ""
  chat_history_ += f"Question: {question} \n Answer: {bot_response}"

  for letter in ''.join(bot_response):
      response += letter + ""
      yield chat_history + [(question, response)]




with gr.Blocks() as demo:
    gr.Markdown('# Zoeken op de website')
    with gr.Tab("Amfi"):

          chatbot = gr.Chatbot()
          message = gr.Textbox(placeholder='Type hier je vraag',label='Waar ben je naar op zoek')
          message.submit(chat_, [chatbot, message], chatbot)

demo.queue().launch(debug = True, share=True, server_port=8000)