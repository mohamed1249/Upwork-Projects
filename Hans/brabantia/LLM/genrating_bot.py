import os
os.environ["OPENAI_API_KEY"] = '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'

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader,LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

max_input_size = 4096
num_outputs = 4096
max_chunk_overlap = 20
chunk_size_limit = 900
directory_path='../data/'

prompt_helper = PromptHelper(max_input_size,num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="davinci-instruct-beta", max_tokens=num_outputs))

documents = SimpleDirectoryReader(directory_path).load_data()
index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# # %%
index.storage_context.persist()