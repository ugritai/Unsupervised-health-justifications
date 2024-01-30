import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import nest_asyncio
from transformers import BitsAndBytesConfig
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import pandas as pd
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain import hub
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs).replace('\n',' ')

def format_results_to_dataframe(results:str,nutrient:str,url:str,context:str)->pd.DataFrame:
  df = pd.DataFrame(results)
  df['nutrient'] = nutrient
  df['url'] = url
  df['context'] = context

  return df


def get_justifications(nutrient_key:str,url:str,question)->list:
  text = WebBaseLoader(url).load()
  # splitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 50,
      length_function = len
  )
  text_chunks = text_splitter.transform_documents(text)

  # Define the path to the pre-trained model you want to use
  modelPath = "sentence-transformers/all-MiniLM-l6-v2"

  # Create a dictionary with model configuration options, specifying to use the CPU for computations
  model_kwargs = {'device':'cuda'}

  # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
  encode_kwargs = {'normalize_embeddings': False}

  # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
  embeddings = HuggingFaceEmbeddings(
      model_name=modelPath,     # Provide the pre-trained model's path
      model_kwargs=model_kwargs, # Pass the model configuration options
      encode_kwargs=encode_kwargs # Pass the encoding options
  )
  db = FAISS.from_documents(text_chunks, embeddings)
  question=question
  # Initialize base retriever with FaissVectorStore
  retriever = db.as_retriever(search_kwargs={"k": 4})

  docs = retriever.get_relevant_documents(question)

  context = format_docs(docs)
  # prompt template
  template = """Human: You are an assistant for question-answering tasks.
   Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use two sentences maximum and keep the answer concise.
  Question: {question}
  Context:{context}
  Answer:"""
  rag_prompt_custom = PromptTemplate.from_template(template)

  print(
      rag_prompt_custom.invoke(
          {"context": context, "question": question}
      ).to_string()
  )
  # Callbacks support token-wise streaming
  callbacks = [StreamingStdOutCallbackHandler()]

  # Chain
  chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)
  # Run
  results = chain({"input_documents": docs, "question": question}, return_only_outputs=False)
  return results,context






# uploading the model to the colab
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.1',
    trust_remote_code=True,
    torch_dtype=bfloat16
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")



# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=64,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


llm = HuggingFacePipeline(pipeline=generate_text)
template = """Human: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
 If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise.
Question: {question}
Context:{context}
Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)
# Chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

callbacks = [StreamingStdOutCallbackHandler()]

nutrients={'Calcium':['https://www.eufic.org/en/vitamins-and-minerals/article/calcium-foods-functions-how-much-do-you-need-more',
         'https://ods.od.nih.gov/factsheets/Calcium-HealthProfessional/',
                      'https://en.wikipedia.org/wiki/Calcium#Biological_and_pathological_role']}


df=pd.DataFrame(columns=['input_documents' ,	'question','context', 	'output_text', 	'nutrient','url'])
for nutrient_key in nutrients.keys():
  questions=["What are the drawbacks of {} in health?".format(nutrient_key),
             "What are the effect of {} deficit in health?".format(nutrient_key),
             "What is the role of {} in health?".format(nutrient_key),
             "What are the effect of high {} in health?".format(nutrient_key),
             "What are the effect of {} excess in health?".format(nutrient_key),
             "What are the benefits of {} in health?".format(nutrient_key),
             ]
  for question in questions:
    urls = nutrients[nutrient_key]
    for url in urls:
      print(nutrient_key,url)
      results,context = get_justifications(nutrient_key,url,question)
      df_result = format_results_to_dataframe(results,nutrient_key,url,context)
      df = pd.concat([df, df_result], ignore_index=True)
df.to_excel('final_mistral2.xlsx')