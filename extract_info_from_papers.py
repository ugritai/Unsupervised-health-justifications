#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   extract_info_from_papers.py
@Time    :   2024/01/26 09:35:16
@Author  :   Bart Ortiz 
@Version :   1.0
@Contact :   bortiz@ugr.es
@License :   CC-BY-SA or GPL3
@Desc    :   None
'''

# import langchain library and the web scrapping library
import langchain
from bs4 import BeautifulSoup
import requests

import pandas as pd
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]

def clean_document_type(document):
    texto = document.page_content
    texto = texto.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # remove extra spaces
    texto = ' '.join(texto.split())
    document.page_content = texto
    return document


url = 'https://www.who.int/news-room/fact-sheets/detail/salt-reduction'
url= 'https://econtent.hogrefe.com/doi/epdf/10.1024/0300-9831/a000224'
text = WebBaseLoader(url).load()
# splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len
)
text_chunks = text_splitter.transform_documents(text)
# remove multiple spaces and \n using langchain
for chunk in text_chunks:
    chunk = clean_document_type(chunk)


text_chunks