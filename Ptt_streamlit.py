import faiss_sumpy as fs
import streamlit as st
import os
import tiktoken
import warnings
import requests
from dotenv import load_dotenv
import textract
import re
import tempfile
from ocr_summarizer import summarize_paragraph
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

FASTAPI_URL1 = os.getenv('FASTAPI_URL1')
FASTAPI_URL2 = os.getenv('FASTAPI_URL2')
FASTAPI_URL3 = os.getenv('FASTAPI_URL3')

def summarize_PDF_file(pdf_file, model, input_title):

    text_summaries = []

    if (pdf_file is not None):
        st.write("PDF 문서를 요약 중입니다. 잠시만 기다려 주세요.")
        # PDF 파일을 바이트 스트림으로 읽기
        file_bytes = pdf_file.read()
        url = f"{FASTAPI_URL3}/upload"
        headers = {"Content-Type": "application/pdf", "titles": input_title}
        response = requests.post(url, data=file_bytes, headers=headers)
        print("pdf sended")
        if response.status_code == 200:
            texts = response.json().get("texts", "")
            st.write("파일이 성공적으로 전송되었습니다.")
            # text = texts.decode('utf-8')
            # 스플리터 지정
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\\n\\n",  # 분할 기준
                chunk_size=2000,   # 청크 사이즈
                chunk_overlap=500, # 중첩 사이즈
            )
            split_texts = text_splitter.split_text(texts)
            st.write(len(split_texts), "단락이 있음.")
            
            # 결과 출력
            st.write("단락별 텍스트:")
            for i, paragraph in enumerate(split_texts):
                st.write(f"단락 {i+1}:")
                st.write(paragraph)
                st.write("---")

            # summaries = []
            # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            #     futures = {executor.submit(summarize_paragraph, paragraph): i for i, paragraph in enumerate(paragraphs_to_summarize)}
            #     for future in concurrent.futures.as_completed(futures):
            #         try:
            #             summary = future.result()
            #             summaries.append(summary)
            #             st.write(f"요약된 단락 {len(summaries)}:")
            #             st.write(summary)
            #             st.write("---")
            #         except Exception as e:
            #             st.write(f"Error during summarization: {e}")
        if model == "hf":
            print("HuggingFace 구상 중")
            # documents = fs.data_load(texts)
            # text_summary = fs.summarize(documents)
            # st.write("요약 텍스트: ", text_summary)
        else:
            print("구현중")
    else:
        st.write("PDF 파일를 업로드하세요.")

    # st.write(text_summary)

            

# ------------- 사이드바 화면 구성 -----------------------
st.sidebar.title('Menu')

# ------------- 메인 화면 구성 --------------------------  
st.title('Paragraph Extraction and Summarization from PDF')

st.header("요약 모델 설정 ")

st.write("요약 모델을 선택하세요.")

input_title = st.text_input("Paper title")
st.write("논문의 제목을 입력하세요.")

radio_selected_model = st.radio("PDF 문서 요약 모델", ["OpenAI","HuggingFace"], index=1, horizontal=True)

upload_file = st.file_uploader("PDF 파일를 업로드하세요.", type="pdf")

if radio_selected_model == "HuggingFace":
    model = "hf"
else:
    st.write("OpenAI 구상 중")

clicked_sum_model = st.button('PDF 문서 요약')

if clicked_sum_model:
    summarize_PDF_file(upload_file, model, input_title)