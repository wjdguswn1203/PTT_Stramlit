# 📂 PaperTechTrend Stramlit 프로젝트 소개
- 기존 PaperTechTrend을 develop 하기 전에 OCR로 PDF를 인식하여 해당 텍스트를 단락별로 나누어 요약. 요약시간과 키워드간의 매칭도를 통해 요약 모델을 비교하는 프로젝트

<br/>
<br/>

## 🎥 구현 화면
## 메인화면
![image](https://github.com/user-attachments/assets/3b330bae-9aca-4b67-bafe-eb0b7bf00aea)


## PDF 업로드 화면
### 업로드 전
![image](https://github.com/user-attachments/assets/3502b4f0-58c0-4e89-92e0-7b02fca3e659)

### OCR 추출 후
![image](https://github.com/user-attachments/assets/fb200090-5605-4326-a17b-56429ac34ad2)


## 키워드 추출 결과
![image](https://github.com/user-attachments/assets/e9d12090-fa8b-4878-8f5b-eac2dd00f05a)


## 요약
### 단락 나누기
![image](https://github.com/user-attachments/assets/9655abca-5c76-46c6-9d55-adf1827d1756)

## 단락별 요약
![image](https://github.com/user-attachments/assets/c2a5ac59-a633-4857-a12c-b29846821413)

## 최종 요약
![image](https://github.com/user-attachments/assets/8ab96fed-8ee7-4d09-8d85-3319f17d5c11)


## 결과 그래프
### 선택창
![Cap 2024-06-25 15-41-36-036](https://github.com/user-attachments/assets/b887d6fd-101a-4290-87c2-625a38654e3d)

### 걸린 시간과 키워드 매칭률 그래프
![Cap 2024-06-26 11-15-40-997](https://github.com/user-attachments/assets/04c4a123-17ac-4495-b74d-8a03a74766a5)

### 요약 시간 그래프
![Cap 2024-06-26 11-02-52-946](https://github.com/user-attachments/assets/19290ee3-935f-42a0-8ac6-10d0f00523b9)

### 키워드 매칭 그래프
![Cap 2024-06-26 11-12-32-057](https://github.com/user-attachments/assets/c4f61a93-2595-4103-8a37-98226643ac35)


### 
## 
<br/>
<br/>



# 👥 팀원 소개

| <img width="250" alt="hj" src="https://github.com/user-attachments/assets/c0af7daa-f81b-4527-b62b-f9ee8d23e311"> | <img width="250" alt="yj" src="https://github.com/user-attachments/assets/bee1516f-d25d-46af-8cee-2771a4d9c917"> | <img width="250" alt="jh" src="https://github.com/user-attachments/assets/0c08e694-5ca3-446a-8af9-e7441b83553f"> | <img width="250" alt="cy"  src="https://github.com/user-attachments/assets/e5ca222f-c577-4f9b-bcdb-1dfa71f82b26">
| --- | --- | --- | --- |
| 🐼[정현주](https://github.com/wjdguswn1203)🐼 | 🐱[송윤주](https://github.com/raminicano)🐱 | 🐶[신지현](https://github.com/sinzng)🐶 | 🐼  [이창영](https://github.com/chang558) 🐼

<br/>
<br/>



<br/>
<br/>

# 📝 내 역할
- streamlit을 사용하여 프론트 페이지 제작
- HuggingFace 모델(facebook/bart-large-cnn)을 사용한 요약
- 단락을 나누는 알고리즘 (langchain.text_splitter 사용)
- streamlit을 사용한 도표 생성
- 요약 모델 소요시간 체크
- 조회된 키워드 데이터 요약 모델에 얼마나 많이 매칭되는지 데이터로 출력



# 🔆 성과
- HuggingFace 요약 모델을 비교해서 가장 효율적인 요약 모델 선정
- keyword 추출하는 모델을 비교해서 가장 신뢰성있는 모델 선정
- 원래 특정 escape character를 통해 문단을 나눴던 알고리즘에서 text_splitter를 통해 토큰별로 나누는 알고리즘을 통해 test file을 30문단에서 17문단으로 줄임



<br/>
<br/>

# 🏆 기술 스택
## Programming language

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<br/>


## Library & Framework

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/> 
<br/>

## Server & deploy

<img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"/> 

<br/>

## API

<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/> <img src="https://img.shields.io/badge/Google Cloud Vision-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white"/> <img src="https://img.shields.io/badge/Keybert-E04E39?style=for-the-badge"/> <img src="https://img.shields.io/badge/Textrank-000000?style=for-the-badge"/> 

<br/>

## Version Control System
<img alt="github" src="https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=white"> 
<br/>


## Communication Tool

<img alt="notion" src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white"> <img alt="kakao" src="https://img.shields.io/badge/KakaoTalk-FFCD00?style=for-the-badge&logo=kakao&logoColor=black"> 


<br/>
<br/>
<br/>
