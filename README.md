# COSE461_NLP_PROJECT

- 2023년도 1학기 자연어처리 과제로 제작한 Conversation Segregation Model입니다.

- 조원 : 류희철, 손혜강, 최진우

## Installation

- 아래 커맨드로 설치할 수 있습니다.

    pip install transformers
    pip install numpy
    pip install pandas


```Python
    data2 = data_load(source="kakao", load_mode=1, test_mode=True)
```
    

## Instruction

- 저희 조는 여러 주제의 긴 대화에 대한 요약을 제공하는 모델을 설계하였습니다. 이 모델은 읽지 않은 채팅이 과도하게 쌓여 일일이 확인할 수 없을 때 주요 내용들을 여러 thread로 구분하여 각
thread에 대한 요약문장을 제공함으로 문장을 해석합니다.
