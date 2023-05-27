# COSE461_NLP_PROJECT

- 2023년도 1학기 자연어처리 과제로 제작한 Conversation Segregation Model입니다.

- 조원 : 류희철, 손혜강, 최진우

## Installation

- 아래 커맨드로 필요한을파일을 설치할 수 있습니다.

```
pip install transformers
pip install numpy
pip install pandas
```

## Using

다음과 같은 명령어로 카카오톡 대화를 입력으로 받아 실행할 수 있습니다
```Python
from utils import data_load
from newpipe import PipeModel

chats = data_load(source="kakao",load_mode=1,test_mode=False)
times = chats[3][0]
sentence = chats[2][0]

pipemodel = PipeModel()
predict_order, predict_summaries = pipemodel.predict(sentence,times)
```
    

## Instruction

- 저희 조는 여러 주제의 긴 대화에 대한 요약을 제공하는 모델을 설계하였습니다. 이 모델은 읽지 않은 채팅이 과도하게 쌓여 일일이 확인할 수 없을 때 주요 내용들을 여러 thread로 구분하여 각
thread에 대한 요약문장을 제공함으로 문장을 해석합니다.


## Model

### 작동 방식

- 기존 thread 요약문들과 새로운 문장 간의 유사도 점수를 비교하여 가장 높은 점수를 갖는 thread에 그 문장을 포함시키며 해당 thread의 요약문을 업데이트합니다. 다만 threshold보다 점수가 낮다면 thread에 포함시키지 않고 새로운 문장의 흐름이 개시되었다고 판단하여 새로운 thread를 형성합니다.

### Thread

 - thread는 이 모델에서 대화 주제의 개념으로 사용되었습니다. 다수의 인원이 대화를 할 때 여러 대화 주제가 병렬적으로 진행된다는 점, 그리고 전체 대화와 별개로 시작하고 종료된다는 점에서 thread와 유사하여 사용했습니다.

### Summarization

- 특정 thread에 포함되는 문장들은 해당 thread에 새로운 문장이 추가될 때마다 그 thread를 대표하는 요약문장이 갱신됩니다.

### Scoring

- 두 문장에 대한 임베딩을 형성하고 형성된 임베딩을 바탕으로 두 문장의 cosine similarity를 구합니다. 다만 대화의 주제는 시간이 지남에 따라 영향력이 떨어진다는 점을 고려하여 시간에 의한 가중치를 고려하여 점수를 계산하였습니다.

### Creating Thread

- 새롭게 생겨나는 thread는 이 thread에 생성 시점과 기준이 되는 요약문을 갖습니다. 다만 대량의 채팅을 입력으로 받아 소량의 thread로 나누는 model의 목적에 따라서 사용자의 편의성을 위해 입력의 양에 따라서 생성되는 thread의 개수를 제한하였습니다.
