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
thread에 대한 요약문장을 제공함으로 사용자가 전체 대화를 이해하도록 돕는 목적으로 제작되었습니다.


## Model

### Mechanism

- 기존 thread 요약문들과 새로운 문장 간의 유사도 점수를 비교하여 가장 높은 점수를 갖는 thread에 그 문장을 포함시키며 해당 thread의 요약문을 업데이트합니다. 다만 threshold보다 점수가 낮다면 thread에 포함시키지 않고 새로운 문장의 흐름이 개시되었다고 판단하여 새로운 thread를 형성합니다.

### Thread

 - thread는 이 모델에서 대화 주제의 개념으로 사용되었습니다. 다수의 인원이 대화를 할 때 여러 대화 주제가 병렬적으로 진행된다는 점, 그리고 전체 대화와 별개로 시작하고 종료된다는 점에서 thread와 유사하여 사용했습니다.

### Summarization

- 특정 thread에 포함되는 문장들은 해당 thread에 새로운 문장이 추가될 때마다 그 thread를 대표하는 요약문장이 갱신됩니다.

 ```
self.threads[bestIndex].append(sentence)
self.threadSum[bestIndex] = self.summarizer("[BOS]" + "[SEP]".join(self.threads[bestIndex]) + "[EOS]", max_length=self.max_length)[0]['summary_text']
self.whichThread.append("thread" + str(bestIndex))
 ```

### Scoring

- 두 문장에 대한 임베딩을 형성하고 형성된 임베딩을 바탕으로 두 문장의 cosine similarity를 구합니다. 다만 대화의 주제는 시간이 지남에 따라 영향력이 떨어진다는 점을 고려하여 시간에 의한 가중치를 고려하여 점수를 계산하였습니다.
```
time_parameter = self.time_weighted((self.threadTime[i],time), self.time_mode)
current_score =  time_parameter *self.similarityCheck(sentence, self.threadSum[i])
```


### Creating Thread

- 새롭게 생겨나는 thread는 이 thread에 생성 시점과 기준이 되는 요약문을 갖습니다. 다만 대량의 채팅을 입력으로 받아 소량의 thread로 나누는 model의 목적에 따라서 사용자의 편의성을 위해 입력의 양에 따라서 생성되는 thread의 개수를 제한하였습니다.

```
if(self.threads != [] and len(self.threadSum) >= self.maxThreadNumber) :
    self.threads[bestIndex].append(sentence)
    self.threadSum[bestIndex] = self.summarizer("[BOS]" + "[SEP]".join(self.threads[bestIndex]) + "[EOS]", max_length=self.max_length)[0]['summary_text']
    self.whichThread.append("thread" + str(bestIndex))
else : 
    newThread = [sentence]
    new_summ = self.summarizer("[BOS]" + "[SEP]".join(newThread) + "[EOS]", max_length=self.max_length)[0]['summary_text']
    if new_summ != "":
        self.whichThread.append("thread" + str(len(self.threadSum)))
        self.threadId.append("thread" + str(len(self.threadSum)))
        self.threadSum.append(new_summ)
        self.threads.append(newThread)
        self.threadTime.append(time)
```


## Thread Learning Model

### Mechanism

-Thread Learning Model의 parameter는 accuracy와 thread 개수입니다. accuracy는 이 모델의 주요 매개변수로 프로젝트의 목적이 대량의 카톡 대화를 요약하고 분리하는 것이기에 적절한 thread를 분리하는 것이 가장 중요합니다. 따라서 다른 것보다 모델 요약의 accuracy를 우선시했습니다.

### Modification Phase

-Modification은 높은 threshold는 많은 thread를 생성하고 낮은 threshold는 적은 thread를 생성한다는 규칙에 따라 수정했습니다. thread 수가 batch size 2보다 크면 모델이 threshold를 줄입니다. 반대 상황에서는 threshold를 높입니다.
- 또한 만일 정확도가 낮아 Modification Phase에 들어왔음에도 thread 개수가 batch size 2와 동일한 경우가 있습니다. 이 경우 정확도가 눈에 띄게 작아지므로 모델은 진행 방향으로 임계값을 아주 약간 이동합니다.

```
if predict_cnt>batch_size and model.thresholdScore >learning_rate:
    model.thresholdScore-=learning_rate
    self.threshold_direction = -1
elif predict_cnt < batch_size and model.thresholdScore < 1-learning_rate:
    model.thresholdScore+=learning_rate
    self.threshold_direction = 1
elif match_score < self.match_threshold / 2:
    if 1-learning_rate > model.thresholdScore > learning_rate:
        model.thresholdScore+=learning_rate * (1/4)* self.threshold_direction
```

### Correction

- 대화라는 입력의 특성 상 무의미한 단어가 포함된 경우는 너무 많고 다양해서 이러한 경우를 특정하고 찾는 데 너무 많은 시간과 메모리를 필요로 했습니다.  따라서  성공한 사례에서 임계값을 보정하여 이러한 사례를 수정했습니다.



## Grid Search Model

- 학습 모델을 구축한 후 local optima에 빠지지 않았는지 확인학이 위하여 Grid Search Model을 만듭니다.

```
max_threshold = -1
for threshold in grid:
    model.thresholdScore = threshold
    total_score, _ = evaluation(model ,dir_path)
```

### Recursive

- grid search model를 반복하여 시행하여 원하는 깊이까지 조정합니다.
```
start = threshold-gap if threshold >= gap else 0
end = threshold+gap if threshold <= 1 - gap else 1
gap = gap/5
iteration = np.arange(start, end, gap)
max_threshold = grid_search(model, dir_path, iteration)
```
