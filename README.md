# COSE461_NLP_PROJECT

- 2023년도 1학기 자연어처리 과제로 제작한 Conversation Segregation Model입니다.

- 조원 : 류희철, 손혜강, 최진우

## Installation

- 아래 커맨드로 필요한 라이브러리를 설치할 수 있습니다.

```
pip install transformers
pip install numpy
pip install pandas
```

## USING
 ```Python
from transformers import AutoTokenizer, BartForConditionalGeneration
model_name = "Hyegang/BART_COSE461_TEAM32"
max_length = 64
num_beams = 5
length_penalty = 1.2
dialogue = ['파스타 하면 사람 많아서 면 불고 그러지 않을까?',
'한 번에 11인분 다 끓일 수 있어?',
'쉽지 않기는 해요… 딴 데 보니까 파스타 두 종류 해서 따로따로 하더라구요',
'월남쌈 진짜 괜찮긴했는데',
'아니면 고기 볶아서 쌈싸먹을까?',
'상추 깻잎 하고 양념고기 사서 볶고',
'오홍',
'옹 이것도 괜찮긴 하다 진짜 집밥 같고',
'간단하긴 할 듯여',
'이게 그냥 제육볶음인가?-??',
'옹 좋은데',
'다들 어떠싱지??',
'오 좋아']

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

start_time = time.time()
inputs = tokenizer("[BOS]" + "[SEP]".join(dialogue) + "[EOS]", return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=num_beams,
    length_penalty=length_penalty,
    max_length=max_length,
    use_cache=True,
)
encode_time = time.time()-start_time
start_time = time.time()
summarization = tokenizer.decode(outputs[0], skip_special_tokens=True)
decode_time = time.time()-start_time
print("==================================================")
print(summarization)


>>>파스타를 하면 사람이 많아서 면을 불지 않을지 이야기하고 월남쌈이나 고기를 볶아서 쌈을 싸 먹을지 이야기한다.
 ```


## DEMO w/ KaokaoTalk data
https://colab.research.google.com/drive/17NtD-kcciKjq_IwVt78G9lRWtRF0Hlmw?usp=sharing    

### Example of DEMO







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

 ```Python
self.threads[bestIndex].append(sentence)
self.threadSum[bestIndex] = self.summarizer("[BOS]" + "[SEP]".join(self.threads[bestIndex]) + "[EOS]", max_length=self.max_length)[0]['summary_text']
self.whichThread.append("thread" + str(bestIndex))
 ```

### Scoring

- 두 문장에 대한 임베딩을 형성하고 형성된 임베딩을 바탕으로 두 문장의 cosine similarity를 구합니다. 다만 대화의 주제는 시간이 지남에 따라 영향력이 떨어진다는 점을 고려하여 시간에 의한 가중치를 고려하여 점수를 계산하였습니다.
```Python
time_parameter = self.time_weighted((self.threadTime[i],time), self.time_mode)
current_score =  time_parameter *self.similarityCheck(sentence, self.threadSum[i])
```


### Creating Thread

- 새롭게 생겨나는 thread는 이 thread에 생성 시점과 기준이 되는 요약문을 갖습니다. 다만 대량의 채팅을 입력으로 받아 소량의 thread로 나누는 model의 목적에 따라서 사용자의 편의성을 위해 입력의 양에 따라서 생성되는 thread의 개수를 제한하였습니다.

```Python
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

```Python
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

```Python
max_threshold = -1
for threshold in grid:
    model.thresholdScore = threshold
    total_score, _ = evaluation(model ,dir_path)
```

### Recursive

- grid search model를 반복하여 시행하여 원하는 깊이까지 조정합니다.

```Python
start = threshold-gap if threshold >= gap else 0
end = threshold+gap if threshold <= 1 - gap else 1
gap = gap/5
iteration = np.arange(start, end, gap)
max_threshold = grid_search(model, dir_path, iteration)
```

## Evaluation

- 프로젝트의 평가를 위해 구현하였으며 주어진 데이터에 대해 rouge score와 matching accuracy를 측정합니다.

### Mechanism

- 대화 속에서 thread를 구분하는 목적에 맞춰 두 개의 dialog를 합친 후 기존의 summary 2개와 분리해낸 thread를 비교하여 평가합니다.

```Python
dialogues1 = sampled_data[0]['dialogues'].tolist()[0][::-1]
dialogues2 = sampled_data[1]['dialogues'].tolist()[0][::-1]
time1 = sampled_data[0]['times'].tolist()[0][::-1]
time2 = sampled_data[1]['times'].tolist()[0][::-1]

new_dialogues = [dialogues1.pop() if (order[i] == sampled_data[0]['id'].values[0]) else dialogues2.pop() for i in range(len(order))]
new_times = [time1.pop() if (order[i] == sampled_data[0]['id'].values[0]) else time2.pop() for i in range(len(order))]
            
predict_order, predict_summaries = model.predict(new_dialogues, new_times)
```

### Accuracy

- 기존의 summary와 thread를 비교하여 얼마나 정확하게 분류하였는지 평가합니다.

```Python
def cal_matching_accuracy(list1: List[str], list2: List[str]) -> float:
    ids = list(set(list1))    
    id1_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[0]}
    id2_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[1]}

    best_match_id1 = max(id1_thread, key=id1_thread.get) if id1_thread else None
    best_match_id2 = max(id2_thread, key=id2_thread.get) if id2_thread else None

    matching_count = sum(1 for id, thread in zip(list1, list2) if thread == best_match_id1 and id == ids[0]) + sum(1 for id, thread in zip(list1, list2) if thread == best_match_id2 and id == ids[1])
    if best_match_id1 == best_match_id2:
        matching_count = max(sum(1 for id, thread in zip(list1, list2) if thread == best_match_id1 and id == ids[0]), sum(1 for id, thread in zip(list1, list2) if thread == best_match_id2 and id == ids[1]))

    matching_id = [best_match_id1 if thread == best_match_id1 and id == ids[0] else best_match_id2 if thread == best_match_id2 and id == ids[1] else '' for id, thread in zip(list1, list2)]

    return matching_count / len(list1), (ids[0], best_match_id1), (ids[1], best_match_id2)
```


## Dater Loader

- 프로젝트의 학습과 평가에 사용되는 AI-Hub 데이터와 실사용에 쓰이는 kakao talk 데이터를 모두 받아들이기 위해 구현했습니다.

### Data Presprocessing

- 대화에는 화자가 나타나지 않고 다양한 화자가 불규칙적으로 나타나므로 화자 구분이 중요합니다. 따라서 화자를 드러내고 연속으로 나타날 경우 병합하는 방식으로 전처리 했습니다.

```Python
def postprocessing(speaker: str, dialogue: str) -> str:
    particile = ''
    initial_consonant = extract_initial_consonant(speaker[-1])
    if initial_consonant in ['ㄱ', 'ㄴ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅈ', 'ㅊ', 'ㅌ', 'ㅎ']:
        particile = '이'
    else:
        particile = '가'
    return speaker + particile + " " + dialogue + "라고 말했다."
```
```Python
if dialogue["participantID"] == prev_speaker_id:
    prev_line += " " + utterance
else:
    if prev_line:
        utts.append(prev_line)
        _times.append(_data["body"]["dialogue"][i-1]["time"][0:5]) # Hour:minute:second(00:00:00) -> Hour:minute(00:00)
        _participantIDs.append(dialogue["participantID"])
    prev_line = utterance
    prev_speaker_id = dialogue["participantID"]
```
