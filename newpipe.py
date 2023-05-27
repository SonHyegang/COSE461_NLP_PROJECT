import torch
from transformers import pipeline, AutoTokenizer, BartForConditionalGeneration
import numpy as np
from utils import data_load
import os

class PipeModel:
    def __init__(self, mode: str="test"):
        self.max_length = 100
        self.num_beams = 5
        self.length_penalty = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained("Hyegang/BART_COSE461_TEAM32")
        self.model = BartForConditionalGeneration.from_pretrained("Hyegang/BART_COSE461_TEAM32")
        self.inference_time = 0
        self.model_name = "pipe"
        self.summarizer = pipeline("summarization", model="Hyegang/BART_COSE461_TEAM32")
        self.time_mode = "sigmoid"
        self.min_max_time = 0
        self.numSentence = 0
        self.maxThreadNumber = 2

        self.threads = [] #threads : thread들(string of list)로 이루어진 이중 리스트
        self.threadSum = [] #threadSum : list of string(summary of each thread)
        self.threadId = [] #threadId : 생성된 thread id관리
        self.threadTime = []
        
        self.whichThread = [] #threadId : thread에 문장 포함될 때마다 어느 thread에 포함됐는지 관리
        self.threadDict = {} #threadDict : thread와 그 요약문 매치
        self.thresholdScore = 0.7

        self.mode = mode

    def time_weighted(self, times: tuple, mode:str = "sigmoid"):
        '''
        times: (first:str, second:str)
                format = "00:00"
        mode:
            'sigmoid', 'tanh', 'linear'
        '''
        float_time = []
        for time in times:
            time = self.time_to_float(time)
            float_time.append(time)
        time_gap = float_time[1]-float_time[0]
        if(self.min_max_time==0):
            return 1

        if(mode == "sigmoid"):
            time_gap = -2 * 4 * time_gap/self.min_max_time + 4
            return 1/(1+np.exp(-time_gap))
        elif (mode == "tanh"):
            time_gap = -2 * 2 * time_gap/self.min_max_time + 2
            return np.exp(2*time_gap)/(np.exp(2*time_gap)+1)
        elif(mode == "linear"):
            time_gap = time_gap/self.min_max_time
            return time_gap
        return

    def makeEmbedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            max_length=self.max_length,
            use_cache=True,
        )
        return outputs
    
    def padding(self, embed_set:list):
        threads_len = list(map(len, embed_set))
        max_len = max(threads_len)
        padded_set = [torch.cat((torch.zeros(max_len-len(embed_set[i])),embed_set[i]), dim=-1) for i in range(2)]

        return padded_set
    
    def similarityCheck(self, sentence1, sentence2):
        #embedding 수정 예정
        embedding1 = self.makeEmbedding(sentence1).to(torch.float32)
        embedding2 = self.makeEmbedding(sentence2).to(torch.float32)
        #embeddings = [embedding1[0],embedding2[0]]
        embeddings = [embedding1.squeeze(),embedding2.squeeze()]
        a = self.padding(embeddings)
        ##a[0]랑 a[1]을 tensor로
        #cos_scores = util.pytorch_cos_sim(self.makeEmbedding(sentence1), self.makeEmbedding(sentence2))[0]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_scores = cos(a[0].reshape(1,-1), a[1].reshape(1,-1))
        return cos_scores
    
    def addSentence(self, sentence, time):
        if (20 >= self.numSentence > 0):
            self.maxThreadNumber = 2
        elif(100 > self.numSentence > 20 ):
            self.maxThreadNumber = 3
        elif(300 > self.numSentence >= 100):
            self.maxThreadNumber = 4
        else : 
            self.maxThreadNumber = 5
        
        if self.mode == "evaluation":
            self.maxThreadNumber = 10
        bestScore = 0
        bestIndex = 0
        #가장 similarity가 높은 threadSum 찾기
        for i in range(0,len(self.threadSum)):
            time_parameter = self.time_weighted((self.threadTime[i],time), self.time_mode)
            current_score =  time_parameter *self.similarityCheck(sentence, self.threadSum[i])
            if(bestScore < current_score):
                bestScore = current_score
                bestIndex = i
        
        #가장 similarity가 높은 threadSum이 thresholdScore보다 높으면 병합
        if (self.threads != []) and (bestScore > self.thresholdScore):
            self.threads[bestIndex].append(sentence)
            #self.threadTime[bestIndex] = time
            self.threadSum[bestIndex] = self.summarizer("[BOS]" + "[SEP]".join(self.threads[bestIndex]) + "[EOS]", max_length=self.max_length)[0]['summary_text']
            self.whichThread.append("thread" + str(bestIndex))#어떤 thread에 추가 되었는지 추적
        #가장 similarity가 높은 threadSum이 thresholdScore보다 낮다면 새로운 thread 생성
        else:
            if(self.threads != [] and len(self.threadSum) >= self.maxThreadNumber) : #thread 개수
                self.threads[bestIndex].append(sentence)
                self.threadSum[bestIndex] = self.summarizer("[BOS]" + "[SEP]".join(self.threads[bestIndex]) + "[EOS]", max_length=self.max_length)[0]['summary_text']
                self.whichThread.append("thread" + str(bestIndex))#어떤 thread에 추가 되었는지 추적
            else : 
                newThread = [sentence]
                new_summ = self.summarizer("[BOS]" + "[SEP]".join(newThread) + "[EOS]", max_length=self.max_length)[0]['summary_text']
                if new_summ != "":
                    self.whichThread.append("thread" + str(len(self.threadSum)))
                    self.threadId.append("thread" + str(len(self.threadSum)))#새로운 thread를 생성했을 때, 그 thread id 새로 추가
                    self.threadSum.append(new_summ)
                    self.threads.append(newThread)
                    self.threadTime.append(time)

        return self.threadSum
    
    
    def predict(self,utterances_list, times):#load 데이터로 받은 list of string 처리
        self.threads = [] #threads : thread들(string of list)로 이루어진 이중 리스트
        self.threadSum = [] #threadSum : list of string(summary of each thread)
        self.threadId = [] #threadId : 생성된 thread id관리
        self.threadTime = []
        self.min_max_time = self.time_to_float(times[-1])-self.time_to_float(times[0])
        self.numSentence = len(times) #문장 개수 count
        
        self.whichThread = [] #threadId : thread에 문장 포함될 때마다 어느 thread에 포함됐는지 관리
        self.threadDict = {} #threadDict : thread와 그 요약문 매치
        for i in range(len(utterances_list)):
            self.addSentence(utterances_list[i], times[i])#한 문장 씩 추가
        for i in range(len(self.threadSum)):
            self.threadDict[self.threadId[i]] = self.threadSum[i]
        return self.whichThread, self.threadDict #array of string(summary)
    #thread의 id(아무 string)와 {thread id : thread의 요약문}  튜플로 출력
    
    def time_to_float(self, time:str) -> float:
        tmp = time.split(":")
        return float(tmp[0]) * 60+float(tmp[1])
    


# chats = data_load("kakao",1,True)
# times = chats[3][0]
# sentence = chats[2][0]
# print(chats)
# pipemodel = PipeModel()
# pipemodel.predict(sentence,times)


# file_dir_num = 0
# while True:
#     file_name = "./log_pipe_"+str(file_dir_num)+".txt"
#     if(os.path.exists(file_name)):
#         file_dir_num+=1
#         continue
#     break


# f = open(file_name, 'a', encoding='UTF-8')
# f.write(str(sentence))
# f.write("="*20+"\n")
# f.write("max_threshold : "+str(pipemodel.thresholdScore)+"\n")
# f.write("time mode : " + str(pipemodel.time_mode)+"\n")
# f.write("time gap: "+str(pipemodel.min_max_time)+"\n")
# f.close()


# for i in range(len(sentence)):
#     print(sentence[i])
#     pipemodel.addSentence(sentence[i], times[i])
#     f = open(file_name, 'a', encoding='UTF-8')
#     f.write("="*20+"\n")
#     f.write(str(i) + " :" + str(pipemodel.threadSum)+"\n")
#     f.write(str(i) + " :" + str(pipemodel.threadTime)+"\n")
#     f.close()



