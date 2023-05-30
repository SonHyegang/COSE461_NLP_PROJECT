from utils import data_load
import random
import os
from newpipe import PipeModel
import pandas as pd
import numpy as np
import atexit

'''
make a mode - reduction, maintenance
'''

class Perceptron:
    def __init__(self):
        self.threshold = 1
        self.learning_rate = 0.01
        self.weight = -5
        self.match_threshold = 0.8
        self.threshold_direction = 0
        self.direction_text = 'none'
            
    def Learning(self, dir_path:str, model, mode = "reduction"):

        '''
        reduction
        maintenance
        '''

        model.thresholdScore = self.threshold
        f = open(file_name, 'a')
        f.write("weight : "+str(self.weight)+"\n")
        f.write("match_threshold : "+str(self.match_threshold)+"\n")
        f.close()

        json_files = [file for file in os.listdir(dir_path) if file.endswith(".json")]
        for json_file in json_files:
            ids, _, dialogues, times, summaries = data_load(source="aihub", load_mode=1, test_mode=False, path=dir_path+"/"+json_file)
            data = {'id':ids, 'dialogues':dialogues, 'times':times, 'summaries':summaries}
            data = pd.DataFrame(data)
            _ids = ids[:]

            batch_size = 2
            id_pairs = []
                
            while len(_ids) > batch_size-1:
                # Perform extraction without replacement
                extracted = random.sample(_ids, batch_size)
                id_pairs.append(extracted)
                
                # Remove the extracted elements from the collection
                for id in extracted:
                    _ids.remove(id)


            for i, id_pair in enumerate(id_pairs):
                sampled_data = []
                for id in id_pair:
                    sampled_data.append(data[data['id'] == id])
                len_sampled_data_dialogues = [len(sampled_data[i]['dialogues'].tolist()[0]) for i in range(batch_size)]
                order = [[sampled_data[i]['id'].values[0]]*len_sampled_data_dialogues[i] for i in range(batch_size)]
                order = [item for sublist in order for item in sublist]
                random.shuffle(order)
                fix_n = min(8, len(sampled_data[0]['dialogues'].tolist()[0]), len(sampled_data[1]['dialogues'].tolist()[0]))
                order = sorted(order[:fix_n]) + order[fix_n:]

                dialogues1 = sampled_data[0]['dialogues'].tolist()[0][::-1]
                dialogues2 = sampled_data[1]['dialogues'].tolist()[0][::-1]
                time1 = sampled_data[0]['times'].tolist()[0][::-1]
                time2 = sampled_data[1]['times'].tolist()[0][::-1]
                
                new_dialogues = [dialogues1.pop() if (order[i] == sampled_data[0]['id'].values[0]) else dialogues2.pop() for i in range(len(order))]
                new_times = [time1.pop() if (order[i] == sampled_data[0]['id'].values[0]) else time2.pop() for i in range(len(order))]
                ans_order = order[:]
                
                predict_order, predict_summary = model.predict(new_dialogues, new_times)
                
                match_score =  self.cal_matching_accuracy(ans_order, predict_order)
                predict_cnt = len(predict_summary)

                if mode == "reduction":
                    self.weight+=self.learning_rate / 5 # 너무 빠르게 감소해서 보정 + weight도 -4 -> -5로 변경
                    learning_rate = self.learning_rate * 1/(1+np.exp(self.weight))
                     # data가 누적됨에 따라 learning rate 감소 -> 후반 data의 영향력 줄여서 일정하게 만들기 위함
                elif mode == "maintenance":
                    learning_rate = self.learning_rate
                if match_score<self.match_threshold or predict_cnt == 1:
                    # thread 개수가 달라도 정확도가 매우 높으면 성공으로 봄(>0.8)
                    # 단 1개일 때는 match_score가 무조건 1이므로 배제
                    if predict_cnt>batch_size and model.thresholdScore >learning_rate:
                        model.thresholdScore-=learning_rate # 개수가 많으면 threshold 감소
                        self.threshold_direction = -1
                        self.direction_text = 'minus'
                    elif predict_cnt <= batch_size and model.thresholdScore < 1-learning_rate:
                        model.thresholdScore+=learning_rate  # 개수가 적으면 threshold 증가.
                        self.threshold_direction = 1
                        self.direction_text = 'plus'
                    elif match_score < self.match_threshold / 2:
                        if 1-learning_rate > model.thresholdScore > learning_rate:
                            #개수가 맞지만 정확도가 낮으므로 threshold를 가던 방향으로 조금 더 이동
                            #이 케이스가 거의 없었음. 특수 케이스에 대한 고려
                            model.thresholdScore+=learning_rate * (1/4)* self.threshold_direction
                            self.direction_text = 'plus-minus'
                    else:
                        self.direction_text = 'none'
                elif model.thresholdScore < 1-learning_rate:
                    model.thresholdScore+=learning_rate
                    self.threshold_direction = 1
                    self.direction_text = 'plus'
                    # data 특성 상 outliner가 많은 편. outliner는 minus를 만드는 요소이기에 성공 시 plus를 넣어서 보정.

                print(predict_summary)
                print(i)
                print(self.direction_text)
                print("threshold: ", model.thresholdScore)
                print("match_score : ", match_score)
                if i%10 == 0:
                    f = open(file_name, 'a')
                    f.write(str(i) + " : threshold = " + str(model.thresholdScore)+", learning_rate = " +str(learning_rate)+"\n")
                    f.write("         : match_score = " +str(match_score)+", threads_count = " +str(predict_cnt)+", direction = "+self.direction_text+"\n")
                    f.close()


        return model.thresholdScore


    def cal_matching_accuracy(self,list1: list[str], list2: list[str]) -> float:

        ids = list(set(list1))    
        id1_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[0]}
        id2_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[1]}

        best_match_id1 = max(id1_thread, key=id1_thread.get) if id1_thread else None
        best_match_id2 = max(id2_thread, key=id2_thread.get) if id2_thread else None
        matching_count = sum(1 for id, thread in zip(list1, list2) if thread == best_match_id1 and id == ids[0]) + sum(1 for id, thread in zip(list1, list2) if thread == best_match_id2 and id == ids[1])

        return matching_count / len(list1)
    
    def get_threshold_score(self, mode:str = "last"):
        '''
        mean : get mean of thresholds
        last : get last threshold
        '''
        f = open(file_name, 'r')
        texts = f.readlines()
        f.close()
        threshold_sum = 0

        if mode == 'mean':
            for i in range(2,len(texts)):
                text = texts[i]
                text = text.split(": ", maxsplit=2)[1].split(",")[0]
               # threshold = 0.xx or match_score = 0.xx
                text = text.split(" = ")
                if text[0] == "threshold":
                    threshold = float(text[1])
                    threshold_sum+=threshold
            return threshold_sum / (len(texts)//2) if len(texts)//2 != 0 else -1
        elif mode == "last":
            threshold = -1
            if len(texts) >= 4:
                text = texts[-2].split(": ", maxsplit=2)[1].split(",")[0]
                # threshold = 0.xx
                text = text.split(" = ")
                threshold = int(text[1])
            return threshold
    

def get_threshold_score(mode:str = "last"):
    '''
    mean : get mean of thresholds
    last : get last threshold
    '''
    f = open(file_name, 'r', encoding="UTF-8")
    texts = f.readlines()
    f.close()
    threshold_sum = 0

    if mode == 'mean':
        for i in range(2,len(texts)):
            text = texts[i]
            text = text.split(": ")[1].split(",")[0]
            # threshold = 0.xx or match_score = 0.xx
            text = text.split(" = ")
            if text[0] == "threshold":
                threshold = float(text[1])
                threshold_sum+=threshold
        mean = threshold_sum / (len(texts)//2) if len(texts)//2 != 0 else -1
        f = open(file_name, 'a')
        f.write("threshold mean = "+str(mean))
        f.close()
        print(mean)
        return mean
    elif mode == "last":
        threshold = -1
        if len(texts) >= 4:
            text = texts[-2].split(": ", maxsplit=2)[1].split(",")[0]
            # threshold = 0.xx
            text = text.split(" = ")
            threshold = int(text[1])
        print(threshold)
        return threshold

file_dir_num = 0
while True:
    file_name = "./log_DEMO5_"+str(file_dir_num)+".txt"
    if(os.path.exists(file_name)):
        file_dir_num+=1
        continue
    break


model = PipeModel(mode="evaluation")
per = Perceptron()

per.threshold = 1
per.learning_rate = 0.001
atexit.register(get_threshold_score, mode="mean")
per.Learning('./test_data',model, mode="reduction")
print(per.get_threshold_score(mode="mean"))
print(per.get_threshold_score(mode="last"))