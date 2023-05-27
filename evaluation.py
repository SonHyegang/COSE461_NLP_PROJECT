"""
HGSon
2023.05.16.
2023.05.19. (rev)
    - Added comparison of baseline and our model
    - Fix index error issue of baseline model :log1_baseline_evaluation_230519.txt
    - Fix matching accuracy calculation issue
2023.05.20. (rev)
    - Modify the rouge scoring code according to the inference return format of our model
2023.05.22. (rev)
    - Modified to fit our model (Model ver :230522_JWChoi)
    - Modify the way dialogue sets are created for evaluation
2023.05.23. (rev)
    - Modify the method of matching accuracy calculation
"""
import os
import random
import pandas as pd
import time
from newpipe import PipeModel
from pororo import Pororo
from transformers import AutoTokenizer, BartForConditionalGeneration
from rouge import Rouge
from utils import data_load
from typing import Tuple, Dict, List


class BaseModel_pororo:
    """
    Baseline for evaluation

    Simple summary without thread division of conversations
    """
    def __init__(self):
        self.model = Pororo(task="summarization", model="abstractive", lang="ko")
        self.inference_time = 0
        self.model_name = "baseline_pororo"
    
    def predict(self, utterances_list):
        utterances = '  '.join(utterances_list)
        predicted_output = self.model(utterances)
        
        return None, predicted_output


class BaseModel_aihub:
    """
    Baseline for evaluation

    Simple summary without thread division of conversations
    """
    def __init__(self):
        self.model = "Hyegang/BART_COSE461_TEAM32"
        self.max_length = 64
        self.num_beams = 5
        self.length_penalty = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = BartForConditionalGeneration.from_pretrained(self.model)
        self.inference_time = 0
        self.model_name = "baseline_BART"
    
    def predict(self, utterances_list):
        input_text = "[BOS]" + "[SEP]".join(utterances_list) + "[EOS]"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=self.num_beams,
            length_penalty=self.length_penalty,
            max_length=self.max_length,
            use_cache=True,
        )
        predicted_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return None, predicted_output
    

def cal_matching_accuracy(list1: List[str], list2: List[str], test_mode :bool=False) -> float:
    """
    Calculate the matching accuracy between two lists based on specific conditions.

    Parameters:
        list1 (List[str]): The first list = predict_order
        list2 (List[str]): The second list = ans_order (id order of created dialogs)

    Returns:
        matching accuracy: The matching accuracy between the two lists.
        raw_id1
        best_match_id1
        raw_id2
        best_match_id2
    """
    ids = list(set(list1))    
    id1_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[0]}
    id2_thread = {elem: list2.count(elem) for elem, id_elem in zip(list2, list1) if id_elem == ids[1]}

    best_match_id1 = max(id1_thread, key=id1_thread.get) if id1_thread else None
    best_match_id2 = max(id2_thread, key=id2_thread.get) if id2_thread else None

    matching_count = sum(1 for id, thread in zip(list1, list2) if thread == best_match_id1 and id == ids[0]) + sum(1 for id, thread in zip(list1, list2) if thread == best_match_id2 and id == ids[1])
    if best_match_id1 == best_match_id2:
        matching_count = max(sum(1 for id, thread in zip(list1, list2) if thread == best_match_id1 and id == ids[0]), sum(1 for id, thread in zip(list1, list2) if thread == best_match_id2 and id == ids[1]))

    matching_id = [best_match_id1 if thread == best_match_id1 and id == ids[0] else best_match_id2 if thread == best_match_id2 and id == ids[1] else '' for id, thread in zip(list1, list2)]

    if test_mode:
        print("================================================================")
        print("ids :" + str(ids[0]), str(ids[1]))
        print("id_thread :" + str(id1_thread), str(id2_thread))
        print("best_match_id :", best_match_id1, best_match_id2)
        print(matching_id)
        print("================================================================")

    return matching_count / len(list1), (ids[0], best_match_id1), (ids[1], best_match_id2)



def evaluation(model, directory_path :str, test_mode :bool=False):
    """
    Evaluation function for our model

    Parameters:
        directory_path : path of json files folder
                Select between training and validation folders when decompressing AIhub data
        test_mode : If test_mode, output the first element of the dataset as an example
                    True == Loaded data sample output
                    False == no output  default
                
    Returns:
        Score : (total_score, {json name, List[([id1, id2], score)]})
    """
    total_score = {'matching_score':0, 'rouge_score':0}
    
    rouge = Rouge()

    # Create evaluation dialog
    json_files = [file for file in os.listdir(directory_path) if file.endswith(".json")]
    score = {key: [] for key in json_files}
    n = 0
    # for i, json_file in enumerate(json_files):
        
    #     matching_accuracy, id1, _, id2, _ = cal_matching_accuracy()
    for json_file in json_files:
        ids, participantIDs, dialogues, times, summaries = data_load(source="aihub", load_mode=0, test_mode=False, path=directory_path+"/"+json_file)
        data = {'id':ids, 'participantIDs':participantIDs, 'dialogues':dialogues, 'times':times, 'summaries':summaries}
        data = pd.DataFrame(data)
        _ids = ids[:]

        temp_score = {'matching_score':0, 'rouge_score':0}

        batch_size = 2
        id_pairs = []
            
        while len(_ids) > batch_size-1:
            # Perform extraction without replacement
            extracted = random.sample(_ids, batch_size)
            id_pairs.append(extracted)
            
            # Remove the extracted elements from the collection
            for id in extracted:
                _ids.remove(id)

        n += len(id_pairs)

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
            ans_summaries = [(i['id'].values[0], i['summaries'].values[0]) for i in sampled_data]
            weights = [len(dialogue) for dialogue in new_dialogues]   
            weights = [weight/sum(weights) for weight in weights] # Normalization 

            # for test
            # predict_order = ans_order[:]
            # predict_summaries = ans_summaries[:]

            start_time = time.time()
            try:
                if model.model_name in ["baseline_aihub", "baseline_pororo"]:
                    predict_order, predict_summaries = model.predict(new_dialogues)
                else:
                    predict_order, predict_summaries = model.predict(new_dialogues, new_times)
                    print(predict_order, predict_summaries)
            except IndexError as e:
                print("An IndexError occurred:", str(e))
                continue
            # predict_order, predict_summaries = inference(new_dialogues) # Our model ['THREAD B', 'THREAD A', 'THREAD B'] ['', '']
            model.inference_time += time.time()-start_time
            
            matching_accuracy = 0
            if model.model_name in ["baseline_aihub", "baseline_pororo"]:
                matching_accuracy = 0
            else:
                # matching_accuracy = sum((predict_id == ans_id)*weight for predict_id, ans_id, weight in zip(predict_order, ans_order, weights))
                matching_accuracy, (id1, best_match_id1), (id2, best_match_id2) = cal_matching_accuracy(ans_order, predict_order)
                print(id1, best_match_id1, id2, best_match_id2)
            rouge_score = 0

            if model.model_name in ["baseline_aihub", "baseline_pororo"]:
                ans_summary = '. '.join([tup[1] for tup in ans_summaries])
                rouge_score += rouge.get_scores(predict_summaries, ans_summary)[0]['rouge-1']['f']
            else:
                # for predict_summary, ans_summary in zip(predict_summaries, ans_summaries):
                #     rouge_score += rouge.get_scores(predict_summary, ans_summary)[0]['rouge-1']['f']
                
                for item in ans_summaries:
                    if item[0] == id1:
                        answer_summary_id1 = item[1]
                    if item[0] == id2:
                        answer_summary_id2 = item[1]
            
                predict_summary_id1 = predict_summaries[best_match_id1]
                predict_summary_id2 = predict_summaries[best_match_id2]

                
                print(predict_summary_id1, predict_summary_id2)


                rouge_score += rouge.get_scores(predict_summary_id1, answer_summary_id1)[0]['rouge-1']['f']
                rouge_score += rouge.get_scores(predict_summary_id2, answer_summary_id2)[0]['rouge-1']['f']
                rouge_score /= batch_size # Normalization

            score[json_file].append((id_pair, matching_accuracy, rouge_score))
            total_score['matching_score'] += matching_accuracy
            total_score['rouge_score'] += rouge_score
            
            temp_score['matching_score'] += matching_accuracy
            temp_score['rouge_score'] += rouge_score

            if test_mode and (i%100 == 0):
                print("================================================================")
                # print(order)

                print("original data 0")
                print(data[data['id']==id_pair[0]].T)
                print()
                print("original data 1")
                print(data[data['id']==id_pair[1]].T)
                print()
                # print("New dialogues")
                # for j, dialogue in enumerate(new_dialogues):
                    # print("%s  %s:%-50s  :%f"%(order[j], predict_order[j], dialogue, weights[j]))
                print("================================================================")
                print("Matching_accuracy :" + str(matching_accuracy))
                print("ROUGE-score(Rouge-1 F1 score) :" + str(rouge_score))
                print("================================================================")
            if i%100 == 0:
                print(json_file + " Iter :" + str(i))
                print("Cumulative inference time :" + str(model.inference_time))
                print("Ans summary :" + str(ans_summaries))
                print("Pred summary :" + str(predict_summaries))
                print("Matching_accuracy :" + str(matching_accuracy))
                print("ROUGE-score(Rouge-1 F1 score) :" + str(rouge_score))
                print("================================================================")    
        
        f = open("./evaluation_log.txt", 'a')
        temp_score['matching_score'] /= len(id_pairs)
        temp_score['rouge_score'] /= len(id_pairs)
        f.write(str(json_file) + "\n")
        f.write("score : " + str(temp_score) + "\n")
        f.write("time : " + str(model.inference_time) + "\n")
        f.close()
        
    total_score['matching_score'] /= n
    total_score['rouge_score'] /= n
    
    f = open("./evaluation_log.txt", 'a')
    f.write("\n================================================================\n")
    f.write(str(model.model_name) + "\n")
    f.write("total_score :" + str(total_score) + "\n")
    f.write("time : " + str(model.inference_time) + "\n")
    f.write("\n================================================================\n")
    f.close()
    
    return (total_score, score)

# for test
# start_time = time.time()
# total_score, score = evaluation(directory_path="/root/data_AIhub/Validation/[라벨]한국어대화요약_valid", test_mode=True) #/root/data_AIhub/Training/[라벨]한국어대화요약_train
# print(total_score)
# print(score['개인및관계.json'])
# print(time.time()-start_time)

# for evaluation
# model = BaseModel_pororo()
# model = BaseModel_aihub()
# model = newpipe_2.PipeModel()
# total_score, score = evaluation(model=model, directory_path="/root/data_AIhub/Validation/[라벨]한국어대화요약_valid", test_mode=True)
# print("==================================================")
# print("Baseline")
# print("total_score :" + str(total_score))
# print("inference time :" + str(model.inference_time))
# print("==================================================")
