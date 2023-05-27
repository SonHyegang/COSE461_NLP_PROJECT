from evaluation import evaluation 
from newpipe import PipeModel
import numpy as np
import os

def grid_search(model, dir_path:str, grid:list):
    max_score = -1
    max_threshold = -1
    for threshold in grid:
        model.thresholdScore = threshold
        total_score, _ = evaluation(model, dir_path, test_mode=True)
        print("\n======================================\n")
        print("threshold :" + str(threshold))
        print("score :" + str(total_score))
        print("\n======================================\n")
        f = open("./log_grid_search.txt", "a")
        f.write("\n======================================\n")
        f.write(str(threshold) + "\n")
        f.write(str(total_score))
        f.write("\n======================================\n")
        f.close()

        # if total_score['rouge_score'] > max_score:
        #     max_score = total_score
        #     max_threshold = threshold

    return max_threshold

def recursive_grid_search(model ,dir_path:str, threshold:float, gap:float, limit:float = 0.05):
    if gap <= limit:
        return threshold
    
    start = threshold-gap if threshold >= gap else 0
    end = threshold+gap if threshold <= 1 - gap else 1
    gap = gap/5
    iteration = np.arange(start, end, gap)
    max_threshold = grid_search(model, dir_path, iteration)
    return recursive_grid_search(model, dir_path, max_threshold, gap, limit)

model = PipeModel(mode = "evaluation")
f = open("./log_grid_search.txt", "a")
f.write("\n======================================\n")
f.write(str(model.model_name) + "\n")
f.write("\n======================================\n")
f.close()
print(model.model_name)
print("====================================")
print(grid_search(model, "/root/data_AIhub_evaluation/Validation/[라벨]한국어대화요약_valid", [x / 10 for x in range(11)][4:][::-1]))