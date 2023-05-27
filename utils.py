"""
HGSon
2023.05.11.
2023.05.16. (rev)
    - Fix return order bug
2023.05.17. (rev)
    - Added pre-processing to KakaoTalk data loader (time, displaying speaker in dialogue, line break)
2023.05.22. (rev)
    - Resolving the KakaoTalk data date change issue (Considering the case where 12:00 am is displayed instead of 00:00 am)
"""
import json
import re
from typing import List, Tuple

def data_load(source: str="aihub", load_mode: int=1, test_mode: bool=False, **args) -> Tuple[List[str], List[List[str]], List[str], List[List[str]]]:
    """
    Dataloader for our model

    Parameters:
        source : Data source
                kakao == Kakaotalk data            (Execute copy and paste of conversation contents in KakaoTalk application)
                aihub == AIhub data     default    (json file)
        load_mode : In the case of load_mode, if the speaker of consecutive speech balloon is the same, 
                    the speech balloons are merged into one speech balloon.
                    0 == Separate individual speech balloon
                    1 == Merge speech balloon if the speaker is the same    default
        test_mode : If test_mode, output the first element of the dataset as an example
                    True == Loaded data sample output
                    False == no output  default
                
    Returns:
        Tuple of ids, participantIDs, dialogues, times, summaries   
        (If KakaoTalk data is the source, id returns -1, participantID returns username, and summary returns "".)
    """
    assert source in ["aihub", "kakao"], 'This data source is not supported.'
    if source == "kakao":
        return load_kakao_data(load_mode, test_mode)
    elif source == "aihub":
        return load_json_data(load_mode, test_mode, **args)

def preprocessing(data: List[str]) -> List[str]:
    """
    Preprocessor for load_kakao_data

    Parameters:
        data : original data
                
    Returns:
        preprocessed data
    """
    previous_element = ""
    preprocessed_data = []

    for _data in data:
        pattern_date = r"\d+년 \d+월 \d+일" # In the case of date display text, increase day by 1 and add 24 hours as much as the date has changed.
        match = re.search(pattern_date, _data)
        if match:
            preprocessed_data.append(_data)
            continue
        
        pattern = r"\[(.*?)\] \[(.*?)\]" # '\n' preprocessing
        match = re.search(pattern, _data)
        if match == None:
            previous_element += ("  " + _data)
        else:
            preprocessed_data.append(previous_element)
            previous_element = _data

    if previous_element:
        preprocessed_data.append(previous_element)

    return preprocessed_data[1:]


def extract_initial_consonant(char: str) -> str:
    """
    Initial consonant extractor

    Parameters:
        char : last letter of speaker
                
    Returns:
        initial consonant : If it is Korean, the initial consonant is returned. Otherwise, the input is returned as it is.
    """
    if '가' <= char <= '힣':
        initial_consonants = [
            'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]
        initial_consonant_index = (ord(char) - ord('가')) // 588
        initial_consonant = initial_consonants[initial_consonant_index]
    else:
        initial_consonant = char
    return initial_consonant


def postprocessing(speaker: str, dialogue: str) -> str:
    """
    Postprocessor for load_kakao_data

    Parameters:
        speaker : kakaotalk id(user name)
        dialogue : original dialogue
                
    Returns:
        postprocessed dialogue
    """
    particile = '' # Attach_particle
    initial_consonant = extract_initial_consonant(speaker[-1])
    if initial_consonant in ['ㄱ', 'ㄴ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅈ', 'ㅊ', 'ㅌ', 'ㅎ']:
        particile = '이'
    else:
        particile = '가'
    return speaker + particile + " " + dialogue + "라고 말했다."
                

def load_kakao_data(load_mode: int, test_mode: bool) -> Tuple[List[str], List[List[str]], List[List[str]], List[str], List[List[str]]]:
    """
    Load kakaotalk data. Execute copy and paste of conversation contents in KakaoTalk application

    Parameters:
        load_mode : In the case of load_mode, if the speaker of consecutive speech balloon is the same, 
                    the speech balloons are merged into one speech balloon.
        test_mode : If test_mode, output the first element of the dataset as an example
    Returns:
        Tuple of ids, participantIDs, dialogues, times, summaries 
    """
    ids = []
    dialogues = []
    times = []
    summaries = []
    participantIDs = []
    
    data = []
    user_input = ""
    print("Enter a value (enter -1 to stop): ")

    while user_input != "-1":
        user_input = input()
        data.append(user_input)
    
    day = -1
    # print(day)
    _participantIDs = []
    _times = []
    utts = []


    data.pop() # Remove -1
    
    data = preprocessing(data)

    if load_mode == 1:
        prev_speaker_id = None
        prev_line = ""
        utts = []
        _times = []
        _participantIDs = []
        
        for _data in data:
            pattern_date = r"\d+년 \d+월 \d+일" # In the case of date display text, increase day by 1 and add 24 hours as much as the date has changed.
            match = re.search(pattern_date, _data)
            if match:
                day += 1
                continue
            matches = re.findall(r'\[(.*?)\]', _data)
            utterance = _data[len(matches[0])+len(matches[1])+6:]

            if matches[0] == prev_speaker_id:
                prev_line += " " + utterance
            else:
                if prev_line:
                    utts.append(postprocessing(prev_speaker_id, prev_line))

                    time_string = matches[1].split()
                    hour, minute = map(int, time_string[1].split(':'))
                    if time_string[0] == '오후': # Convert the time to 24-hour format if it is in the afternoon
                        hour += 12
                    if (time_string[0] == '오전') and (hour == 12):
                        hour = 24
                    if day != -1:
                        hour += day*24
                    
                    _times.append(f'{hour:02d}:{minute:02d}')

                    _participantIDs.append(prev_speaker_id )
                prev_line = utterance
                prev_speaker_id = matches[0]
        if prev_line:
            utts.append(postprocessing(matches[0], prev_line))
            
            time_string = matches[1].split()
            hour, minute = map(int, time_string[1].split(':'))
            if time_string[0] == '오후': # Convert the time to 24-hour format if it is in the afternoon
                hour += 12
            if (time_string[0] == '오전') and (hour == 12):
                hour = 24
            if day != -1:
                hour += day*24
            # print(day, hour, minute)
            _times.append(f'{hour:02d}:{minute:02d}')
            
            _participantIDs.append(matches[0])
        
    else: # load_mode == 0:
        for _data in data:
            pattern_date = r"\d+년 \d+월 \d+일" # In the case of date display text, increase day by 1 and add 24 hours as much as the date has changed.
            match = re.search(pattern_date, _data)
            if match:
                day += 1
            else:
                continue

            matches = re.findall(r'\[(.*?)\]', _data)
            _participantIDs.append(matches[0])
            
            time_string = matches[1].split()
            hour, minute = map(int, time_string[1].split(':'))
            if time_string[0] == '오후': # Convert the time to 24-hour format if it is in the afternoon
                hour += 12
            if (time_string[0] == '오전') and (hour == 12):
                hour = 24
            if day != -1:
                hour += day*24
            # print(day, hour, minute)
            _times.append(f'{hour:02d}:{minute:02d}')

            utts.append(postprocessing(_participantIDs, _data[len(matches[0])+len(matches[1])+6:]))

    ids.append(-1)
    participantIDs.append(_participantIDs)
    times.append(_times)
    dialogues.append(utts)
    summaries.append("")
    
    if test_mode:
        print("======================================================================================")
        print("The first element of the dataset")
        print("{0:<20} :".format("ids[0]") + str(ids[0]))
        print("{0:<20} :".format("participantIDs[0]") + str(participantIDs[0]))
        print("{0:<20} :".format("times[0]") + str(times[0]))
        print("{0:<20} :".format("dialogues[0]") + str(dialogues[0]))
        print("{0:<20} :".format("summaries[0]") + str(summaries[0]))
        # print(len(dialogues[0]), len(times[0]), len(participantIDs[0]))
        print("======================================================================================")

    return ids, participantIDs, dialogues, times, summaries


def load_json_data(load_mode: int, test_mode: bool, **args) -> Tuple[List[str], List[List[str]], List[str], List[List[str]]]:
    """
    Load dialogue summarization dataset json files of https://aihub.or.kr/aidata/30714

    Parameters:
        load_mode : In the case of load_mode, if the speaker of consecutive speech balloon is the same, 
                    the speech balloons are merged into one speech balloon.
        test_mode : If test_mode, output the first element of the dataset as an example
        path : path of json file
    Returns:
        Result of file, which is a tuple of ids, participantIDs, dialogues, times, summaries   
    """
    path = args['path']
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)

    ids = []
    dialogues = []
    times = []
    summaries = []
    participantIDs = []
    
    for _data in data["data"]:
        ids.append(_data["header"]["dialogueInfo"]["dialogueID"])

        prev_speaker_id = None
        prev_line = ""
        utts = []
        _times = []
        _participantIDs = []
        if load_mode == 1:
            for i, dialogue in enumerate(_data["body"]["dialogue"]):
                utterance = dialogue["utterance"].strip()

                if dialogue["participantID"] == prev_speaker_id:
                    prev_line += " " + utterance
                else:
                    if prev_line:
                        utts.append(prev_line)
                        _times.append(_data["body"]["dialogue"][i-1]["time"][0:5]) # Hour:minute:second(00:00:00) -> Hour:minute(00:00)
                        _participantIDs.append(dialogue["participantID"])
                    prev_line = utterance
                    prev_speaker_id = dialogue["participantID"]
            if prev_line:
                utts.append(prev_line)
                _times.append(dialogue["time"][0:5])
                _participantIDs.append(dialogue["participantID"])
        else: # load_mode == 0:
            for dialogue in _data["body"]["dialogue"]:
                utterance = dialogue["utterance"].strip()
                utts.append(utterance)
                _times.append(dialogue["time"][0:5])
                _participantIDs.append(dialogue["participantID"])


        times.append(_times)
        dialogues.append(utts)
        participantIDs.append(_participantIDs)
        summaries.append(_data["body"].get("summary"))

    if test_mode:
        print("======================================================================================")
        print("The first element of the dataset")
        print("{0:<20} :".format("ids[0]") + str(ids[0]))
        print("{0:<20} :".format("participantIDs[0]") + str(participantIDs[0]))
        print("{0:<20} :".format("times[0]") + str(times[0]))
        print("{0:<20} :".format("dialogues[0]") + str(dialogues[0]))
        print("{0:<20} :".format("summaries[0]") + str(summaries[0]))
        # print(len(dialogues[0]), len(times[0]), len(participantIDs[0]))
        print("======================================================================================")
    return ids, participantIDs, dialogues, times, summaries

# Test
# data1 = data_load(source="aihub", load_mode=0, test_mode=True, path = "/root/data_AIhub/Training/[라벨]한국어대화요약_train/개인및관계.json")
# data1 = data_load(source="aihub", load_mode=1, test_mode=True, path = "/root/data_AIhub/Training/[라벨]한국어대화요약_train/개인및관계.json")
# data2 = data_load(source="kakao", load_mode=0, test_mode=True)
# data2 = data_load(source="kakao", load_mode=1, test_mode=True)
# data2 = data_load(source="kakao", load_mode=1, test_mode=True)
"""
Input :
[고대 19 류희철] [오후 1:17] 난 다 ㄱㄴ
[고대 19 류희철] [오후 1:17] 좋음
[손혜강] [오후 1:20] 나도 둘다 가능
[고대 19 류희철] [오후 1:35] 그럼 정보관 ㄱ?
[고대 19 최진우] [오후 2:01] 저번처럼 ㄱㄱ
[고대 19 최진우] [오후 2:02] ㅋㅋㅋㅋ
[손혜강] [오후 2:03] ㅇㅋ
[고대 19 최진우] [오후 7:59] 오신분??
[손혜강] [오후 8:00] 지난주 그 룸에 도착
[손혜강] [오후 8:00] 방금요
-1

Output
ids[0]               :-1
participantIDs[0]    :['고대 19 류희철', '고대 19 류희철', '손혜강', '고대 19 류희철', '고대 19 최진우', '고대 19 최진우', '손혜강', '고대 19 최진우', '손혜강', '손혜강']
times[0]             :['13:17', '13:17', '13:20', '13:35', '14:01', '14:02', '14:03', '19:59', '20:00', '20:00']
dialogues[0]         :['난 다 ㄱㄴ', '좋음', '나도 둘다 가능', '그럼 정보관 ㄱ?', '저번처럼 ㄱㄱ', 'ㅋㅋㅋㅋ', 'ㅇㅋ', '오신분??', '지난주 그 룸에 도착', '방금요']
summaries[0]         :

ids[0]               :-1
participantIDs[0]    :['고대 19 류희철', '손혜강', '고대 19 류희철', '고대 19 최진우', '손혜강', '고대 19 최진우', '손혜강']
times[0]             :['13:20', '13:35', '14:01', '14:03', '19:59', '20:00', '20:00']
dialogues[0]         :['난 다 ㄱㄴ 좋음', '나도 둘다 가능', '그럼 정보관 ㄱ?', '저번처럼 ㄱㄱ ㅋㅋㅋㅋ', 'ㅇㅋ', '오신분??', '지난주 그 룸에 도착 방금요']
summaries[0]         :
"""


"""
Issue 1. day 처리
[손혜강] [오후 10:49] 이모티콘
2023년 5월 17일 수요일
[고대 19 최진우] [오전 10:21] 지금 좀 돌려보니까
=> +24 시간 되도록

Issue 2. \t, \n 처리
input
[고대 19 최진우] [오전 10:23] 말풍선 내부에 \n들어가있으면 \n지우면 될듯!
[고대 19 최진우] [오전 10:24] 그리고

meaningful한 문장의 기준이 조금 애매해서..
의미가 없다면 어떤 thread에 합쳐져도 summary text를 건들지 않지 않을까? 그리고 thread마다 다른 색깔로 칠해서 보여줘야하는데 의미가 없는 문장은 아예 색깔을 안 칠하는 것도 이상함
[손혜강] [오후 10:49] 이모티콘
2023년 5월 17일 수요일
[고대 19 최진우] [오전 10:21] 지금 좀 돌려보니까
2023년 5월 18일 수요일
[고대 19 손혜강] [오전 9:31] ㅇㅋ

Issue 3.
발화자가 ~라고 말했다.
지금 좀 돌려보니까
최진우가 지금 좀 돌려보니까라고 말했다.

Issue 4.
대 19 최진우] [오전 10:23] 말풍선 내부에 \n들어가있으면 \n지우면 될듯!
[고대 19 최진우] [오전 10:24] 그리고

meaningful한 문장의 기준이 조금 애매해서..
의미가 없다면 어떤 thread에 합쳐져도 
=> 말풍선 내부에 \n들어가있으면 \n지우면 될듯! 그리고 
말풍선 내부에 \n들어가있으면 \n지우면 될듯! [SEP] 그리고
"""