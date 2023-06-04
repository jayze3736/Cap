import json

# sample 파싱 결과 확인 - json 형태를 대략적으로 파악하기 위해 정의
def print_parse_sample(sample):
    for sentence in sample['sentences']:
        print("id: " + sentence['id'])
        print("origin_text: " + sentence['origin_text'])
        print("text: " + sentence['text'])
        print("intensity_sum: " , sentence['intensity_sum'])
        print()

# origin_text와 text가 다른 경우를 확인
def print_parse_diff_text(json):
    for data in json:
        for sentence in data['sentences']:
            if(sentence['origin_text'] != sentence['text']):
                print("origin_text: " + sentence['origin_text'])
                print("text: " + sentence['text'])
            
       
# intensity 투표 점수에 대한 종합의 고유 값을 모두 출력 
def print_uq_intensity_sum(json):
    intensity_list = []
    for data in json:
        for sentence in data['sentences']:
            intensity_list.append(sentence['intensity_sum'])
    print("unique intensity values")
    print(list(set(intensity_list))) # print unique list element


def parse_to_txt(json, txt_path):
    f = open(txt_path, 'w', encoding='utf-8')

    for data in json:
        for sentence in data['sentences']:
            line = sentence['text'] + "|" + str(sentence['intensity_sum']) + '\n'
            f.write(line)

# main # 
label_train = open("D:/DL_dataset/text_ethic_data/1.Training/LabelData/talksets-train-1_aihub.json", 'r', encoding='utf-8'); # 한글을 읽으려면 UTF8 파라메터 필요
json_label_train = json.load(label_train)
parse_to_txt(json_label_train, "D:/Test_parse.txt")
# print(len(json_label_train)) # json의 총 개수 확인
# print_parse_sample(json_label_train[0]) # sample을 파싱하여 확인
# print_parse_diff_text(json_label_train) # origin_text와 text가 다른 경우 출력
# print_uq_intensity_sum(json_label_train) # intensity sum의 unique value 출력

