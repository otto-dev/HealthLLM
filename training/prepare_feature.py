import re

words = []

with open('feature.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if re.match(r'^\d+', line):
            text_after_number = re.sub(r'^\d+\s*', '', line)
            words.extend(text_after_number.split('、'))
        else:
            words.extend(line.split('、'))

processed_lst = []

for item in words:
    if item.startswith('.'):
        text_without_spaces = item[1:].replace(' ', '')
        processed_lst.append(text_without_spaces)
    else:
        # print(item)
        symptoms_list = [symptom.strip() for symptom in item.split(',')]
        for i in symptoms_list:
            processed_lst.append(i)
print(processed_lst)

with open("/Users/chongzhang/PycharmProjects/ai_for_health_final/training/feature_update.txt", "w") as file:
    for item in processed_lst:
        file.write("%s\n" % item)