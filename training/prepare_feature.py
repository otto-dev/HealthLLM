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

print(words)
processed_lst = []

for item in words:
    if item.startswith('.'):
        processed_lst.append(item[1:])
    else:
        # print(item)
        symptoms_list = [symptom.strip() for symptom in item.split(',')]
        for i in symptoms_list:
            processed_lst.append(i)

print(processed_lst)

text = "Fatigue, weakness, pale skin, shortness of breath, dizziness, headache, cold hands and feet, chest pain, irregular heartbeat, difficulty concentrating, brittle nails, restless legs syndrome"

symptoms_list = [symptom.strip() for symptom in text.split(',')]
print(symptoms_list)
