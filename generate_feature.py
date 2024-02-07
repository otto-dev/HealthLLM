import openai


def generate_feature(path):
    my_feature_list = []
    related_work = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            my_feature_list.append(line)
    question = []
    for i in my_feature_list:
        openai.api_key = "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"
        messages = []
        system_message = "please list the names of the symptoms in order like these examples\n" + 'Cold ------ Runny ' \
                                                                                                  'nose, stuffy nose, ' \
                                                                                                  'sneezing, ' \
                                                                                                  'sore throat, ' \
                                                                                                  'cough, hoarseness, ' \
                                                                                                  'headache, ' \
                                                                                                  'sore eyes, ' \
                                                                                                  'fatigue, ' \
                                                                                                  'minor body aches, ' \
                                                                                                  'fever or low-grade ' \
                                                                                                  'fever, ear pain, ' \
                                                                                                  'chest tightness, ' \
                                                                                                  'or difficulty ' \
                                                                                                  'breathing.'
        messages.append({"role": "system", "content": system_message})
        message = f'Please list the symptoms of {i}? in the following format'
        messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response["choices"][0]["message"]["content"]
        question.append(reply)
        print(reply)

    return question


if __name__ == '__main__':
    OPENAI_API_KEY = "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"
    path = '/Users/chongzhang/PycharmProjects/ai_for_health_final/label and feature/output_target.txt'
    question = generate_feature(path)
    # with open("/Users/jmy/Desktop/ai_for_health_final/training/feature.txt", "w") as file:
    #     for item in question:
    #         file.write("%s\n" % item)
