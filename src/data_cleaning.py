import pandas as pd
import json
import itertools


def process_intend_pair(mental_intend_dict):
    tag, patterns, responses = mental_intend_dict['tag'], mental_intend_dict['patterns'], mental_intend_dict['responses']
    possible_pair = list(itertools.product(patterns, responses))
    pair = [ {'tag':tag, 'patterns':one_pair[0], 'responses':one_pair[1]} for one_pair in possible_pair]
    return(pair)

def process_gpt_file(file_json):
    df_json = json.load(open(file_json))
    df_json = pd.DataFrame.from_dict(df_json, orient='columns')
    return(df_json)

## mental FAQ in https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot?resource=download
mental_faq = pd.read_csv('./data/hugging_face/Mental_Health_FAQ.csv')
mental_faq = mental_faq[['Questions','Answers']]
mental_faq['tag'] = 'general'


## mental conversation data from https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data
mental_intend= open('./data/hugging_face/intents.json')

mental_intend = json.load(mental_intend)['intents']
mental_intend = [ process_intend_pair(i) for i in mental_intend]
mental_intend = [item for sublist in mental_intend for item in sublist]


mental_intend = pd.DataFrame.from_dict(mental_intend, orient='columns')
mental_intend.columns = ['tag','Questions','Answers']



anxiety_gpt = process_gpt_file('./data/gpt/anxiety_gpt.json')
depression_gpt = process_gpt_file('./data/gpt/depression_gpt.json')
mental_sadness_gpt = process_gpt_file('./data/gpt/mental_sadness_gpt.json')


mental_data = pd.concat([mental_intend, mental_faq, anxiety_gpt, depression_gpt, mental_sadness_gpt])
print(mental_data.shape)
mental_data = mental_data.drop_duplicates()
print(mental_data.shape)


mental_data.to_csv('./data/mental_question_pair_collections.csv', index = False)