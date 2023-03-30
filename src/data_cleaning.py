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

def remove_prefix(text, prefix):
    if(text.startswith(prefix)):
        text = text[1:]
    if(text.endswith(prefix)):
        text = text[:len(text) - 1]
    return text

## mental FAQ in https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot?resource=download
mental_faq = pd.read_csv('./data/hugging_face/Mental_Health_FAQ.csv')
mental_faq = mental_faq[['Questions','Answers']]
mental_faq['tag'] = 'general'
print('mental_faq has shape', mental_faq.shape)


## mental conversation data from https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data
mental_intend= open('./data/hugging_face/intents.json')

mental_intend = json.load(mental_intend)['intents']
mental_intend = [ process_intend_pair(i) for i in mental_intend]
mental_intend = [item for sublist in mental_intend for item in sublist]

mental_intend = pd.DataFrame.from_dict(mental_intend, orient='columns')
mental_intend.columns = ['tag','Questions','Answers']

print('mental_intend has shape', mental_intend.shape)



anxiety_gpt = process_gpt_file('./data/gpt/anxiety_gpt.json')
depression_gpt = process_gpt_file('./data/gpt/depression_gpt.json')
mental_sadness_gpt = process_gpt_file('./data/gpt/mental_sadness_gpt.json')

print('anxiety_gpt has shape', anxiety_gpt.shape)
print('depression_gpt has shape', depression_gpt.shape)
print('mental_sadness_gpt has shape', mental_sadness_gpt.shape)



## second round dataset

manual_data = pd.read_excel('./data/second_round/Q&A_manually_collected.xlsx', sheet_name='工作表1')
manual_data = manual_data[manual_data['Answer'].notna()]
manual_data['tag'] = 'manual'
manual_data = manual_data[['tag','Question','Answer']]
manual_data.columns = ['tag','Questions','Answers']

print('manually clooected data has shape', manual_data.shape)



chatgpt_second_round = pd.read_excel('./data/second_round/Seperate_Answer_query.xlsx', sheet_name='Sheet1')
chatgpt_second_round = chatgpt_second_round[chatgpt_second_round['Answer'].notna()]
chatgpt_second_round['Question'] = [ remove_prefix(i, '"') for i in chatgpt_second_round['Question'] ]
chatgpt_second_round['Answer'] = [ remove_prefix(i, ')').replace('\n','') for i in chatgpt_second_round['Answer'] ]
chatgpt_second_round['tag'] = 'chatgpt'
chatgpt_second_round = chatgpt_second_round[['tag','Question','Answer']]
chatgpt_second_round.columns = ['tag','Questions','Answers']

print('chat gpt second round has shape', chatgpt_second_round.shape)


mental_data = pd.concat([mental_intend, mental_faq, anxiety_gpt, depression_gpt, mental_sadness_gpt, manual_data, chatgpt_second_round])
print('final data without dedup', mental_data.shape)
mental_data = mental_data.drop_duplicates()
print('final data with dedup', mental_data.shape)


mental_data.to_csv('./data/mental_question_pair_collections_second_round.csv', index = False, sep = '\t')