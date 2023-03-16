import pandas as pd
import json
import itertools

## mental FAQ in https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot?resource=download
mental_faq = pd.read_csv('./data/Mental_Health_FAQ.csv')
mental_faq = mental_faq[['Questions','Answers']]
mental_faq['tag'] = 'general'


def process_intend_pair(mental_intend_dict):
    tag, patterns, responses = mental_intend_dict['tag'], mental_intend_dict['patterns'], mental_intend_dict['responses']
    possible_pair = list(itertools.product(patterns, responses))
    pair = [ {'tag':tag, 'patterns':one_pair[0], 'responses':one_pair[1]} for one_pair in possible_pair]
    return(pair)


## mental conversation data from https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data
mental_intend= open('./data/intents.json')

mental_intend = json.load(mental_intend)['intents']
mental_intend = [ process_intend_pair(i) for i in mental_intend]
mental_intend = [item for sublist in mental_intend for item in sublist]

mental_intend = pd.DataFrame.from_dict(mental_intend, orient='columns')
mental_intend.columns = ['tag','Questions','Answers']

mental_data = pd.concat([mental_intend, mental_faq])
print(mental_data.shape)
mental_data = mental_data.drop_duplicates()
print(mental_data.shape)


mental_data.to_json('./data/hugging_face_collected_qapair.json', orient= 'records')