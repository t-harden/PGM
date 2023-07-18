# coding = utf-8

import pickle
import time
import openai
import json

openai.api_key = "your OpenAI api key"


def make_spec(spec):  # extract specifications

    spec = spec[spec.index("{"):]
    spec = json.loads(spec)
    title = spec["Title"]
    inputs = spec["Inputs"]
    outputs = spec["Outputs"]
    activities = spec["Activities"]
    result = [title, inputs, outputs, activities]

    return result


if __name__ == '__main__':

    '''read data'''
    query_list = [10, 1014, 1097, 113, 1189, 1201]
    with open('../data/dict_BW.pkl', 'rb') as f:
        dict_BW = pickle.load(f)
    # read planning prompting
    with open('../data/PGM_FewShot_stage1.txt', 'r') as file:
        FewShot_stage1 = file.read()
    # read spec-generation prompting
    with open('../data/PGM_FewShot_stage2.txt', 'r') as file:
        FewShot_stage2 = file.read()



    '''main component'''
    dict_plan = {}
    dict_spec = {}
    for query_key in query_list:
        query_text = dict_BW[query_key][1]
        print("---", query_key, "---")
        print("query_text: ", query_text)

        "--stage1: generate plan--"
        stage1_prompt = FewShot_stage1.replace('!!!', query_text)
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": stage1_prompt}
          ],
          temperature=0,
          n=1
        )
        plan = completion.choices[0].message["content"]
        dict_plan[query_key] = plan


        "--stage2: generate structured specification--"
        temp_str = FewShot_stage2.replace('!!!', query_text)
        phase2_prompt = temp_str.replace('###', plan)
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": phase2_prompt}
          ],
          temperature=0,
          n=1
        )
        spec = completion.choices[0].message["content"]
        temp_spec = make_spec(spec)
        temp_spec.insert(1, query_text)
        dict_spec[query_key] = temp_spec

        time.sleep(40) # wait for access restriction

    '''save results'''
    with open('dict_plan.pkl', 'wb') as f:
        pickle.dump(dict_plan, f, pickle.HIGHEST_PROTOCOL)
    with open('dict_spec.pkl', 'wb') as f:
        pickle.dump(dict_spec, f, pickle.HIGHEST_PROTOCOL)


