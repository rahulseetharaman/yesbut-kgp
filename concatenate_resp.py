import os 
import json
import jsonlines

answers = []

for f in os.listdir('yesbut_sg/combined/gpt4o'):
    line = json.load(open(f'yesbut_sg/combined/gpt4o/{f}'))
    answers.append(line)

with jsonlines.open("yesbut_sg_gpt4o_0shot.json", "w") as writer:
    for a in answers:
        a['overall_description'] = a['ground_truth']
        a['response'] = a['final_answer']
        writer.write(a)
    
