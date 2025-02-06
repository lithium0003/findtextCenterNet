import json
import glob
import os

data_path = 'train_data2'

jsonfiles = sorted(glob.glob(os.path.join(data_path, '*.json')))

for jsonfile in jsonfiles:
    with open(jsonfile, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for i, pos in enumerate(data['textbox']):
        text = pos['text']
        if text is None:
            continue

        if len(text) > 1:
            c = text.encode('utf-32-be')
            t = c[:4].decode('utf-32-be')
            data['textbox'][i]['text'] = t
    
    with open(jsonfile, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
