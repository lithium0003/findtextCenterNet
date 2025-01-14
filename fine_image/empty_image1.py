#!/usr/bin/env python3

import sys
from PIL import Image, ImageDraw
import json

if len(sys.argv) < 2:
    print(sys.argv[0],'target.png')
    exit(1)

target_file = sys.argv[1]

with open(target_file+'.json', 'r', encoding='utf-8') as file:
    out_dict = json.load(file)

out_dict['textbox'] = []

with open(target_file+'.json', 'w', encoding='utf-8') as file:
    json.dump(out_dict, file, indent=2, ensure_ascii=False)

linesfile = target_file + '.lines.png'
sepsfile = target_file + '.seps.png'

lines_all = Image.open(linesfile)
draw = ImageDraw.Draw(lines_all)
draw.rectangle((0,0,lines_all.width,lines_all.height), fill=0)
lines_all.save(linesfile)

lines_all = Image.open(sepsfile)
draw = ImageDraw.Draw(lines_all)
draw.rectangle((0,0,lines_all.width,lines_all.height), fill=0)
lines_all.save(sepsfile)
