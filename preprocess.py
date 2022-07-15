import json
import os
from random import shuffle


def get_list(path):
    r = []
    with open(path, 'r', encoding = 'utf-8') as lunyu1:
        lunyu1_json = json.load(lunyu1)
        for i in lunyu1_json:
            for j in i['contents']:
                r.append(j)
        return r


l = get_list("./data/lunyu/part1.json") + \
    get_list("./data/shiji/part1.json") + \
    get_list("./data/shiji/part2.json") + \
    get_list("./data/zhongyong/part1.json") + \
    get_list("./data/zhongyong/part1.json") + \
    get_list("./data/zhongyong/part2.json")

shuffle(l)

if os.path.exists("./data/all.jsonl"):
    # 存在，则删除文件
    os.remove("./data/all.jsonl")
with open("./data/all.jsonl", 'a', encoding = 'utf-8') as all:
    for i in l:
        all.write(json.dumps(i, ensure_ascii = False) + "\n")

if os.path.exists("./data/train.jsonl"):
    # 存在，则删除文件
    os.remove("./data/train.jsonl")
with open("./data/train.jsonl", 'a', encoding = 'utf-8') as train:
    for i, item in enumerate(l):
        if i <= int(len(l) * 0.9):
            train.write(json.dumps(item, ensure_ascii = False) + "\n")

if os.path.exists("./data/valid.jsonl"):
    # 存在，则删除文件
    os.remove("./data/valid.jsonl")
with open("./data/valid.jsonl", 'a', encoding = 'utf-8') as valid:
    for i, item in enumerate(l):
        if i > int(len(l) * 0.9):
            valid.write(json.dumps(item, ensure_ascii = False) + "\n")
