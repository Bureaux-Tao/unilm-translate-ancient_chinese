import json

import pandas as pd

source = []
target = []
added = []
with open("./data/all.jsonl", encoding = 'utf-8') as f:
    i = 0
    for l in f:
        l = json.loads(l)
        if len(l['source'].strip()) + len(l['target'].strip())+3 <= 256:
            i += 1
        source.append(len(l['source'].strip()))
        target.append(len(l['target'].strip()))
        added.append(len(l['source'].strip()) + len(l['target'].strip()))
s = pd.DataFrame(source)
t = pd.DataFrame(target)
a = pd.DataFrame(added)
print(s.describe())
print(t.describe())
print(a.describe())
print(i / a.count())
print(a.count() - i)
