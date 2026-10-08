import re
from Lab01 import Lab01

with open('./Oblomov.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()

words = re.findall(r'\w+(?:-\w+)*|[.,!?]', text)

words_test_split = words[int(len(words)*0.9):]
words_train_split = words[:int(len(words)*0.9)]

print(len(words_test_split))
print(len(words_train_split))

lab01 = Lab01()

print(words[:20])
lab01.fit(words[:20])

print(lab01.prob('', 'глава'))






