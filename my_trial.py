"""動作確認ファイル
"""
import math
import random
import itertools
import collections
from pprint import pprint

def dict_combine():
    actions = ['A','B','X','Y','Z','R','L','ZR','ZL','left','up','right','down']
    states = list(range(10))

    result = {}

    for a, s in itertools.product(actions, states):
        result[a,s] = random.randint(a=0, b=9)

    pprint(result)


if __name__ == '__main__':
    dict_combine()