import numpy as np
import datetime
import json
import requests
import csv

def re_label(src, c_label):
    for i, c in enumerate(c_label):
        src[src[..., -1] == c, -1] = i
    return src, np.unique(src[..., -1]).astype(int)


def log(path, *args):
    msg = ' '.join(map(str, [datetime.datetime.now(), '>'] + list(args)))
    # print(msg)
    with open('{0}/log.txt'.format(path), 'at') as fd: fd.write(msg + '\n')


def write_slack(user_name, s):
    with open('./slack_token.txt', 'r') as f:
        slack_token = f.read()
    requests.post(slack_token, data = json.dumps({'text':s,'username':user_name,'icon_emoji':':smile:','link_names':1,}))


def output_condition(path, *args):
    dicts = {}
    for i in args:
        dicts.update(vars(i))
    print(dicts)
    print('\n-----output condition-----\n')
    with open('{0}/condition.csv'.format(path), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in dicts.keys():
            writer.writerow(['{0}: {1}'.format(i, dicts[i])])
    return writer