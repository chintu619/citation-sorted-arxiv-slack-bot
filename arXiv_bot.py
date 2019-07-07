# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:26:45 2019

@author: Chaitanya Narisetty
"""
import os
import pandas
from webutils import sayHello, search, post

if __name__== '__main__':
    verify=True # or provide a link to appropriate ssl crt
    Web_hook_url = "https://hooks.slack.com/services/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" # provide an appropriate web hook
    posted_ids_file = './tracked_ids/cat.csv'
    target_categories = ['cs.CV']#, 'cs.GR', 'stat.ML']
    # post first message
    sayHello(Web_hook_url, verify)
    for category in target_categories:
        posted_ids_file_ = posted_ids_file.replace('cat',category)
        if os.path.exists(posted_ids_file_):
            posted_ids = pandas.read_csv(posted_ids_file_,names=['ids'])
            posted_ids = list(posted_ids['ids'])
        else:
            posted_ids = []
        Texts = []
        search(category, posted_ids, Texts, verify)
        '''
        try:
            post(category,Texts,Web_hook_url,verify)
        except:
            print('Error: Unable to post. Check the web-hook URL!')
        df = pandas.DataFrame(posted_ids)
        df.to_csv(posted_ids_file_,header=False,index=False)
        '''
