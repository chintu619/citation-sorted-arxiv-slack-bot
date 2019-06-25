# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:26:45 2019

@author: 0000011331759
"""

import os,requests,pandas,json,feedparser
import scholarly
import numpy as np
import time

def getAuthorCitations(author):
#    print(author)
    # check if author has initials in name
    author_name_without_initials = author.split(' ')[0] + ' ' + author.split(' ')[-1]
    if author != author_name_without_initials:
        flag_authorHasInitialsInName = True
    else: flag_authorHasInitialsInName = False

    search_query = scholarly.search_author(author)
    try: # check if author is listed in google scholar
        author_stats = next(search_query)
        try: # if multiple people exist with same name, skip author
            next(search_query); author_count = 2
        except: author_count = 1
    except:
        author_count = 0

    # if author is uniquely identified in scholar, get citation count
    if author_count == 1:
        try: author_citation_count = author_stats.citedby
        except: author_citation_count = 0
    elif (author_count == 0) and flag_authorHasInitialsInName:
        author_citation_count = getAuthorCitations(author_name_without_initials)
    else:
        author_citation_count = 0

    return author_citation_count

def search(posted_ids,Texts,verify):
    while True:
        counter = 0
        arXiv_url = 'http://export.arxiv.org/api/query?search_query='
        query='(cat:stat.ML+OR+cat:cs.CV)&start=0&max_results=100&sortBy=lastUpdatedDate&sortOrder=descending'
        arXiv_url = arXiv_url+query
        data = requests.get(arXiv_url,verify=verify).text
        entry_elements = feedparser.parse(data)['entries']

        # WARNING: more than 100 papers per day can lead to temporary IP ban
        flag_IPban = False
        all_paper_info = []; all_texts = []; paper_citation_count = []
        for val in entry_elements:
            paper_info = {}
            paper_info['url']= val['id']
            paper_info['title'] = val['title']
            paper_info['abstract'] = val['summary']
            paper_info['date'] = val['published']

            if paper_info['url'] not in posted_ids:
                if not flag_IPban:
                    citations = 0 # total citation count of the paper
                    # get a list of all the authors in the paper
                    author_list = [author['name'] for author in val['authors']]
                    for author in author_list: # for each author, get citation count
                        try:
                            citations += getAuthorCitations(author) # update total citation count
                        except:
                            #time.sleep(2)
                            citations = np.nan
                            print('Could not get citations of ' + author)
    #                        raise
                            flag_IPban = True
                            break

                all_paper_info.append(paper_info['url'])
                paper_citation_count.append(citations)
                counter += 1; print('Paper:', counter, 'Citations:', citations)
                if citations is np.nan:
                    all_texts.append(('Title:*UNRANKED! - {title}*\nDate:{date}\nAbstract' + \
                                    ':```{abstract}```\n{url}').format(**paper_info))
                else:
                    all_texts.append(('Title:*{title}*\nDate:{date}\nAbstract' + \
                                    ':```{abstract}```\n{url}').format(**paper_info))

        # rank the papers according to their authors' citation counts
        rankings = np.argsort(paper_citation_count)[::-1]
        all_paper_info_sorted = [all_paper_info[rank] for rank in rankings]
        all_texts_sorted = [all_texts[rank] for rank in rankings]
        posted_ids.extend(all_paper_info_sorted)
        Texts.extend(all_texts_sorted)

        return 0

def post(Texts,Web_hook_url,verify):
    for val in Texts:
        post = {'text':val,
                'username':u'Bot',
                'icon_emoji':u':thinking_face:',
                'unfurl_links':True,
                'link_names':1,}
        data = json.dumps(post).encode("utf-8")
        requests.post(Web_hook_url, data = data,verify=verify)

if __name__== '__main__':
    verify='./verify/ZscalerRootCertificate.crt'
    Web_hook_url = "Web hook url"
    posted_ids_file = './posted_ids.csv'
    if os.path.exists(posted_ids_file):
        posted_ids = pandas.read_csv(posted_ids_file,names=['ids'])
        posted_ids = list(posted_ids['ids'])
    else:
        posted_ids = []

    Texts = []
    search(posted_ids,Texts,verify)
    try:
        post(Texts,Web_hook_url,verify)
    except:
        print('Error: Unable to post. Check the web-hook URL!')
    df = pandas.DataFrame(posted_ids)
    df.to_csv(posted_ids_file,header=False,index=False)
