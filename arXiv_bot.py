# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:26:45 2019

@author: Chaitanya Narisetty
"""

import os,requests,pandas,json,feedparser
import scholarly
import numpy as np
import time
import datetime as dt

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
    posted_urls = list(posted_ids['url'])
    while True:
        counter = 0
        arXiv_url = 'http://export.arxiv.org/api/query?search_query='
        query='(cat:stat.ML+OR+cat:cs.CV+OR+cat:cs.LG)&start=0&max_results=1000' + \
              '&sortBy=lastUpdatedDate&sortOrder=descending'
        arXiv_url = arXiv_url+query
        data = requests.get(arXiv_url,verify=verify).text
        entry_elements = feedparser.parse(data)['entries']

        # WARNING: more than 100 new papers per day can lead to temporary IP ban
        flag_IPban = False
        all_paper_info = []; all_texts = []; paper_citation_count = []
        for val in entry_elements:
            paper_info = {}
            paper_info['url']= val['id']
            paper_info['title'] = ''.join(val['title'].split('\n '))
            paper_info['abstract'] = ' '.join(val['summary'].split('\n'))
            paper_info['release_date'] = val['published']
            paper_info['date'] = dt.date.today().strftime('%Y-%m-%d')

            if (paper_info['url'] not in posted_urls) and (paper_info['url'][-2:] == 'v1'):

                if not flag_IPban:
                    # sleep for 15 minutes after checking of every 50 papers to avoid IP ban
                    if (counter > 0) and (counter%50 == 0): time.sleep(15*60)

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

                paper_citation_count.append(citations)
                counter += 1; print('Paper:', counter, 'Citations:', citations)

                if citations is np.nan:
                    all_texts.append(('Title:*UNRANKED! - {title}*\nDate:{release_date}\nAbstract' + \
                                    ':```{abstract}```\n{url}').format(**paper_info))
                    title_onset = 'UNRANKED! - '
                else:
                    all_texts.append(('Title:*{title}*\nDate:{release_date}\nAbstract' + \
                                    ':```{abstract}```\n{url}').format(**paper_info))
                    title_onset = ''

                release_date = paper_info['release_date'].split('T')[0]
                title = title_onset + paper_info['title']
                all_paper_info.append([paper_info['url'],paper_info['date'],citations,title,release_date])

        # rank the papers according to their authors' citation counts
        rankings = np.argsort(paper_citation_count)[::-1]
        all_texts_sorted = [all_texts[rank] for rank in rankings]
        all_paper_info = pandas.DataFrame(all_paper_info, columns=['url','date','weight','title', 'release_date'])
        posted_ids = pandas.concat((all_paper_info, posted_ids), ignore_index=True)
        Texts.extend(all_texts_sorted)

        return posted_ids, Texts

def post(Texts,Web_hook_url,verify):
    post_text = ''; counter = 1
    for val in Texts:
        # only use title and url information
        title = val.split('\n')[0][7:-1] # remove first 6 letters (title:*) and last letter (*)
        url = 'http://arxiv.org/abs/' + val.split('http://arxiv.org/abs/')[-1].split('\n')[0]
        post_text += str(counter) + '. ' + title + ' - ' + url + '\n'
        counter += 1
    post = {'text':post_text,
            'username':u'Bot',
            'icon_emoji':u':thinking_face:',
            'unfurl_links':False,
            'link_names':1,
            'channel': '#misc-news-arxiv',}
    data = json.dumps(post).encode("utf-8")
    requests.post(Web_hook_url, data = data,verify=verify)

def makeMarkdown(posted_ids):
    if len(posted_ids) == 0: return 0

    lastRelease_date = max(posted_ids.date)
    lastRelease_dt = dt.datetime.strptime(lastRelease_date, '%Y-%m-%d')
    lastRelease_papers = posted_ids.loc[posted_ids['date'] == lastRelease_date].reset_index(drop=1)

    oneweekpastdate = (lastRelease_dt - dt.timedelta(7)).strftime('%Y-%m-%d')
    lastweek_papers = posted_ids.loc[(posted_ids['date'] <= lastRelease_date) & \
                                     (posted_ids['date'] > oneweekpastdate)]
    lastweek_papers = lastweek_papers.sort_values(by='weight', ascending=False \
                                                 ).reset_index(drop=1)

    with open('README.md', mode='r') as md_file:
        markdown_prev = ''.join(md_file.readlines())

    before_daily = markdown_prev.split("## Daily top 10\n")[0]
    after_weekly = markdown_prev.split("## Weekly top 20\n")[1].split('</details>\n')
    after_weekly = '</details>\n'.join(after_weekly[1:])

    markdown = before_daily + "## Daily top 10\n"
    for itr in range(len(lastRelease_papers)):
        paper = lastRelease_papers.loc[itr]
        if itr == 10:
            markdown += "<details><summary>today's remaining papers</summary>\n" + \
                        "  <ol start=11>\n"
        if itr < 10:
            markdown += str(itr+1)+ '. *'+paper['title']+'* [url]('+paper['url']+')\n'
        else:
            markdown += '    <li><i>'+paper['title']+'</i> <a href="'+\
                                      paper['url']+'">url</a></li>\n'

    if len(lastRelease_papers) > 10: markdown += "  </ol>\n</details>\n\n" + "## Weekly top 20\n"
    else: markdown += "<details><summary>today's remaining papers</summary>\n" + \
                        "  <ol start=11>\n" + "  </ol>\n</details>\n\n" + "## Weekly top 20\n"

    for itr in range(len(lastweek_papers)):
        paper = lastweek_papers.loc[itr]
        if itr == 20:
            markdown += "<details><summary>this week's remaining papers</summary>\n" + \
                        "  <ol start=21>\n"
        if itr < 20:
            markdown += str(itr+1)+ '. *'+paper['title']+'* [url]('+paper['url']+')\n'
        else:
            markdown += '    <li><i>'+paper['title']+'</i> <a href="'+\
                                      paper['url']+'">url</a></li>\n'

    if len(lastweek_papers) > 20: markdown += "  </ol>\n</details>\n" + after_weekly
    else: markdown += "<details><summary>this week's remaining papers</summary>\n" + \
                        "  <ol start=21>\n" + "  </ol>\n</details>\n" + after_weekly

    with open('README.md', mode='w', encoding="utf-8") as md_file: md_file.write(markdown)

if __name__== '__main__':
    verify=True # or provide a link to appropriate ssl crt
    Web_hook_url = "Web hook url" # provide an appropriate web hook
    posted_ids_file = './posted_ids.csv'
    if os.path.exists(posted_ids_file):
        posted_ids = pandas.read_csv(posted_ids_file)
    else:
        posted_ids = pandas.DataFrame([], columns=['url', 'date', 'weight', 'title', 'release_date'])

    Texts = []
    posted_ids, Texts = search(posted_ids,Texts,verify)
    try:
        post(Texts,Web_hook_url,verify)
    except:
        print('Error: Unable to post. Check the web-hook URL!')
    posted_ids = posted_ids.sort_values(by=['date','weight'], ascending=False).reset_index(drop=1)
    posted_ids.to_csv(posted_ids_file, index=False)
    makeMarkdown(posted_ids)

    # # auto-update git repo
    # try:
    #     if os.popen('git diff README.md posted_ids.csv').read():
    #         print(os.popen('git add README.md posted_ids.csv').read())
    #         #print(os.popen('git status').read())
    #         print(os.popen('git commit -m "Updated to today\'s feed."').read())
    #         print(os.popen('git push').read())
    # except: print('Please check git master status.')
