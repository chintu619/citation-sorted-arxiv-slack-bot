import feedparser
import pandas
import requests
import datetime as dt
from grabURL import grabUrls

_SSLVERIFYCRT = True # '../ArxivSorting/verify/ZscalerRootCertificate.crt'
arXiv_url = 'http://export.arxiv.org/api/query?search_query='
query='(cat:stat.ML+OR+cat:cs.CV+OR+cat:cs.LG)&start=0&max_results=2000' + \
      '&sortBy=lastUpdatedDate&sortOrder=descending'
arXiv_url = arXiv_url+query
data = requests.get(arXiv_url,verify=_SSLVERIFYCRT).text
entry_elements = feedparser.parse(data)['entries']

prev_month = (dt.date.today()-dt.timedelta(days=31)).strftime('%y%m')

posted_ids = pandas.read_csv('./posted_ids.csv')
posted_urls = list(posted_ids['url'])
counter = 0
for val in entry_elements:
    paper_release_month = val['id'].split('abs/')[-1].split('.')[0]
    if (val['id'][:-2] not in posted_urls) and (paper_release_month >= prev_month):
        paper_title = ''.join(val['title'].split('\n '))
        paper_abstr = ' '.join(val['summary'].split('\n'))
        if 'arxiv_comment' in val: paper_comment = ' '.join(val['arxiv_comment'].split('\n'))
        else: paper_comment = ''
        paper_alt_url = grabUrls(paper_abstr + ' ' + paper_comment)
        counter += 1
        if paper_alt_url: print('Paper',counter,paper_alt_url[0]+' '+paper_title)
        else: print('Paper',counter,paper_title)
# print('No. of new papers:', counter)