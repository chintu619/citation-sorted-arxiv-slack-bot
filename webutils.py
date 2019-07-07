import os,requests,pandas,json,feedparser
import scholarly
import numpy as np
import time
import re
import math

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def sigmoid(count, mid):
    return 1./(1. + math.exp((-count + mid)/mid))

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def getAuthorCitations(author):
#    print(author)
    # check if author has initials in name
    author_name_without_initials = author.split(' ')[0] + ' ' + author.split(' ')[-1]
    if author != author_name_without_initials:
        flag_authorHasInitialsInName = True
    else:
        flag_authorHasInitialsInName = False
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

def getRelevance(seenTopics, sentences):
    score = len(seenTopics)
    relevantTopics = seenTopics.copy()
    return (score, relevantTopics)

def search(category, posted_ids, Texts, verify):
    while True:
        counter = 0
        arXiv_url = 'http://export.arxiv.org/api/query?search_query='
        if not posted_ids:
            query='(cat:{})&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending'.format(category)
        else:
            query='(cat:{})&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending'.format(category)
        arXiv_url = arXiv_url+query
        data = requests.get(arXiv_url,verify=verify).text
        entry_elements = feedparser.parse(data)['entries']
        #print(entry_elements[0]['title'], entry_elements[0]['date'])
        # WARNING: more than 100 papers per day can lead to temporary IP ban
        flag_IPban = False
        all_paper_info = []; all_texts = []; paper_citation_count = []; content_score = []
        for val in entry_elements:
            paper_info = {}
            paper_info['url']= val['id']
            paper_info['title'] = val['title']
            paper_info['abstract'] = val['summary']
            paper_info['date'] = val['published']

            if paper_info['url'] not in posted_ids:
                if not flag_IPban:
                    # citation wrapper

                    citations = 0 # total citation count of the paper
                    valid_count = 0.
                    for author in val['authors']: # for each author, get citation count
                        try:
                            citations += getAuthorCitations(author['name']) # update total citation count
                            valid_count += 1.
                        except:
                            #time.sleep(2)
                            print('Could not get citations of ' + author['name'])
                    # abstract keywords sorting
                    abstract = paper_info['abstract'].lower().replace('-', ' ')
                    abstractSentences = split_into_sentences(abstract)
                    unseenTopics, unseenDataTypes, unseenMethods = getQueryItems(category)
                    seenTopics = []; seenDataTypes = []; seenMethods = []

                    for sentence in abstractSentences:
                        for topic in unseenTopics:
                            if topic in sentence:
                                unseenTopics.remove(topic)
                                seenTopics.append(topic)

                        for dataType in unseenDataTypes:
                            if dataType in sentence:
                                unseenDataTypes.remove(dataType)
                                seenDataTypes.append(dataType)

                        for method_ in unseenMethods:
                            if method_ in sentence:
                                unseenMethods.remove(method_)
                                seenMethods.append(method_)

                    topicRelevanceScore, relevantTopics = getRelevance(seenTopics, abstractSentences)
                    paper_info['topics'] = seenTopics
                    paper_info['content_score'] = topicRelevanceScore
                    paper_info['methods'] = seenMethods
                    paper_info['data_type'] = seenDataTypes
                    paper_info['citation_score'] = sigmoid(citations/valid_count, 50.)

                all_paper_info.append(paper_info)
                paper_citation_count.append(citations)
                content_score.append(topicRelevanceScore)

                counter += 1;
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

def post(category,Texts,Web_hook_url,verify):
    for val in Texts:
        post = {'text':val,
                'username':u'Bot',
                'icon_emoji':u':thinking_face:',
                'unfurl_links':True,
                'link_names':1,}
        data = json.dumps(post).encode("utf-8")
        #requests.post(Web_hook_url, data = data,verify=verify)

def sayHello(Web_hook_url, verify):
    post = {'text':'hello slack! \n this is a debug run, ignore!',
            'username':u'arxivx',
            'icon_emoji':u':thinking_face:',
            'unfurl_links':True,
            'link_names':1,}
    data = json.dumps(post).encode("utf-8")
    requests.post(Web_hook_url, data = data, verify = verify)

def getQueryItems(category):
    if category == 'stat.ML':
        topics = ['variational', 'inference', 'dynamic', 'bayesian', 'time series', \
        'state space', 'pose estimation', 'human pose', 'mesh', 'texture', \
        'gaussian', 'forecast', 'visualisation', 'compression', 'streaming', 'monocular', \
        'gradient', 'regularization', 'synthesis', 'point cloud', \
        'nonparametric', 'density estimation', 'latent variable model', 'non parametric', \
        'medical', 'health', 'data augmentation', 'graphic', 'multi view', 'depth estimation',\
        'autoencoder', 'linear regression', 'encoder decoder']
        data = ['image', 'video', 'temporal', 'static']
        method = ['neural network', 'cnn', 'convolution', 'rnn', 'lstm', 'recurrent']

    elif category == 'cs.CV':
        topics = ['texture', 'mesh', \
        '3d object', 'point cloud', 'geometry', 'camera', 'graphic', 'depth estimation', 'monocular', \
        'medical', 'health', 'temporal', 'real time', 'human', 'body', 'pose estimation', \
        'synthesis', 'camera pose', '3d surface', 'reconstruction', 'depth prediction', 'monocular', \
        'camera calibration', 'uncalibrated', 'image compression', 'character', 'multi person',\
        'pose track', '2d pose', '3d pose']
        data = ['image', 'video', 'temporal', 'static']
        method = ['neural network', 'cnn', 'convolution', 'rnn', 'lstm', 'recurrent', 'end to end']

    return (topics, data, method)
