from . import get_aozora
from . import get_wikipedia
import os

def aozora():
    urls = get_aozora.get_aozora_urls()
    os.makedirs(os.path.join('train_data3','aozora'), exist_ok=True)

    count = 0
    for url in urls:
        print(count, '/', len(urls), url)
        while True:
            try:
                txt = get_aozora.get_contents(url)
            except:
                continue
            break
        if not txt.strip():
            continue
        filename = os.path.join('train_data3','aozora','aozora_%08d.txt'%count)
        with open(filename, 'w') as wf:
            wf.write(txt)
        count += 1

def wikipedia(lang='ja', rep=40):
    os.makedirs(os.path.join('train_data3','wikipedia_%s'%lang), exist_ok=True)

    pageids = set()
    for i in range(rep):
        print(i, '/', rep)
        pageids |= set(get_wikipedia.get_random_wordid(lang=lang, count=500))
    count = 0
    for pageid in pageids:
        print(count, '/', len(pageids), pageid)
        while True:
            try:
                txt = get_wikipedia.get_word_content(pageid, lang=lang)
            except:
                continue
            break
        if not txt.strip():
            continue
        filename = os.path.join('train_data3','wikipedia_%s'%lang,'wikipedia_%s_%08d.txt'%(lang,count))
        with open(filename, 'w') as wf:
            wf.write(txt)
        count += 1

def process():
    aozora()
    for lang in ['en','ko','fr','de','it']:
        wikipedia(lang)
    for lang in ['ja']:
        wikipedia(lang, rep=40*4)

if __name__=='__main__':
    process()
