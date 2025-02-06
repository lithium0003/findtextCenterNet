import json
import sys
import urllib.parse
import urllib.request
import os
import zipfile
import io
import csv
import re
from html.parser import HTMLParser

code_list = {}
with open('data/codepoints.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        d1,d2,d3 = row[0].split('-')
        d1 = int(d1)
        d2 = int(d2)
        d3 = int(d3)
        c = row[1]
        c = int(c, 16)
        if c > 0x10FFFF:
            txt = chr((c & 0xFFFF0000) >> 16) + chr((c & 0xFFFF))
        else:
            txt = chr(c)
        code_list['%d-%02d-%02d'%(d1,d2,d3)] = txt

def get_aozora_urls():
    aozora_csv_url = 'https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip'

    xhtml_urls = []
    html = urllib.request.urlopen(aozora_csv_url)
    with zipfile.ZipFile(io.BytesIO(html.read())) as myzip:
        with myzip.open('list_person_all_extended_utf8.csv') as myfile:
            reader = csv.reader(io.TextIOWrapper(myfile))
            idx = -1
            for row in reader:
                if idx < 0:
                    idx = [i for i, x in enumerate(row) if 'URL' in x]
                    idx = [i for i in idx if 'HTML' in row[i]]
                    if len(idx) == 0:
                        exit()
                    idx = idx[0]
                    continue
                if row[idx].startswith('https://www.aozora.gr.jp/cards/'):
                    xhtml_urls.append(row[idx])
    return sorted(set(xhtml_urls))

class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = False
        self.count = 0
        self.startpos = (-1,-1)
        self.endpos = (-1,-1)

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            if self.main:
                self.count += 1
            elif ('class', 'main_text') in attrs:
                self.main = True
                self.startpos = self.getpos()
        
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.main:
                if self.count == 0:
                    self.endpos = self.getpos()
                else:
                    self.count -= 1

def get_contents(url):
    html = urllib.request.urlopen(url, timeout=10)
    contents = html.read().decode('cp932')
    parser = MyHTMLParser()
    parser.feed(contents)
    maintext = []
    for lineno, line in enumerate(contents.splitlines()):
        if parser.startpos[0] == lineno + 1:
            maintext.append(line[parser.startpos[1]:])
        elif parser.startpos[0] < lineno + 1 <= parser.endpos[0]:
            if parser.endpos[0] == lineno + 1:
                if parser.endpos[1] == 0:
                    pass
                else:
                    maintext.append(line[:parser.endpos[1]])
            else:
                maintext.append(line)
    maintext = '\n'.join(maintext)
    maintext = re.sub(r'／″＼', '〴〵', maintext)
    maintext = re.sub(r'／＼', '〳〵', maintext)
    maintext = re.sub(r'<ruby><rb>(.*?)</rb>.*?<rt>(.*?)</rt>.*?</ruby>', '\uFFF9\\1\uFFFA\\2\uFFFB', maintext)
    m = True
    while m:
        m = re.search(r'<img .*?/(\d-\d\d-\d\d)\.png.*?>', maintext)
        if m:
            maintext = maintext[:m.start()] + code_list[m.group(1)] + maintext[m.end():]
    maintext = re.sub(r'<span class="notes">.*?</span>', r'', maintext)
    maintext = re.sub(r'<[^>]*?>', r'', maintext)
    return maintext

if __name__ == '__main__':
    from util_funcs import decode_ruby
    
    urls = get_aozora_urls()
    for u in urls:
        print(u)
        print(decode_ruby(get_contents(u)))