import json
import sys
import urllib.parse
import urllib.request
import os

# Wikipedia API
WIKI_URL_JP = "https://ja.wikipedia.org/w/api.php?"
WIKI_URL_EN = "https://en.wikipedia.org/w/api.php?"
WIKI_URL_KO = "https://ko.wikipedia.org/w/api.php?"

# 記事を1件、ランダムに取得するクエリのパラメータを生成する
def set_url_random():
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random', #ランダムに取得
        'rnnamespace': 0, #標準名前空間を指定する
        'rnlimit': 1 #結果数の上限を1にする(Default: 1)
    }
    return params

# 指定された記事の内容を取得するクエリのパラメータを生成する
def set_url_extract(pageid):
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'extracts',
        'pageids': pageid, #記事のID
        'explaintext': '',
    }
    return params

#ランダムな記事IDを取得
def get_random_wordid(en=False,ko=False):
    if en:
        request_url = WIKI_URL_EN
    elif ko:
        request_url = WIKI_URL_KO
    else:
        request_url = WIKI_URL_JP
    request_url += urllib.parse.urlencode(set_url_random())
    html = urllib.request.urlopen(request_url, timeout=10)
    html_json = json.loads(html.read().decode('utf-8'))
    pageid = (html_json['query']['random'][0])['id']
    return pageid

def get_word_content(pageid, en=False,ko=False):
    if en:
        request_url = WIKI_URL_EN
    elif ko:
        request_url = WIKI_URL_KO
    else:
        request_url = WIKI_URL_JP
    request_url += urllib.parse.urlencode(set_url_extract(pageid))
    html = urllib.request.urlopen(request_url, timeout=10)
    html_json = json.loads(html.read().decode('utf-8'))
    explaintext = html_json['query']['pages'][str(pageid)]['extract']
    return explaintext

if __name__ == '__main__':
    pageid = get_random_wordid()
    extract = get_word_content(pageid)
    print(extract)

