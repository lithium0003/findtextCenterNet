import json
import urllib.parse
import urllib.request

# Wikipedia API
WIKI_URL = "https://%s.wikipedia.org/w/api.php?"

# 記事を1件、ランダムに取得するクエリのパラメータを生成する
def set_url_random(count=1):
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random', #ランダムに取得
        'rnnamespace': 0, #標準名前空間を指定する
        'rnlimit': count, #結果数の上限
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
def get_random_wordid(lang='ja', count=1):
    request_url = WIKI_URL%lang
    request_url += urllib.parse.urlencode(set_url_random(count))
    html = urllib.request.urlopen(request_url, timeout=10)
    html_json = json.loads(html.read().decode('utf-8'))
    pageid = [page['id'] for page in html_json['query']['random']]
    return pageid

def get_word_content(pageid, lang='ja'):
    request_url = WIKI_URL%lang
    request_url += urllib.parse.urlencode(set_url_extract(pageid))
    html = urllib.request.urlopen(request_url, timeout=10)
    html_json = json.loads(html.read().decode('utf-8'))
    explaintext = html_json['query']['pages'][str(pageid)]['extract']
    return explaintext

if __name__ == '__main__':
    pageid = get_random_wordid(count=1)
    extract = get_word_content(pageid[0])
    print(extract)

