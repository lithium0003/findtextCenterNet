import numpy as np
import glob
import os
import csv
import string
import time

from render_font.get_aozora import get_aozora_urls, get_contents, decode_ruby
from render_font.get_wikipedia import get_random_wordid, get_word_content
from render_font.renderer import Canvas, UNICODE_WHITESPACE_CHARACTERS
from render_font.handwrite import HandwriteCanvas

with open(os.path.join('data','wordlist.txt'),'r') as f:
    wordlist = f.read().splitlines()

with open(os.path.join('data','en_wordlist.txt'),'r') as f:
    en_wordlist = f.read().splitlines()

aozora_urls = get_aozora_urls()
glyphs = {}

jp_char_list = ''
jp_type_list = {}

with open(os.path.join('data','id_map.csv'),'r') as f:
    reader = csv.reader(f)
    for row in reader:
        code = bytes.fromhex(row[2]).decode()
        i = int.from_bytes(code.encode('utf-32le'), 'little')
        glyphs[i] = code
        if 3 <= int(row[3]) <= 5:
            jp_char_list += code
        jp_type_list.setdefault(int(row[3]), [])
        jp_type_list[int(row[3])].append(code)
jp_char_list = list(jp_char_list)

#ラテン1補助
for c in range(0xA1,0x100):
    if c in [0xA8, 0xAA, 0xAD, 0xAF, 0xB2, 0xB3, 0xB4, 0xB8, 0xB9, 0xBA]:
        continue
    glyphs[c] = chr(c)

#ラテン文字拡張
for c in range(0x100, 0x250):
    glyphs[c] = chr(c)

#IPA拡張（国際音声記号
for c in range(0x250, 0x2B0):
    glyphs[c] = chr(c)

#ギリシア文字及びコプト文字
for c in range(0x370, 0x400):
    if c in [0x374, 0x375, 0x378, 0x379, 0x37A]:
        continue
    if c in range(0x380, 0x386):
        continue
    if c in [0x38B, 0x38D, 0x3A2]:
        continue
    glyphs[c] = chr(c)

#キリール文字（キリル文字）
for c in range(0x400, 0x500):
    if c in range(0x483, 0x48A):
        continue
    glyphs[c] = chr(c)

#キリール文字補助
for c in range(0x500, 0x530):
    glyphs[c] = chr(c)

#一般句読点
for c in range(0x2010, 0x2028):
    glyphs[c] = chr(c)
for c in range(0x2030, 0x205F):
    glyphs[c] = chr(c)

#通貨記号
for c in range(0x20A0, 0x20C0):
    glyphs[c] = chr(c)

#文字様記号
for c in range(0x2100, 0x2150):
    glyphs[c] = chr(c)

#数字に準じるもの
for c in range(0x2150, 0x218A):
    glyphs[c] = chr(c)

#矢印
for c in range(0x2190, 0x2200):
    glyphs[c] = chr(c)

#数学記号
for c in range(0x2200, 0x2300):
    glyphs[c] = chr(c)

#その他の技術用記号
for c in range(0x2300, 0x2400):
    glyphs[c] = chr(c)

#囲み英数字
for c in range(0x2460, 0x2500):
    glyphs[c] = chr(c)

#その他の記号
for c in range(0x2600, 0x2700):
    glyphs[c] = chr(c)

#装飾記号
for c in range(0x2700, 0x27C0):
    glyphs[c] = chr(c)

#CJKの記号及び句読点
for c in range(0x3001, 0x303E):
    if c in range(0x302A, 0x3030):
        continue
    glyphs[c] = chr(c)

#平仮名
for c in range(0x309D, 0x30A0):
    glyphs[c] = chr(c)

#片仮名
for c in range(0x30FD, 0x3100):
    glyphs[c] = chr(c)

#囲みCJK文字・月
for c in range(0x3220, 0x3260):
    glyphs[c] = chr(c)
for c in range(0x3280, 0x3300):
    glyphs[c] = chr(c)

#CJK互換用文字
for c in range(0x3300, 0x3400):
    glyphs[c] = chr(c)


glyphs_list = list(glyphs.values())

sim_glyphs_list = '''
高橋髙橋高槗髙槗
高𣘺髙𣘺高𫞎髙𫞎
高崎高﨑高碕高嵜
髙崎髙﨑髙碕髙嵜
山崎山﨑山碕山嵜
斉藤斎藤齊藤齋藤
斉籐斎籐齊籐齋籐
渡辺渡邊渡邉
渡邉󠄂渡邉󠄃渡邉󠄄渡邉󠄅
渡邉󠄆渡邉󠄇渡邉󠄈渡邉󠄉
渡邉󠄊渡邉󠄋渡邉󠄌渡邉󠄍
渡邉󠄎渡邊󠄁渡邊󠄂渡邊󠄃
渡邊󠄄渡邊󠄅渡邊󠄆渡邊󠄇
吉𠮷吉𠮷吉𠮷吉𠮷
吉𠮷吉𠮷吉𠮷吉𠮷
吉𠮷吉𠮷吉𠮷吉𠮷
吉𠮷吉𠮷吉𠮷吉𠮷
浜濱濵
浜崎濱崎濵崎
浜﨑濱﨑濵﨑
土井圡井𡈽井
草彅草薙草凪
太郎太朗太郞
鷗外鴎外
百間百閒
亜亞歩步勲勳
猪豬兎兔栄榮
衛衞鋭銳園薗
奥奧己巳已薫薰
角⻆学學亀龜
仮假尭堯暁曉
国國蔵藏桑桒
経經厳嚴巌巖
児兒紘絋桜櫻
雑雜聡聰沢澤
静靜渋澁寿壽
収收渉涉将將
晋晉真眞慎愼
昴昂瀬瀨関關
摂攝曽曾荘莊
滝瀧塚塚伝傳
戸戶土圡𡈽徳德
富冨尚尙羽羽
瑶瑤遥遙彦彥
桧檜寛寬淵渕
船舩穂穗舖舗
毎每槙槇松柗
来來翠翆恵惠
萌萠柳桺栁薮藪
豊豐遊游弥彌
与與誉譽横橫
吉𠮷頼賴楽樂
凛凜涼凉禄祿
喜㐂巴色邑
戊戉戍戌ゐゑヰヱ
釜釡窯竈橋槗𣘺𫞎
穣穰譲讓松柗枩枀
鉄鉃銕鐵鐡
𠮷野家吉野家
𠮷呑み吉呑み
𠮷兆吉兆
弁護士辯護士
弁理士辨理士
新潟県新泻県
深圳市深セン市
'''
sim_glyphs_list = list(sim_glyphs_list.replace('\n',''))


jpfontlist = glob.glob(os.path.join('data','jpfont','*'))
enfontlist = glob.glob(os.path.join('data','enfont','*'))

ignore_list = [
    'HanaMinA.ttf',
    'Kaisei*.ttf',
    'Murecho*.ttf',
    'PixelMplus*.ttf',
    'sawarabi-gothic-medium.ttf',
    'toroman.ttf',
]
jpvfontlist = set(jpfontlist)
for ign in ignore_list:
    r = set(glob.glob(os.path.join('data','jpfont',ign)))
    jpvfontlist -= r
jpvfontlist = list(jpvfontlist)

jp_furi_fontlist = [
    'BIZUDGothic-Bold.ttf',
    'BIZUDGothic-Regular.ttf',
    'BIZUDMincho-Regular.ttf',
    'BIZUDPGothic-Bold.ttf',
    'BIZUDPGothic-Regular.ttf',
    'BIZUDPMincho-Regular.ttf',
    'GenShinGothic-Monospace-Normal.ttf',
    'GenShinGothic-Normal.ttf',
    'GenShinGothic-P-Normal.ttf',
    'HanaMinA.ttf',
    'KleeOne-Regular.ttf',
    'KleeOne-SemiBold.ttf',
    'MochiyPopOne-Regular.ttf',
    'MochiyPopPOne-Regular.ttf',
    'Murecho-Black.ttf',
    'Murecho-Bold.ttf',
    'Murecho-ExtraBold.ttf',
    'Murecho-ExtraLight.ttf',
    'Murecho-Light.ttf',
    'Murecho-Medium.ttf',
    'Murecho-Regular.ttf',
    'Murecho-SemiBold.ttf',
    'Murecho-Thin.ttf',
    'NotoSansJP-Regular.otf',
    'NotoSerifJP-Regular.otf',
    'OradanoGSRR.ttf',
    'SourceHanSans-Regular.otf',
    'XANO-mincho-U32.ttf',
    'ipaexg.ttf',
    'ipaexm.ttf',
    'mgenplus-2m-regular.ttf',
    'mgenplus-2p-regular.ttf',
    'sawarabi-mincho-medium.ttf',
]
jp_furi_fontlist = [os.path.join('data','jpfont',f) for f in jp_furi_fontlist]
jp_vfuri_fontlist = set(jp_furi_fontlist)
for ign in ignore_list:
    r = set(glob.glob(os.path.join('data','jpfont',ign)))
    jp_vfuri_fontlist -= r
jp_vfuri_fontlist = list(jp_vfuri_fontlist)

jp_yoshi_fontlist = [
    'GenShinGothic-Monospace-Normal.ttf',
    'GenShinGothic-Normal.ttf',
    'GenShinGothic-P-Normal.ttf',
    'mgenplus-2m-regular.ttf',
    'mgenplus-2p-regular.ttf',
]
jp_yoshi_fontlist = [os.path.join('data','jpfont',f) for f in jp_yoshi_fontlist]


def get_random_wari(rng):
    print('get_random_wari')
    txt = ''
    for k in range(50):
        m_l = rng.integers(5, 20)
        kanji = list(rng.choice(jp_type_list[5], 100))
        main = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanji, m_l))
        w_l = rng.integers(10, 30)
        kanji = list(rng.choice(jp_type_list[5], 100))
        wari = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanji, w_l))
        m_l = rng.integers(5, 20)
        kanji = list(rng.choice(jp_type_list[5], 100))
        main2 = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+kanji, m_l))
        txt += main + '（' + wari + '）' + main2

    size = int(np.exp(rng.uniform(np.log(32), np.log(128))))
    direction = 1 if rng.random() < 0.5 else 2
    line_charcount = rng.integers(20, 40)
    sc_w = np.minimum(line_charcount * size, 2000)
    line_count = 25
    sc_h = min(int(2000 / size), line_count)
    font = rng.choice(jpfontlist) if direction == 1 else rng.choice(jpvfontlist)

    header_str = '%d '%(rng.integers(1000))
    m_l = rng.integers(2, 5)
    header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
    for _ in range(5):
        header_str += '　'
        m_l = rng.integers(2, 5)
        header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    footer_str = '%d '%(rng.integers(1000))
    m_l = rng.integers(2, 5)
    footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
    for _ in range(5):
        footer_str += '　'
        m_l = rng.integers(2, 5)
        footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, direction==1, bold=bold, italic=italic) as canvas:
        canvas.set_linewidth(sc_w)
        canvas.set_linemax(sc_h)
        canvas.line_space_ratio = rng.uniform(1.0,2.0)
        canvas.set_header(header_str)
        canvas.set_footer(footer_str)
        d = canvas.draw_wari(txt)
        d['font'] = font
    
    return d

def get_random_furigana(rng):
    # 0: num
    # 1: A
    # 2: a
    # 3: あ
    # 4: ア
    # 5: 亜
    # 8: 弌

    print('get_random_furigana')
    size = int(np.exp(rng.uniform(np.log(20), np.log(150))))
    count = 4000 // size

    txt = '　'
    for _ in range(count):
        p = rng.random()
        if p < 0.7:
            # 漢字にひらがな
            if rng.random() < 0.2:
                before = ''.join(rng.choice(jp_type_list[5], rng.integers(1,5)))
            else:
                before = ''.join(rng.choice(jp_type_list[3], 1))
            m_l = rng.integers(1, 6)
            main = ''.join(rng.choice(jp_type_list[5]+jp_type_list[8], m_l))
            ruby = ''.join(rng.choice(jp_type_list[3], rng.integers(m_l, m_l * 2 + 4)))
            if rng.random() < 0.2:
                after = ''.join(rng.choice(jp_type_list[5], rng.integers(1,5)))
            else:
                after = ''.join(rng.choice(jp_type_list[3], 1))
            txt += before+'\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'+after
        elif p < 0.9:
            # 日本語に傍点
            m_l = rng.integers(1, 15)
            main1 = ''.join(rng.choice(jp_type_list[3], 20))
            main2 = ''.join(rng.choice(jp_type_list[4], 10))
            main3 = ''.join(rng.choice(jp_type_list[5], 10))
            main = ''.join(rng.choice(list(main1+main2+main3), m_l))
            if rng.random() < 0.95:
                ruby = '﹅'
            else:
                ruby = ''.join(rng.choice(['•','◦','●','○','◎','◉','▲','△','﹅','﹆'],1))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        elif p < 0.95:
            # 漢字にカタカナ
            m_l = rng.integers(1, 6)
            main = ''.join(rng.choice(jp_type_list[5]+jp_type_list[8], m_l))
            ruby = ''.join(rng.choice(jp_type_list[4], rng.integers(m_l, m_l * 2 + 4)))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        elif p < 0.975:
            # alphabetに日本語
            m_l = rng.integers(5, 20)
            main = ''.join(rng.choice(jp_type_list[2], m_l))
            ruby = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5]+jp_type_list[8], rng.integers(3, m_l)))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'
        else:
            # 日本語にalphabet
            m_l = rng.integers(3, 5)
            main = ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5]+jp_type_list[8], m_l))
            ruby = ''.join(rng.choice(jp_type_list[2], rng.integers(10, 20)))
            txt += '\uFFF9'+main+'\uFFFA'+ruby+'\uFFFB'

        m_l = rng.integers(1, 10)
        main = ''.join(rng.choice(jp_type_list[3]+list(rng.choice(jp_type_list[5], 100)), m_l))
        txt += main

        if rng.random() < 0.1:
            txt += '\n　'
        elif rng.random() < 0.1:
            txt += '　'
        elif rng.random() < 0.4:
            txt += '、'
        elif rng.random() < 0.4:
            txt += '。'

    header_str = '%d '%(rng.integers(1000))
    m_l = rng.integers(2, 5)
    header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
    for _ in range(5):
        header_str += '　'
        m_l = rng.integers(2, 5)
        header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    footer_str = '%d '%(rng.integers(1000))
    m_l = rng.integers(2, 5)
    footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
    for _ in range(5):
        footer_str += '　'
        m_l = rng.integers(2, 5)
        footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    direction = 1 if rng.random() < 0.5 else 2
    font = rng.choice(jp_furi_fontlist) if direction == 1 else rng.choice(jp_vfuri_fontlist)

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, direction==1, bold=bold, italic=italic) as canvas:
        canvas.set_linewidth(min(80 * size, 2000))
        canvas.line_space_ratio = rng.uniform(1.5,2.5)
        canvas.set_header(header_str)
        canvas.set_footer(footer_str)
        d = canvas.draw(txt)
        d['font'] = font

    return d        

def get_random_textline(rng, single=False):
    max_text = 32*1024
    while True:
        print('get_random_textline')
        p = rng.random()
        try:
            yoshi = False
            if p < 0.005:
                en = False
                yoshi = True

                content = ''.join(rng.choice(sim_glyphs_list, size=max_text))
                content = ''.join([c if rng.random() > 0.01 else (' ' if rng.random() < 0.5 else '　') for c in content])

                for _ in range(100):
                    nami = ''.join(['〰'] * rng.integers(2,20))
                    idx = rng.integers(2,len(content))
                    content = content[:idx] + nami + content[idx:]

                ln_count = len(content) // 100
                for _ in range(ln_count):
                    idx = rng.integers(2,len(content))
                    content = content[:idx] + '\n　' + content[idx:]

            elif p < 0.2: 
                en = False

                content = ''.join(rng.choice(glyphs_list, size=max_text))
                content = ''.join([c if rng.random() > 0.01 else (' ' if rng.random() < 0.5 else '　') for c in content])
                
                ln_count = len(content) // 100
                for _ in range(ln_count):
                    idx = rng.integers(2,len(content))
                    content = content[:idx] + '\n　' + content[idx:]

            elif p < 0.6:
                # en
                en = True

                pageid = get_random_wordid(en=True)
                content = get_word_content(pageid, en=True)
            else:
                en = False

                if rng.random() < 0.5:
                    # aozora
                    url = rng.choice(aozora_urls)
                    content = get_contents(url)
                else:
                    pageid = get_random_wordid()
                    content = get_word_content(pageid)
        except OSError:
            time.sleep(1)
            continue
        if len([c for c in content if c not in UNICODE_WHITESPACE_CHARACTERS]) < 2049:
            time.sleep(1)
            continue
        else:
            break
    
    direction = 1 if en or rng.random() < 0.5 else 2
    current_fontlist = enfontlist if en else (jpfontlist if direction == 1 else jpvfontlist)
    if yoshi:
        current_fontlist = jp_yoshi_fontlist
    
    start = rng.integers(0, max(1, len(content)-4*1024))
    if len(content) - start > 4*1024:
        end = rng.integers(start + 4*1024, min(start + max_text, len(content)))
    else:
        end = len(content)

    txt = content[start:end]

    font = rng.choice(current_fontlist)

    if en:
        size = int(np.exp(rng.uniform(np.log(30), np.log(128))))
    else:
        size = int(np.exp(rng.uniform(np.log(18), np.log(128))))
    line_charcount = rng.integers(20, 40)
    sec = 1 if single else 2
    sc_w = np.minimum(line_charcount * size, 2000)
    line_count = len(txt) // (line_charcount * sec)
    sc_h = min(int(2000 / size), line_count)

    header_str = '%d '%(rng.integers(1000))
    if en:
        m_l = rng.integers(2, 5)
        header_str += ''.join(rng.choice(jp_type_list[0]+jp_type_list[1]+jp_type_list[2], m_l))
        for _ in range(5):
            header_str += ' '
            m_l = rng.integers(2, 5)
            header_str += ''.join(rng.choice(jp_type_list[0]+jp_type_list[1]+jp_type_list[2], m_l))
    else:
        m_l = rng.integers(2, 5)
        header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
        for _ in range(5):
            header_str += '　'
            m_l = rng.integers(2, 5)
            header_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    footer_str = '%d '%(rng.integers(1000))
    if en:
        m_l = rng.integers(2, 5)
        footer_str += ''.join(rng.choice(jp_type_list[0]+jp_type_list[1]+jp_type_list[2], m_l))
        for _ in range(5):
            footer_str += ' '
            m_l = rng.integers(2, 5)
            footer_str += ''.join(rng.choice(jp_type_list[0]+jp_type_list[1]+jp_type_list[2], m_l))
    else:
        m_l = rng.integers(2, 5)
        footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))
        for _ in range(5):
            footer_str += '　'
            m_l = rng.integers(2, 5)
            footer_str += ''.join(rng.choice(jp_type_list[3]+jp_type_list[4]+jp_type_list[5], m_l))

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, direction==1, bold=bold, italic=italic) as canvas:
        canvas.set_linewidth(sc_w)
        canvas.set_linemax(sc_h)
        if single:
            canvas.line_space_ratio = rng.uniform(1.5,2.0)
        else:
            canvas.line_space_ratio = rng.uniform(1.0,2.0)
        if sec > 1:
            canvas.set_section(sec, rng.uniform(0.1, 3.0))
        canvas.set_header(header_str)
        canvas.set_footer(footer_str)
        d = canvas.draw(txt)
        d['font'] = font
    
    return d

def get_random_il(rng):
    print('get_random_il')
    font = rng.choice(enfontlist)
    content = []
    ligatures = ['fi','ffi','fl','ffl','fj','ffj','tt','ti','tti','tj','ttj','il','ll','I',"I'"]
    for _ in range(rng.integers(64,256)):
        word = ''
        for _ in range(4):
            if rng.random() < 0.5:
                word += rng.choice(list('abcdefghijklmnopqrstuvwxyz'))[0]
            word += ''.join(rng.choice(ligatures))
        if rng.random() < 0.5:
            word += '!'
        word = "“%s”"%word
        content.append(word)

    size = int(np.exp(rng.uniform(np.log(48), np.log(128))))

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, bold=bold, italic=italic) as canvas:
        d = canvas.random_draw(content, 2048, 2048, rng)
        d['font'] = font

    return d

def get_random_word(rng):
    print('get_random_word')
    p = rng.random()
    if p < 0.5:
        content = rng.choice(en_wordlist, size=rng.integers(64, 256))
        font = rng.choice(enfontlist)
    else:
        content = rng.choice(wordlist, size=rng.integers(32, 64))
        font = rng.choice(jpvfontlist)
    content = list(content)
    for _ in range(rng.integers(32)):
        k = int(np.power(10.,rng.uniform(1.,6.)))
        content.append("%d"%k)

    size = int(np.exp(rng.uniform(np.log(48), np.log(128))))

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, bold=bold, italic=italic) as canvas:
        d = canvas.random_draw(content, 2048, 2048, rng)
        d['font'] = font

    return d

def get_random_grid(rng):
    print('get_random_grid')
    col = rng.integers(3, 5)
    row = rng.integers(3, 5)
    value = []
    for i in range(row):
        line_value = []
        for j in range(col):
            p = rng.random()
            if p < 0.5:
                v = '%f'%rng.uniform(-1000,1000)
            elif p < 0.75:
                v = ''.join(rng.choice(list(string.ascii_letters), size=rng.integers(4, 16)))
                if len(v) > 8:
                    v = v[:8] + '\n' + v[8:]
            else:
                v = ''.join(rng.choice(jp_char_list, size=rng.integers(4, 16)))
                if len(v) > 8:
                    v = v[:8] + '\n' + v[8:]

            line_value.append(v)
        value.append(line_value)

    font = rng.choice(jpfontlist)
    size = int(np.exp(rng.uniform(np.log(20), np.log(128))))

    italic = rng.random() < 0.1
    bold = rng.random() < 0.2
    with Canvas(font, size, bold=bold, italic=italic) as canvas:
        d = canvas.random_drawgrid(value, rng)
        d['font'] = font

    return d

def get_random_hendwrite(rng):
    max_text = 32*1024
    while True:
        print('get_random_hendwrite')
        p = rng.random()
        if p < 0.3: 
            content = ''.join(rng.choice(jp_char_list, size=rng.integers(1024, 4096)))
        else:
            try:
                if rng.random() < 0.5:
                    # aozora
                    url = rng.choice(aozora_urls)
                    content = get_contents(url)
                else:
                    pageid = get_random_wordid()
                    content = get_word_content(pageid)
            except OSError:
                time.sleep(1)
                continue
        if len([c for c in content if c not in UNICODE_WHITESPACE_CHARACTERS]) < 2049:
            time.sleep(1)
            continue
        else:
            break
    
    start = rng.integers(0, max(1, len(content)-8*1024))
    if len(content) - start > 8*1024:
        end = rng.integers(start + 8*1024, min(start + max_text, len(content)))
    else:
        end = len(content)

    txt = content[start:end]

    ratio = np.exp(rng.uniform(np.log(24), np.log(128))) / 128
    direction = 1 if rng.random() < 0.5 else 2
    line_charcount = rng.integers(20, 40)
    sc_w = np.minimum(int(line_charcount * ratio * 128), 2000)
    line_count = len(txt) // line_charcount
    min_line = int(600 / (128 * ratio))
    sc_h = rng.integers(min_line, max(min_line+1, min(int(2000 / (128 * ratio)), line_count)))

    canvas = HandwriteCanvas(ratio, horizontal=direction==1)
    canvas.set_linewidth(sc_w)
    canvas.set_linemax(sc_h)
    canvas.line_space_ratio = rng.uniform(1.0,2.0)
    d = canvas.draw(txt)
    
    return d

def get_random_text(rng):
    p = rng.random()

    if p < 0.4:
        return get_random_textline(rng)
    elif p < 0.55:
        return get_random_wari(rng)
    elif p < 0.7:
        return get_random_word(rng)
    elif p < 0.85:
        return get_random_furigana(rng)
    elif p < 0.9:
        return get_random_grid(rng)
    elif p < 0.95:
        return get_random_il(rng)
    else:
        return get_random_hendwrite(rng)

def get_random_text2(rng):
    p = rng.random()
    if p < 0.2:
        return get_random_wari(rng)
    return get_random_furigana(rng)

if __name__ == '__main__':
    from matplotlib import rcParams
    rcParams['font.serif'] = ['IPAexMincho', 'IPAPMincho', 'Hiragino Mincho ProN']

    import matplotlib.pyplot as plt

    rng = np.random.default_rng()

    while True:
        d = get_random_text(rng)
        #d = get_random_textline(rng)
        #d = get_random_il(rng)
        #d = get_random_furigana(rng)
        #d = get_random_il(rng)
        print()
        print(d.get('font'))
        print(decode_ruby(d['str']))

        plt.figure(tight_layout=True)
        plt.imshow(d['image'])
        plt.show()
        continue

        plt.figure(tight_layout=True)
        plt.imshow(d['image'])

        for (c, t), (cx, cy, w, h) in zip(d['code_list'], d['position']):
            points = [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy + h / 2],
                [cx - w / 2, cy + h / 2],
                [cx - w / 2, cy - h / 2],
            ]
            points = np.array(points)
            if t & 2 > 0:
                plt.plot(points[:,0], points[:,1], 'g', linewidth = 1.0)
            elif t & 4 > 0:
                plt.plot(points[:,0], points[:,1], 'y', linewidth = 1.0)
            elif t & 1 > 0:
                plt.plot(points[:,0], points[:,1], 'c', linewidth = 1.0)
            elif t & 8 > 0:
                plt.plot(points[:,0], points[:,1], 'm', linewidth = 1.5)
            else:
                plt.plot(points[:,0], points[:,1], 'w', linewidth = 0.5)
        #     #plt.text(cx, cy, chr(c), fontsize=28, color='blue', family='serif')
        plt.gca().axis('off')

        plt.show()
