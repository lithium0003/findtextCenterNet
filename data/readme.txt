フォルダ構成
    enfont/          英語用フォントフォルダ
    jpfont/          日本語用フォントフォルダ
    handwritten/     手書き文字画像フォルダ
    background/      背景画像フォルダ(自然画像等の文字が画像中に含まれていないものを使用する)
    load_font/       fontのビットマップを取得するプログラム
    codepoints.csv   青空文庫の外字をUnicodeに変換するテーブル
    id_map.csv       学習用文字リスト

make_kanji_list.sh
このファイルを実行することで、必要なリストが生成される。

    漢字一覧の1面.txt
    漢字一覧の2面.txt
    非漢字一覧.txt
        Wikipediaより取得した、jisコードとunicodeの対応一覧。

    このファイルから、以下のファイルが生成される。
        1st_kanji.txt
        第1水準漢字

        2nd_kanji.txt
        第2水準漢字

        3rd_kanji.txt
        第3水準漢字

        4th_kanji.txt
        第4水準漢字

        codepoints.csv
        jisコードと、unicodeとの対応

        id_map.csv
        学習用の、文字のクラス分けをした、文字一覧


other_list.txt
記号のリスト

make_glyphid.py
id_map.csvを生成する

load_font/load_font
fontのビットマップを得る
Usage: load_font/load_font font_path size unicode

