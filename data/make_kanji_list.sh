#!/bin/bash
grep '第1水準' 漢字一覧の1面.txt | sed -n 's/.*U+\([0-9a-fA-F]*\).*/\1/p' > 1st_kanji.txt
grep '第2水準' 漢字一覧の1面.txt | sed -n 's/.*U+\([0-9a-fA-F]*\).*/\1/p' > 2nd_kanji.txt
grep '第3水準' 漢字一覧の1面.txt | sed -n 's/.*U+\([0-9a-fA-F]*\).*/\1/p' > 3rd_kanji.txt
grep '第4水準' 漢字一覧の2面.txt | sed -n 's/.*U+\([0-9a-fA-F]*\).*/\1/p' > 4th_kanji.txt
grep -o '[0-9]\{1,\}-[0-9]\{1,\}-[0-9]\{1,\}.*\(U+[0-9a-fA-F]* *\)\+' 非漢字一覧.txt | sed -e 's/0x[0-9a-fA-F]*//' -e 's/\t/,/' -e 's/U+//g' | sed 's/[^-,0-9A-Fa-f]//g' >codepoints.csv
grep -o '[0-9]\{1,\}-[0-9]\{1,\}-[0-9]\{1,\}.*\(U+[0-9a-fA-F]* *\)\+' 漢字一覧の1面.txt | sed -e 's/0x[0-9a-fA-F]*//' -e 's/\t/,/' -e 's/U+//g' | sed 's/[^-,0-9A-Fa-f]//g' >>codepoints.csv
grep -o '[0-9]\{1,\}-[0-9]\{1,\}-[0-9]\{1,\}.*\(U+[0-9a-fA-F]* *\)\+' 漢字一覧の2面.txt | sed -e 's/0x[0-9a-fA-F]*//' -e 's/\t/,/' -e 's/U+//g' | sed 's/[^-,0-9A-Fa-f]//g' >>codepoints.csv
python3 make_glyphid.py
