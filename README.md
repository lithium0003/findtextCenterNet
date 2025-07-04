[English](README.en.md)

# findtextCenterNet
機械学習による日本語OCR

CenterNet　https://github.com/xingyizhou/CenterNet
の手法で、
Backbone networkに EfficientNetV2 https://github.com/google/automl/tree/master/efficientnetv2
を使用しています。

後段は、各文字の特徴量ベクトルを文として入力して、Transformerにより文字コードとして文章を出力します。

iOS/macOS アプリとしてこのモデルを使えるようにしたもの
https://apps.apple.com/us/app/bunkoocr/id1611405865

Windowsでこのモデルを使えるようにしたもの
https://lithium03.info/product/bunkoOCR.html

# Example
## 手書き文字
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1.png" width="500">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_result.png" width="500">

## フォント
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2.png" width="1400">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_result.png" width="1400">


# とりあえず実行したいんだけど
事前に学習済みの重みデータ、model.pt, model3.ptをダウンロードして、findtextCenterNet/フォルダ直下に配置します。

```bash
wget https://huggingface.co/lithium0003/findtextCenterNet/resolve/main/model.pt
wget https://huggingface.co/lithium0003/findtextCenterNet/resolve/main/model3.pt
```

検出したBoxを配列するためのモジュール、textline_detectをコンパイルします。
```bash
make -C textline_detect
```

対象ファイルを指定して、run_ocr.pyを呼び出します。
```bash
./run_ocr.py img/test1.png
```

```
吾輩は猫である
名前はまだない

手書き文字認識
```
のように認識文章が出力されます。

# Details 
## detector(step1)

入力画像は 768x768x3

![TextDetector diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/TextDetector.drawio.svg "TextDetector")

EfficientNetV2-XLの出力(入力の1/32サイズ)と、1/4,1/8,1/16サイズとなるのブロックからの途中出力を引き出し、UpSampling2Dで、最終的に
(1/4サイズの)192x192xNの出力を得ます。

モデルの出力は、中心位置のヒートマップ(keyheatmap)x1、ボックスサイズ(sizes)x2、
文字の連続ライン(textline)x1、文字ブロックの分離線(separator)ｘ1、ルビである文字(code1_ruby)x1、
ルビの親文字(code2_rubybase)x1、圏点(code4_emphasis)x1、空白の次文字(code8_space)x1の 192x192x9のマップと、
文字の100次元特徴ベクトル 192x192x100のマップが出力されます。

文字の特徴ベクトルの事前学習として、文字の特徴ベクトルを1文字ずつ文字コードに変換するモデルを後段に付けて学習を行います。

![CodeDecoder diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/CodeDecoder.drawio.svg "CodeDecoder")

文字は、UTF32で1つのコードポイントとして表されるとして、1091,1093,1097での剰余を学習させて、[Chinese remainder theorem](https://ja.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E3%81%AE%E5%89%B0%E4%BD%99%E5%AE%9A%E7%90%86)
により算出した値のうち、0x3FFFFより小さいものが得られた場合に有効としています。

最終的には、この後段は使用せず、文字の特徴ベクトルの連続をTransformerに入力して、文字コード列を得ます。

## result image

例に挙げた画像では、モデルの出力は以下のようになります。

| 項目 | image |
| --- | ------ |
| 元画像 | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1.png" width="400"> |
| 中心位置のヒートマップ(keyheatmap) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_keymap.png" width="400"> |
| 文字の連続ライン(textline) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_textline.png" width="400"> |
| 文字ブロックの分離線(separator) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_separator.png" width="400"> |

| 項目 | image |
| --- | ------ |
| 元画像 | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2.png" width="1400"> |
| 中心位置のヒートマップ(keyheatmap) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_keymap.png" width="1400"> |
| 文字の連続ライン(textline) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_textline.png" width="1400"> |
| ふりがな(code1) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_code1.png" width="1400"> |
| 親文字(code2) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_code2.png" width="1400"> |

中心位置のヒートマップ(keyheatmap)のピーク位置から、文字位置を特定します。文字位置のボックスを、文字の連続ライン(textline)に沿って並べて、文字列の並びを得ます。
このとき、文字ブロックの分離線(separator)で認識したブロック境界を越えないように文章列を分離します。

また、この例ではふりがなと親文字が検出されていますが、圏点文字(code4)と空白の次の文字(code8)についても同様に検出し、マークしておき、後段のTransformerに入れるときに
追加して入れます。

## transformer(step3)

step1により、入力画像は、100次元特徴ベクトルの列に変換されます。
各文字には、縦書きか横書きかのフラグ、空白の次の文字であるかのフラグ、ふりがなであるかどうかのフラグ、ふりがなの親文字であるかのフラグ、圏点が振られているかのフラグ、
改行フラグの6次元を付加します。
こうして得られた106次元のベクトル列を、Transformerにより文字コードに変換します。

Transformerのエンコーダは最大100文字、デコーダーは最大100文字としました。
Encoder、Decoder共に、hidden_dim=512, head_num=16, hopping_num=16とし、PositionalEncodingはsin初期化の学習ありです。
Decoderの出力は、1091,1093,1097での剰余により符号化します。

![Transformer diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/Transformer.drawio.svg "Transformer")


Decoderは、SOT=1で開始し、EOT=2で終了するまでの数値をUnicodeコードポイントとして学習させます。
このとき、高速に並列推論できるように、MSK=3でマスクした場所を、応答させるように学習させます。
空白を埋めるのは、PAD=0です。

マスクがないとき
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Input | SOT | t | e | s | t | EOT | PAD |
| Output | SOT | t | e | s | t | EOT | PAD |

全てマスクがあるとき
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Input | MSK | MSK | MSK | MSK | MSK | MSK | MSK |
| Output | SOT | t | e | s | t | EOT | PAD |

一部マスクがあるとき
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Input | SOT | MSK | e | MSK | t | MSK | MSK |
| Output | SOT | t | e | s | t | EOT | PAD |

# Prepare
Python3でPyTorchを使用します。

```bash
sudo apt install -y python3-pip pkg-config
python3 -m venv venv/torch
. venv/torch/bin/activate
pip3 install torch torchvision torchaudio
pip3 install tensorboard einops webdataset scipy boto3 pyzmq cython pillow-heif
```

学習データを作成するのに使用する、render_fontをコンパイルするのに、libfreetype6-devが必要です

```bash
sudo apt install libfreetype6-dev
```

学習データを作成する前に、render_fontをコンパイルしておく必要があります。
```bash
make -C make_traindata/render_font
```

Transformerの学習データを作成する前に、processer3.pyxをコンパイルしておく必要があります。
```bash
CPLUS_INCLUDE_PATH=$(python3 -c 'import numpy; print(numpy.get_include())') cythonize -i make_traindata/processer3.pyx
```

step1の学習をするのに使用する、downloaderをコンパイルするのに、libcurl4-openssl-devが必要です

```bash
sudo apt install libcurl4-openssl-dev
```

step1の学習をするのに、downloaderをコンパイルしておく必要があります。

```bash
make -C dataset/downloader_src && cp dataset/downloader_src/downloader dataset/
```

step1の学習をするのに、processer.pyxをコンパイルしておく必要があります。

```bash
CPLUS_INCLUDE_PATH=$(python3 -c 'import numpy; print(numpy.get_include())') cythonize -i dataset/processer.pyx
```

後段のTransformerに入れるために、検出した文字ボックスを整列させる必要があります。
このためのプログラムlinedetectはcppで記述しています。
```bash
make -C textline_detect
```

# Training for detector(step1)
## Make train dataset for step1
学習用データセットは、https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/ 以下にあります。
Step1を学習する際には、データセットをWebから直接ロードして学習できるようになっていますが、帯域が必要となるのでダウンロードしてから学習することもできます。
ダウンロードするに次のようにします。
```bash
mkdir train_data1 && cd train_data1
curl -LO "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train[00000000-00001023].tar"
curl -LO "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test[00000000-00000063].tar"
```

自身で学習データを作成するには、フォントデータが必要です。
適宜フォントデータを配置してください。
著作権法30条の4の規定により、機械学習の学習を目的とする場合は筆者が使用したデータをお渡しすることができます。
希望する方は、https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/data.tar.gz よりダウンロードしてください。
ここに含まれるフォントデータは機械学習のためにのみ使用できるので、その他の目的に使用する場合は別途自分でオリジナル配布元から用意してください。

展開してできる、data フォルダを置いた上で、以下のコマンドにより、train_data1 フォルダに学習用データセットを準備します。
```bash
cd make_taindata
./make_traindata1.py 64 1024
mv train_data1 ../
```
この例では、test=64, train=1024ファイルを作成します。

また、xlサイズで学習させる場合は、事前重みをmodels/フォルダに配置する必要があります。
```bash
cd models
wget https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/efficientnetv2-xl-21k.npz
```

## Train for step1
```bash
./train1.py --lr=1e-3 --logstep=10 --output=1000 --gamma=0.95 32
```

step1の学習には、 dataset/downloader_src 以下にソースのある downloader を用いて、 WebDataset としてデータセットをWebから直接ロードする方法と、
ローカルの train_data1/ フォルダに配置した学習データをロードする方法があります。
dataset/data_detector.py の get_dataset関数の local_disk = False フラグで切り替えることができます。

step1の学習時に、高速にaugmentationやデータ変換をする必要があるので、cythonで dataset/processer.pyx をコンパイルしておく必要があります。

train1.pyは、 result1/model.pt に学習済み重みを出力します。

## Test for step1
step1学習済みパラメータを、model.ptに置いた状態で、
test_image1_torch.pyを実行すると推論できます。

```bash
./test_image1_torch.py img/test1.png
```

# Finetune for detector(step2)
step1の文字検出器で、実際のデータを処理したときにi上手くいかない例があれば、特にその例で補充の学習を行うことができます。
## Make train dataset for step2
step1学習済みパラメータを、model.ptに置いた状態で、推論結果をjsonファイルとして出力します。

```bash
fine_image/process_image1_torch.py train_data2/target.png
```

実行すると、(filename).json (filename).lines.png (filename).seps.png の3ファイルが結果として出力されます。

### 文字の修正
認識した文字を確認してjsonを修正するには、 fix_process_image1.py を使用することができます。
```bash
fine_image/fix_process_image1.py train_data2/target.png
```

<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_json1.png" width="400">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_json2.png" width="400">

文字ボックスをダブルクリックすることで、文字や属性を修正できます。

### 文字の連続ラインなどの修正
認識した文字の連続ラインや、文章の分離ラインを修正するには、 fix_line_image1.py を使用することができます。
```bash
fine_image/fix_line_image1.py train_data2/target.png
fine_image/fix_line_image1.py train_data2/target.png seps
```

<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_line1.png" width="400">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_line2.png" width="400">

## Train for step2(finetune detector)
```bash
./train1.py --lr=1e-4 --logstep=10 --output=1000 --weight1=0.5 --weight2=1.0 32
```

step2の学習には、step1で使用したデータに加えて、train_data2/ フォルダ以下に用意したデータを使用して追加学習します。
学習を始める前に、 result2/model.pt として、step1の結果の重みデータを用意しておきます。

## Test for step2
step2学習済みパラメータを、model.ptに置いた状態で、
test_image1_torch.pyを実行すると推論できます。

```bash
./test_image1_torch.py img/test1.png
```

# Training for Transformer(step3) 
step3の学習用データセットは、
* https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data3.tar.gz 
* https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data4.tar.gz 

に用意しています。この中に含まれる著作物は、著作権法30条の4の規定により再配布していますので、法の規定に従って使用してください。

## 自分で用意する場合
step1,(step2)の学習が終わったdetectorの文字検出器を用いて、後段のTransformerの学習データを作成します。

### 文字特徴量のサンプリング
make_traindata3.pyを用いて、文字の画像データから、文字検出器が出力する各文字ごとの特徴量を収集します。

```bash
cd make_traindata
./make_traindata3.py
```
このプロセスは、無限に生成し続けますので適当なところで止めてください。
おおむね必要な文字コードが全部出てきたくらいで止めるといいでしょう。

code_featuresフォルダ以下に、各文字の文字コードごとにファイルができます。
次に、1つのファイルにまとめます。

save_feature.pyを用いて、features.npzを生成します。
```bash
./save_feature.py
mkdir -p ../train_data3
mv features.npz ../train_data3/
```

事前学習済みのfeatures.npzは、train_data3.tar.gzに含まれています。

### step3の学習データtrain_data3の作成
次に、日本語および英語などの文章(wikipediaおよび青空文庫)からTransformer用の学習データをtrain_data3 フォルダに作成します。
```bash
cd train_data3
python3 make_data.py
```

### Finetune用画像からのstep3の学習データtrain_data4の作成
補充の学習データとして、手動で修正したtrain_data2があれば、そのデータもTransformerの学習に使用できます。
train_data2をtrain_data4としてコピーした上で、次にように変換します。
この変換には、textline_detect/linedetect による文の方向検索のモジュールのコンパイルが事前に必要です。
```bash
fine_image/process_image4_torch.py train_data4/target.png
```

## Train for step3
```bash
./train3.py 1024
```

step3の学習では、Transformerを、wikipediaや青空文庫などの自然言語文章により正しい文章となるように学習します。
また、ランダムな文字列により文字コードも特に学習させます。

学習済み重みは、result3/model.pt に保存されます。

## Test for step3
学習済みパラメータを、model.pt, model3.pt に置いた状態で、
test_image3_torch.pyを実行すると推論できます。

```bash
./test_image3_torch.py img/test1.png
```


# Reference 
- Objects as Points
https://arxiv.org/abs/1904.07850
- EfficientNetV2
https://arxiv.org/abs/2104.00298
- PyTorchではじめるAI開発　(p.256-)
https://www.amazon.co.jp/dp/B096WWVFJN
- B2T Connection: Serving Stability and Performance in Deep Transformers
https://arxiv.org/abs/2206.00330
- Schedule-Free Learning
https://github.com/facebookresearch/schedule_free
- Differential Transformer
https://arxiv.org/abs/2410.05258
- Understanding How Positional Encodings Work in Transformer Model
https://aclanthology.org/2024.lrec-main.1478/
- Mask-Predict: Parallel Decoding of Conditional Masked Language Models
https://arxiv.org/abs/1904.09324
- Attending to Future Tokens For Bidirectional Sequence Generation
https://arxiv.org/abs/1908.05915
- Large Language Diffusion Models
https://arxiv.org/abs/2502.09992
