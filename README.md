# findtextCenterNet
機械学習による日本語OCR

CenterNet　https://github.com/xingyizhou/CenterNet
の手法で、
Backbone networkに EfficientNetV2 https://github.com/google/automl/tree/master/efficientnetv2
を使用しています。

現在、OCRの前段まで完成しております。後段は、各文字の特徴量ベクトルを文として入力して、Transformerにより
文字コードとして文章を出力する予定です。

# Example
## 手書き文字
![手書き文字サンプル1入力](https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1.png "入力1")
![手書き文字サンプル1出力](https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_result.png "出力1")

## フォント
![フォントサンプル2入力](https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2.png "入力2")
![フォントサンプル2出力](https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_result.png "出力2")

# Details 

入力画像は 512x512x3

EfficientNetV2-XLの出力(入力の1/32サイズ)と、1/4,1/8,1/16サイズとなるのブロックからの途中出力を引き出し、UpSampling2Dで、最終的に
256x256xNの出力を得ます。

```mermaid
  flowchart TD;
      Input[Image 512x512x3]-->stem;
      subgraph EfficientNetV2-XL;
        stem-->block1;
        block1-->block2;
        block2-->block3;
        block3-->block4;
        block4-->block5;
        block5-->block6;
        block6-->block7;
        block7-->top;
      end;
      block7 -- BatchNormalization --> P5_in[16x16x640];
      block5 -- BatchNormalization --> P4_in[32x32x256];
      block3 -- BatchNormalization --> P3_in[64x64x96];
      block2 -- BatchNormalization --> P2_in[128x128x64];
      subgraph LeafMap;
        P5_in -- Conv2D k5x5 --> P5[16x16x32] -- UpSampling2D x16 --> P5_out[256x256x32];
        P4_in -- Conv2D k5x5 --> P4[32x32x16] -- UpSampling2D x8 --> P4_out[256x256x15];
        P3_in -- Conv2D k5x5 --> P3[64x64x8] -- UpSampling2D x4 --> P3_out[256x256x8];
        P2_in -- Conv2D k5x5 --> P2[128x128x8] -- UpSampling2D x2 --> P2_out[256x256x8];
        P5_out & P4_out & P3_out & P2_out -- Concat --> top_1[256x256xM1] -- Conv2D k3x3 --> top_2[256x256xM2] -- Conv2D k3x3 --> LeafOut[256x256xN];
      end;

```

モデルの出力は、中心位置のヒートマップ(keyheatmap)x1、ボックスサイズ(sizes)x2、オフセット(offsets)x2、
文字の連続ライン(textline)x1、文字ブロックの分離線(sepatator)ｘ1、ルビである文字(code1_ruby)x1、
ルビの親文字(code2_rubybase)x1、圏点(code4_emphasis)x1、空白の次文字(code8_space)x1の 256x256x11のマップと、
文字の64次元特徴ベクトル 256x256x64のマップが出力されます。

文字の特徴ベクトルの事前学習として、文字の特徴ベクトルを1文字ずつ文字コードに変換するモデルを後段に付けて学習を行います。

```mermaid
  flowchart TD;
    Input[Feature 64]-- Dense --> Output1091[modulo 1091];
    Input[Feature 64]-- Dense --> Output1093[modulo 1093];
    Input[Feature 64]-- Dense --> Output1097[modulo 1097];
```

文字は、UTF32で1つのコードポイントとして表されるとして、1091,1093,1097での剰余を学習させて、[Chinese remainder theorem](https://ja.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E3%81%AE%E5%89%B0%E4%BD%99%E5%AE%9A%E7%90%86)
により算出した値のうち、0x10FFFFより小さいものが得られた場合に有効としています。

最終的には、この後段は使用せず、文字の特徴ベクトルの連続をTransformerに入力して、文字コードの列を得る予定です。

# Prepare 
Python3でtensorflowを使用します。

```bash
pip3 install tensorflow
pip3 install matplotlib
pip3 install scikit-image
```

学習データを作成するのに使用する、render_fontをコンパイルするのに、libfreetype6-devが必要です

```bash
sudo apt install libfreetype6-dev
```

学習データを作成する前に、load_fontをコンパイルしておく必要があります。
```bash
make -C render_font
```

# Make train dataset
学習用データセットは、https://bucket.lithium03.info/dataset20230627/train_data1/ 以下にあります。
ダウンロードするには以下のようにします。
```bash
mkdir train_data1 && cd train_data1
curl -O "https://bucket.lithium03.info/dataset20230627/train_data1/test0000000[0-4].tfrecords"
curl -O "https://bucket.lithium03.info/dataset20230627/train_data1/train00000[000-299].tfrecords"
```

自身で学習データを作成するには、フォントデータが必要です。
resource_list.txtを参照して、適宜フォントデータを配置してください。
著作権法30条の4の規定により、機械学習の学習を目的とする場合はこれらのデータをお渡しすることができます。
希望する方は、[メール](<mailto:contact@lithium03.info>)を送ってください。

以下のコマンドで、train_data1 フォルダに学習用データセットを準備します。
```bash
./make_traindata1.py　5 300
```
この例では、test=5, train=300ファイルを作成します。

# Train
```bash
./train1.py
```

# Test
学習データを、ckpt1/　に置いた状態で、
test_image1.pyを実行すると推論できます。

```bash
./test_image1.py img/test1.png
```

# Reference 
- Objects as Points
https://arxiv.org/abs/1904.07850
- EfficientNetV2
https://arxiv.org/abs/2104.00298
- PyTorchではじめるAI開発　(p.256-)
https://www.amazon.co.jp/dp/B096WWVFJN


