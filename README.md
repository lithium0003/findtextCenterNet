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

EfficientNetV2-XLの出力(入力の1/32サイズ)と、1/4,1/8,1/16サイズとなるのブロックからの途中出力を引き出し、strinde 2のConv2DTransposeで、最終的に
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
      block7 --> P5_in[16x16x640];
      block5 --> P4_in[32x32x256];
      block3 --> P3_in[64x64x96];
      block2 --> P2_in[128x128x64];
      subgraph LeafMap;
        P5_in -- Conv2D k1 --> P5[16x16x80] -- UpSampling2D x16 --> P5_out[256x256x80];
        P4_in -- Conv2D k1 --> P4[32x32x32] -- UpSampling2D x8 --> P4_out[256x256x32];
        P3_in -- Conv2D k1 --> P3[64x64x12] -- UpSampling2D x4 --> P3_out[256x256x12];
        P2_in -- Conv2D k1 --> P2[128x128x8] -- UpSampling2D x2 --> P2_out[256x256x8];
        P5_out & P4_out & P3_out & P2_out -- Conv2D k1 --> top_out[256x256xM] -- Conv2D k1 --> LeafOut[256x256xN];
      end;

```

モデルの出力は、中心位置のヒートマップ(keyheatmap)x1、ボックスサイズ(sizes)x2、オフセット(offsets)x2、
文字の連続ライン(textline)x1、文字ブロックの分離線(sepatator)ｘ1の 256x256x7のマップと、
文字の128次元特徴ベクトル 256x256x128のマップが出力されます。

文字の特徴ベクトルの事前学習として、文字の特徴ベクトルを1文字ずつ文字コードに変換するモデルを後段に付けて学習を行います。

```mermaid
  flowchart TD;
    Input[Feature 128]-- Dense --> Output1091[modulo 1091];
    Input[Feature 128]-- Dense --> Output1093[modulo 1093];
    Input[Feature 128]-- Dense --> Output1097[modulo 1097];
```

文字は、UTF32で1つのコードポイントとして表されるとして、1091,1093,1097での剰余を学習させて、[Chinese remainder theorem](https://ja.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E3%81%AE%E5%89%B0%E4%BD%99%E5%AE%9A%E7%90%86)
により算出した値のうち、0x10FFFFより小さいものが得られた場合に有効としています。

最終的には、この後段は使用せず、文字の特徴ベクトルの連続をTransformerに入力して、文字コードの列を得る予定です。

# Prepare 
Python3でtensorflowを使用します。

```bash
pip3 install tensorflow
pip3 install tensorflow_addons
pip3 install matplotlib
pip3 install scikit-image
```

学習の際に、horovodを使用する場合は、インストールします。

```bash
pip3 install --no-cache-dir horovod
```

学習時に使用するload_fontをコンパイルするのに、libfreetype6-devが必要です

```bash
sudo apt install libfreetype6-dev
```

学習前に、load_fontをコンパイルしておく必要があります。
```bash
cd data/load_font
make
```

学習には、フォントデータが必要です。
resource_list.txtを参照して、適宜フォントデータを配置してください。
著作権法30条の4の規定により、機械学習の学習を目的とする場合はデータをお渡しすることができます。
筆者と同じデータで学習を希望する方は、[メール](<mailto:contact@lithium03.info>)を送ってください。

# Train
## horovodを使用して、A100の8並列で学習する場合

```bash
./run_hvd.sh
```

## 1枚のA100で学習する場合

```bash
./train_batch.py 14
```

## 1枚の1080Tiで学習する場合(うまくいかない可能性があります)

```bash
./train1.py
```

## Windowsで学習させる場合

テキストの読み書きを、utf8に強制してください。
```bash
python.exe -X utf8 train1.py
```

# Test
学習データを、ckpt/　に置いた状態で、
test_image.pyを実行すると推論できます。

```bash
./test_image.py img/test1.png
```

# Reference 
- Objects as Points
https://arxiv.org/abs/1904.07850
- EfficientNetV2
https://arxiv.org/abs/2104.00298
- PyTorchではじめるAI開発　(p.256-)
https://www.amazon.co.jp/dp/B096WWVFJN


