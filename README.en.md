[Japanese](README.md)

# findtextCenterNet
Japanese OCR with deeplearning

This model uses the method of CenterNet https://github.com/xingyizhou/CenterNet and 
uses EfficientNetV2 https://github.com/google/automl/tree/master/efficientnetv2 for backbone model.

After detect a latent features of character, this model generate UTF-32 codes using Encoder-Decoder type Transformer. 

iOS/macOS app
https://apps.apple.com/us/app/bunkoocr/id1611405865

Windows version
https://lithium03.info/product/bunkoOCR.html

# Example
## handwritten text
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1.png" width="500">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_result.png" width="500">

## font text
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2.png" width="1400">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_result.png" width="1400">


# Anyway, I want to run the model.
Please download the pre-trained weights, `model.pt`, `model3.pt`, and place just below on the folder findtextCenterNet/

```bash
wget https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/model.pt
wget https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/model3.pt
```

Call `run_ocr.py` with specify the target image.
```bash
./run_ocr.py img/test1.png
```

Detected text shows like this:
```
吾輩は猫である
名前はまだない

手書き文字認識
```

# Details 
## detector(step1)

Input image size is 768x768x3

![TextDetector diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/TextDetector.drawio.svg "TextDetector")

Final output is 192x192xN (1/4 of input size), processed 4 outputs with UpSampling2D; output of EfficientNetV2-XL(1/32 of input image size) and outputs from the middle block of 1/4,1/8,1/16 sizes.
Model output are map(192x192x9) and latent features(192x192x100);
the map are "keyhearmap"(x1) center position of bounding box, "sizes"(x2) box size of width and height,
"textline"(x1) character chain line of text, "separator"(x1) separator line of text block,
"code1_ruby"(x1) ruby(furigana) or not, "code2_rubybase"(x1) ruby base text(parent of furigana) or not,
"code4_emphasis"(x1) emphasis or not, "code8_space"(x1) next of space.
The latent features are map of latent feature vector of a text charactor.

For pre-training of latent feature vector, add CodeDecoder model that converts each latent feature vector to charactor UTF-32 code.

![CodeDecoder diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/CodeDecoder.drawio.svg "CodeDecoder")

The text charactor expressed by UTF-32 codepoint, the value lower than 0x3FFFF calculated with [Chinese remainder theorem](https://ja.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E3%81%AE%E5%89%B0%E4%BD%99%E5%AE%9A%E7%90%86), the model trained modulo of 1091,1093,1097.

In the end, this CodeDecoder is not used, the input stream of latent features are converted with Transformer to Unicode(UTF-32) text.

## result image

The model outpus of example images are as follows. 

| item | image |
| --- | ------ |
| orignal image | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1.png" width="400"> |
| center position of bounding box (keyheatmap) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_keymap.png" width="400"> |
| character chain line of text (textline) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_textline.png" width="400"> |
| separator line of text block (separator) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test1_separator.png" width="400"> |

| item | image |
| --- | ------ |
| orignal image | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2.png" width="1400"> |
| center position of bounding box (keyheatmap) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_keymap.png" width="1400"> |
| character chain line of text (textline) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_textline.png" width="1400"> |
| ruby (code1) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_code1.png" width="1400"> |
| ruby base text (code2) | <img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/test2_code2.png" width="1400"> |

The peak point of "keyheatmap" shows the text charactor bounding box center postion.
These boxes form into a line by "textline" that not cross over the "separator"

This example shows furigana(ruby), emphasis(code4) and next of space(code8) are also like furigana(ruby), marked to the latent features vector, and input to the Transformer.

## transformer(step3)

In the step1, a imput image convert to a stream of 100-dimension latent vector.
Each text charactor appended 6 flags of horizontal/vertical text, next of space, ruby, ruby base, emphasis, and new line.
This 106-dimension vector stream are converted with Transformer to Unicode text.

Transformer consists of Encoder (max 100) and Decoder (max 100).
Both encoder and decoder have parameters, hidden_dim=512, head_num=16, hopping_num=16, PositionalEncoding is trainable and initialized sinusoidal encoding.
Decoder output is Unicode text which coded UTF-32 and modulo of 1091,1093,1097.

![Transformer diagram](https://github.com/lithium0003/findtextCenterNet/blob/main/img/Transformer.drawio.svg "Transformer")

Train the decoder to output the value of Unicode points, start of SOT=1, end of EOT=2.
Training uses masked prediction for inference spead, train to output the correct code on MSK=3 position.
Padding is PAD=0.

No mask
| Index | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Input | SOT | t | e | s | t | PAD |
| Output | t | e | s | t | EOT | PAD |

All mask (for first inference loop)
| Index | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Input | SOT | MSK | MSK | MSK | MSK | MSK |
| Output | t | e | s | t | EOT | PAD |

Partial mask
| Index | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Input | SOT | MSK | e | MSK | t | PAD |
| Output | t | e | s | t | EOT | PAD |

# Prepare
Use PyTorch and Python3

```bash
sudo apt install -y python3-pip pkg-config
python3 -m venv venv/torch
. venv/torch/bin/activate
pip3 install torch torchvision torchaudio
pip3 install tensorboard einops webdataset scipy boto3 pyzmq cython pillow-heif
```

For compile render_font which used to create train data, need `libfreetype6-dev`
```bash
sudo apt install libfreetype6-dev
```

Before create train data, need to compile `render_font`
```bash
make -C make_traindata/render_font
```

Before create train data for Transformer, need to compile `processer3.pyx`
```bash
CPLUS_INCLUDE_PATH=$(python3 -c 'import numpy; print(numpy.get_include())') cythonize -i make_traindata/processer3.pyx
```

For compile downloader which used to train step1, need `libcurl4-openssl-dev`
```bash
sudo apt install libcurl4-openssl-dev
```

Before train step1, need to compile `downloader`
```bash
make -C dataset/downloader_src && cp dataset/downloader_src/downloader dataset/
```

Before train step1, need to compile `processer.pyx`
```bash
CPLUS_INCLUDE_PATH=$(python3 -c 'import numpy; print(numpy.get_include())') cythonize -i dataset/processer.pyx
```

For aline the detected boxes before input to Transformer, using the cpp module `linedetect`
```bash
make -C textline_detect
```

# Training for detector(step1)
## Make train dataset for step1
Training dataset for step1 in
https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/ 

In step1, you can directory load the dataset from web, but you want to reduce the bandwidth, you can download before trainging.
```bash
mkdir train_data1 && cd train_data1
curl -LO "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/train{00000000..00001023}.tar"
curl -LO "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/train_data1/test{00000000..00000063}.tar"
```

If you want to make train data your own, need the font files.
In japanese law, you can use the authors training dataset. Please contact to contact@lithium03.info .

After extract data.tar.gz and place `data` folder, prepare the step1 train dataset in `train_data1` as follows,
```bash
cd make_taindata
./make_traindata1.py 64 1024
mv train_data1 ../
```
This example makes test=64, train=1024 files.

## Train for step1
```bash
./train1.py --lr=1e-3 --logstep=10 --output=1000 --gamma=0.95 32
```

In step1 training, two way of data loding, direct web loading or local loading.
The direct web loading method uses `downloader` and `WebDataset`.
The local loading method load from the folder `train_data1/`.
You can switch the method with the flag `local_disk = False` of function `get_dataset` in `dataset/data_detector.py`.

Before step1 training, compile `dataset/processer.pyx` by cython for faster augmentation or data conversion.

After step1, output `result1/model.pt`

## Test for step1
You can inference the step1 result as follows.
Place `model.pt` on top folder.
```bash
./test_image1_torch.py img/test1.png
```

# Finetune for detector(step2)
If your own text are not recognized well, you can additional training those samples.
## Make train dataset for step2
Place the step1 output parameter to `model.pt` on top folder.
As follows, output step1 result as json file. 
```bash
fine_image/process_image1_torch.py train_data2/target.png
```

Get 3 files as result, (filename).json, (filename).lines.png and (filename).seps.png

### Fix text
You can use `fix_process_image1.py` for fix text on json.
```bash
fine_image/fix_process_image1.py train_data2/target.png
```

<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_json1.png" width="400">
<img src="https://github.com/lithium0003/findtextCenterNet/blob/main/img/fix_image_json2.png" width="400">

To fix the text or attribute, double-click on the box.

### Fix text line or separate line
You can use `fix_line_image1.py` for fix text line or separate line.
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

In step2, the trainging uses additionaly `train_data2/` folder data and also step1 data.
Before step2, copy the weight of step1 as `result2/model.pt`.

## Test for step2
You can inference the step2 result as follows.
Place `model.pt` on top folder.
```bash
./test_image1_torch.py img/test1.png
```

# Training for Transformer(step3)
In japanese law, you can use the authors training dataset. Please contact to contact@lithium03.info .

## Case for your own training data
Using the text detector trained after step1 (and step2), make train data for Transformer.

### Sampling latent features
Using `make_traindata3.py`, sample the latent features vector from the image of text.

```bash
cd make_traindata
./make_traindata3.py
```
This process is infinite loop, breke some point after all charactor code that you need are appeared.

Output files each charactor code in `code_features/`.
Next, combine that as follows;

Make `features.npz` with `save_feature.py` form code_features.
```bash
./save_feature.py
mkdir -p ../train_data3
mv features.npz ../train_data3/
```

### make train_data3 for step3 training dataset
Make training dataset for Transformer as follows.
```bash
cd train_data3
python3 make_data.py
```

### make train_data4 for step3 training dataset, from fine-tuning images
If you already prepare the train_data2 for fine-tuning, you can also using it.
Copy `train_data2` to `train_data4` and convert as follows,
```bash
fine_image/process_image4_torch.py train_data4/target.png
```
Before this conversion, need to compile `textline_detect/linedetect`

## Train for step3
```bash
./train3.py 1024
```

In step3, train the Transformer to output the natural text.

After step3, output `result3/model.pt`

## Test for step3
You can inference the step3 result as follows.
Place `model.pt` and `model3.pt` on top folder.
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
