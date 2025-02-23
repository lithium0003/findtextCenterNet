#include <stdio.h>
#include <fstream>
#include <iostream>

#ifdef _WIN64
#include <fcntl.h>
#include <io.h>
#endif

#include <algorithm>
#include <cstdint>

#include "line_detect.h"

double ruby_cutoff = 0.5;
double rubybase_cutoff = 0.5;
double space_cutoff = 0.5;
double emphasis_cutoff = 0.5;
float line_valueth = 0.3;
float sep_valueth = 0.1;
float sep_valueth2 = 0.15;
const float sep_clusterth = 10.0;
const int linearea_th = 20;
double allowwidth_next_block = 1.0;
// double ignore_small_size_block_ratio = 0.7;
// bool ignore_small_size_block = false;
int scale = 4;

int run_mode = 0;
int width = 0;
int height = 0;

int main(int argn, char **argv) 
{
#ifdef _WIN64
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#else
    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);
#endif

    fread(&run_mode, sizeof(uint32_t), 1, stdin);

    fread(&width, sizeof(uint32_t), 1, stdin);
    fread(&height, sizeof(uint32_t), 1, stdin);

    std::vector<float> lineimage(width*height);
    fread(lineimage.data(), sizeof(float), width*height, stdin);
    std::vector<float> sepimage(width*height);
    fread(sepimage.data(), sizeof(float), width*height, stdin);

    int boxcount = 0;
    fread(&boxcount, sizeof(uint32_t), 1, stdin);

    std::cerr << boxcount << std::endl;

    std::vector<charbox> boxes(boxcount);
    for(int i = 0; i < boxcount; i++) {
        boxes[i].id = i;
        boxes[i].block = -1;
        boxes[i].idx = -1;
        boxes[i].subidx = -1;
        boxes[i].subtype = 0;
        boxes[i].direction = 0;
        boxes[i].double_line = 0;
        fread(&boxes[i].cx, sizeof(float), 1, stdin);
        fread(&boxes[i].cy, sizeof(float), 1, stdin);
        fread(&boxes[i].w, sizeof(float), 1, stdin);
        fread(&boxes[i].h, sizeof(float), 1, stdin);
        fread(&boxes[i].code1, sizeof(float), 1, stdin);
        fread(&boxes[i].code2, sizeof(float), 1, stdin);
        fread(&boxes[i].code4, sizeof(float), 1, stdin);
        fread(&boxes[i].code8, sizeof(float), 1, stdin);
        // ルビ親文字
        if(boxes[i].code2 > rubybase_cutoff) {
            boxes[i].subtype |= 2;
        }
        // ルビの文字
        if(boxes[i].code1 > ruby_cutoff) {
            boxes[i].subtype |= 2+4;
        }
        // 空白
        if(boxes[i].code8 > space_cutoff) {
            boxes[i].subtype |= 8;
        }
        // 圏点
        if(boxes[i].code4 > emphasis_cutoff) {
            boxes[i].subtype |= 16;
        }
        // fprintf(stderr, "box %d cx %f cy %f w %f h %f c1 %f c2 %f c4 %f c8 %f t %d\n", 
        //     boxes[i].id, boxes[i].cx, boxes[i].cy, boxes[i].w, boxes[i].h, 
        //     boxes[i].code1, boxes[i].code2, boxes[i].code4, boxes[i].code8,
        //     boxes[i].subtype);
    }

    process(lineimage, sepimage, boxes);

    std::sort(boxes.begin(), boxes.end(), [](auto a, auto b) {
        if(a.block != b.block) return a.block < b.block;
        if(a.idx != b.idx) return a.idx < b.idx;
        if(a.subidx != b.subidx) return a.subidx < b.subidx;
        return a.id < b.id;
    });

    uint32_t count = boxes.size();
    fwrite(&count, sizeof(int32_t), 1, stdout);

    for(int i = 0; i < boxes.size(); i++) {
        // fprintf(stderr, "box %d cx %f cy %f w %f h %f block %d idx %d sidx %d stype %d c1 %f c2 %f c4 %f c8 %f d %d\n", 
        //     boxes[i].id, boxes[i].cx, boxes[i].cy, boxes[i].w, boxes[i].h, 
        //     boxes[i].block, boxes[i].idx, boxes[i].subidx, boxes[i].subtype,
        //     boxes[i].code1, boxes[i].code2, boxes[i].code4, boxes[i].code8,
        //     boxes[i].double_line);
        
        fwrite(&boxes[i].id, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].block, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].idx, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].subidx, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].subtype, sizeof(int32_t), 1, stdout);
    }

    return 0;
}