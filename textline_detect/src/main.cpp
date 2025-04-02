#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef _WIN64
#include <fcntl.h>
#include <io.h>
#endif

#include <algorithm>
#include <cstdint>

#include "line_detect.h"
#include "process.h"

double ruby_cutoff = 0.5;
double rubybase_cutoff = 0.5;
double space_cutoff = 0.5;
double emphasis_cutoff = 0.5;
float line_valueth = 0.4;
float sep_valueth = 0.1;
float sep_valueth2 = 0.15;
const float sep_clusterth = 10.0;
const int linearea_th = 20;
double allowwidth_next_block = 1.5;
double allow_sizediff = 0.5;
int page_divide = 0;
int scale = 4;

int run_mode = 0;
int width = 0;
int height = 0;

int main(int argc, char **argv) 
{
    for(int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if(arg.find("--ruby_cutoff=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> ruby_cutoff;
            std::cerr << "ruby_cutoff=" << ruby_cutoff << std::endl;
        }
        if(arg.find("--rubybase_cutoff=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> rubybase_cutoff;
            std::cerr << "rubybase_cutoff=" << rubybase_cutoff << std::endl;
        }
        if(arg.find("--space_cutoff=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> space_cutoff;
            std::cerr << "space_cutoff=" << space_cutoff << std::endl;
        }
        if(arg.find("--emphasis_cutoff=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> emphasis_cutoff;
            std::cerr << "emphasis_cutoff=" << emphasis_cutoff << std::endl;
        }
        if(arg.find("--line_valueth=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> line_valueth;
            std::cerr << "line_valueth=" << line_valueth << std::endl;
        }
        if(arg.find("--sep_valueth=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> sep_valueth;
            std::cerr << "sep_valueth=" << sep_valueth << std::endl;
        }
        if(arg.find("--sep_valueth2=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> sep_valueth2;
            std::cerr << "sep_valueth2=" << sep_valueth2 << std::endl;
        }
        if(arg.find("--allowwidth_next_block=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> allowwidth_next_block;
            std::cerr << "allowwidth_next_block=" << allowwidth_next_block << std::endl;
        }
        if(arg.find("--allow_sizediff=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> allow_sizediff;
            std::cerr << "allow_sizediff=" << allow_sizediff << std::endl;
        }
        if(arg.find("--page_divide=") != std::string::npos) {
            std::string vstr = arg.substr(arg.find('=')+1);
            std::stringstream(vstr) >> page_divide;
            std::cerr << "page_divide=" << page_divide << std::endl;
        }
    }

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
        fwrite(&boxes[i].page, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].section, sizeof(int32_t), 1, stdout);
    }

    return 0;
}