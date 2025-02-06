#pragma once

#include <vector>

struct charbox {
    int id;
    int block;
    int idx;
    int subtype; // 1: vert, 2,4: (10, rubybase, 11, ruby), 8: sp
    int subidx;
    int double_line;
    double direction;
    float cx;
    float cy;
    float w;
    float h;
    float code1;
    float code2;
    float code4;
    float code8;
};

extern double ruby_cutoff;
extern double rubybase_cutoff;
extern double space_cutoff;
extern float line_valueth;
extern float sep_valueth;
extern float sep_valueth2;
extern const float sep_clusterth;
extern const int linearea_th;
extern double allowwidth_next_block;
// extern double ignore_small_size_block_ratio;
// extern bool ignore_small_size_block;
extern int scale;

extern int run_mode;
extern int width;
extern int height;

void process(
    const std::vector<float> &lineimage, 
    const std::vector<float> &sepimage,
    std::vector<charbox> &boxes);

void print_chaininfo(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain);
