#pragma once

struct charbox {
    int id;
    int block;
    int idx;
    int subtype; // 1: vert, 2,4: (10, rubybase, 11, ruby), 8: sp, 16: emphasis / 32: alone ruby
    int subidx;
    int double_line;
    int page;
    int section;
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
extern double emphasis_cutoff;
extern double space_cutoff;
extern float line_valueth;
extern float sep_valueth;
extern float sep_valueth2;
extern const float sep_clusterth;
extern const int linearea_th;
extern double allowwidth_next_block;
extern double allow_sizediff;
extern int page_divide;
extern int scale;

extern int run_mode;
extern int width;
extern int height;

void call(int imwidth,
          int imheight,
          const float *linedata,
          const float *sepdata,
          int count,
          const float *boxdata,
          int *outinfo);
