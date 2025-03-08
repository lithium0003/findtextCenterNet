#include "hough_linefind.h"
#include "search_loop.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <iostream>

// 文字ボックスと線分の接触を検出して、ライン上にboxを並べる
std::vector<std::vector<int>> chain_boxes(
    int lineid_count,
    std::vector<charbox> &boxes,
    const std::vector<double> &angle_map,
    const std::vector<int> &lineid_map)
{
    fprintf(stderr, "chain boxes1\n");
    std::vector<std::vector<int>> line_box_chain(lineid_count);
    for(int boxid = 0; boxid < boxes.size(); boxid++) {
        // ふりがなは後で
        if((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
        
        for(int di = 0; di < std::max(boxes[boxid].w, boxes[boxid].h) / 2; di+=scale) {
            {
                int yi = boxes[boxid].cy;
                int y = yi / scale;
                int xi = boxes[boxid].cx - di;
                int x = xi / scale;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        if(fabs(direction) < M_PI_4) {
                            // 横書き
                            if(run_mode == 2) continue;
                        }
                        else {
                            // 縦書き
                            if(run_mode == 1) continue;
                        }
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy;
                int y = yi / scale;
                int xi = boxes[boxid].cx + di;
                int x = xi / scale;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        if(fabs(direction) < M_PI_4) {
                            // 横書き
                            if(run_mode == 2) continue;
                        }
                        else {
                            // 縦書き
                            if(run_mode == 1) continue;
                        }
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy - di;
                int y = yi / scale;
                int xi = boxes[boxid].cx;
                int x = xi / scale;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        if(fabs(direction) < M_PI_4) {
                            // 横書き
                            if(run_mode == 2) continue;
                        }
                        else {
                            // 縦書き
                            if(run_mode == 1) continue;
                        }
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy + di;
                int y = yi / scale;
                int xi = boxes[boxid].cx;
                int x = xi / scale;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        if(fabs(direction) < M_PI_4) {
                            // 横書き
                            if(run_mode == 2) continue;
                        }
                        else {
                            // 縦書き
                            if(run_mode == 1) continue;
                        }
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
        }
    }
    return line_box_chain;
}

// 既に検出できた文字の大きさを元に、線を太くして判定範囲を広げる
void line_grow(
    int lineid_count,
    std::vector<double> &angle_map,
    std::vector<int> &lineid_map,
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker)
{
    fprintf(stderr, "line grow\n");
    std::vector<float> line_width(lineid_count);
    std::vector<int> lineid_map2(width*height, -1);

    for(int i = 0; i < lineid_count; i++) {
        float max_width = 0;
        for(const auto &boxid: line_box_chain[i]) {
            if (fabs(boxes[boxid].direction) < M_PI_4) {
                // 横書き
                max_width = std::max(max_width, boxes[boxid].h);
            }
            else {
                // 縦書き
                max_width = std::max(max_width, boxes[boxid].w);
            }
        }
        line_width[i] = max_width / scale;
    }
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(lineblocker[y * width + x]) continue;
            int lineid;
            if((lineid = lineid_map[y * width + x]) < 0) continue;
            lineid_map2[y * width + x] = lineid;
            float direction = angle_map[y * width + x];
            if(fabs(direction) < M_PI_4) {
                // 横書き
                if(run_mode == 2) continue;
            }
            else {
                // 縦書き
                if(run_mode == 1) continue;
            }
            if(fabs(direction) < M_PI_4) {
                // 横書き
                int max_width = line_width[lineid];
                for(int y2 = y; y2 >= std::max(0, y-max_width); y2--) {
                    if(lineblocker[y2 * width + x]) break;
                    if(lineid_map2[y2 * width + x] < 0) {
                        lineid_map2[y2 * width + x] = lineid;
                        angle_map[y2 * width + x] = direction;
                    }
                }
            }
            else {
                // 縦書き
                int max_width = line_width[lineid];
                for(int x2 = x; x2 >= std::max(0, x-max_width/2); x2--) {
                    if(lineblocker[y * width + x2]) break;
                    if(lineid_map2[y * width + x2] < 0) {
                        lineid_map2[y * width + x2] = lineid;
                        angle_map[y * width + x2] = direction;
                    }
                }
                for(int x2 = x; x2 < std::min(width, x+max_width/2+1); x2++) {
                    if(lineblocker[y * width + x2]) break;
                    if(lineid_map2[y * width + x2] < 0) {
                        lineid_map2[y * width + x2] = lineid;
                        angle_map[y * width + x2] = direction;
                    }
                }
            }
        }
    }
    lineid_map = lineid_map2;
}

int detect_line(
    std::vector<int> &lineid_map,
    const std::vector<float> &lineimage, 
    const std::vector<bool> &lineblocker)
{
    fprintf(stderr, "detect line\n");
    std::vector<int> idx(width*height);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&lineimage](auto a, auto b){
        return lineimage[a] > lineimage[b];
    });
    int lineid_count = 0;
    for(auto i: idx) {
        if(lineimage[i] < line_valueth) break;
        if(lineid_map[i] >= 0) continue;

        std::vector<int> stack;
        stack.push_back(i);
        while(!stack.empty()) {
            int i2 = stack.back();
            stack.pop_back();

            if(lineid_map[i2] >= 0) continue;
            if(lineblocker[i2]) continue;
            if(lineimage[i2] < line_valueth) continue;
            lineid_map[i2] = lineid_count;
            int x0 = i2 % width;
            int y0 = i2 / width;

            for(int y = y0-1; y <= y0+1; y++) {
                for(int x = x0-1; x <= x0+1; x++) {
                    if(x < 0 || x >= width || y < 0 || y >= height) continue;
                    int i3 = y * width + x;
                    if(lineid_map[i3] >= 0) continue;
                    if(lineblocker[i3]) continue;
                    if(lineimage[i3] < line_valueth) continue;
                    stack.push_back(i3);
                }
            }
        }
        lineid_count++;
    }
    return lineid_count;
}

void set_angle(
    int lineid_count,
    std::vector<double> &angle_map,
    const std::vector<int> &lineid_map)
{
    std::vector<std::vector<int>> lineid_list(lineid_count);
    std::vector<double> angle_list(lineid_count);
    for(int i = 0; i < lineid_map.size(); i++) {
        int id = lineid_map[i];
        if(id < 0) continue;
        lineid_list[id].push_back(i);
    }
    for (int line_id = 0; line_id < lineid_count; line_id++) {
        int max_x = 0;
        int min_x = width;
        int max_y = 0;
        int min_y = height;
        for(auto idx: lineid_list[line_id]) {
            int x = idx % width;
            int y = idx / width;
            max_x = std::max(x, max_x);
            min_x = std::min(x, min_x);
            max_y = std::max(y, max_y);
            min_y = std::min(y, min_y);
        }
        if(max_x - min_x < max_y - min_y) {
            angle_list[line_id] = M_PI_2;    
        }
        else {
            angle_list[line_id] = 0;
        }
    }
    for(int i = 0; i < angle_map.size(); i++) {
        int id = lineid_map[i];
        if(id < 0) continue;
        angle_map[i] = angle_list[id];
    }
}

std::vector<std::vector<int>> linefind(
    std::vector<charbox> &boxes,
    const std::vector<float> &lineimage, 
    const std::vector<bool> &lineblocker)
{
    std::vector<double> angle_map(width*height, std::nan(""));
    std::vector<int> lineid_map(width*height, -1);
    int lineid_count = detect_line(lineid_map, lineimage, lineblocker);
    set_angle(lineid_count, angle_map, lineid_map);

    std::vector<std::vector<int>> line_box_chain = chain_boxes(lineid_count, boxes, angle_map, lineid_map);
    line_grow(lineid_count, angle_map, lineid_map, boxes, line_box_chain, lineblocker);

    // 太くした線でもう一度、文字ボックスと線分の接触を検出して、ライン上にboxを並べる
    line_box_chain = chain_boxes(lineid_count, boxes, angle_map, lineid_map);

    fix_chain_info(boxes, line_box_chain);

    return line_box_chain;
}