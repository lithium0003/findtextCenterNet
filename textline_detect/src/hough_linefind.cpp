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
                max_width = std::max(max_width, boxes[boxid].h);
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
        if(lineblocker[i]) continue;

        int start_x = i % width;
        int start_y = i / width;
        
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

            std::vector<int> tmp;
            for(int y = y0-2; y <= y0+2; y++) {
                for(int x = x0-2; x <= x0+2; x++) {
                    if(x < 0 || x >= width || y < 0 || y >= height) continue;
                    if(run_mode == 1 && abs(y - start_y) > 10) continue;
                    if(run_mode == 2 && abs(x - start_x) > 10) continue;

                    int i3 = y * width + x;
                    if(lineid_map[i3] >= 0) continue;
                    if(lineblocker[i3]) goto next_loop;
                    if(lineimage[i3] < line_valueth) continue;
                    tmp.push_back(i3);
                }
            }
            std::copy(tmp.begin(), tmp.end(), std::back_inserter(stack));
            
            next_loop:
            ;
        }
        lineid_count++;
    }
    return lineid_count;
}

int set_angle(
    int lineid_count,
    std::vector<double> &angle_map,
    std::vector<int> &lineid_map)
{
    std::vector<std::vector<int>> lineid_lists(lineid_count);
    std::vector<double> angle_list;
    std::vector<std::vector<int>> valid_lineid_lists;
    for(int i = 0; i < lineid_map.size(); i++) {
        int id = lineid_map[i];
        if(id < 0) continue;
        lineid_lists[id].push_back(i);
    }
    for (const auto& lineid_list: lineid_lists) {
        int max_x = 0;
        int min_x = width;
        int max_y = 0;
        int min_y = height;
        for(auto i: lineid_list) {
            int x = i % width;
            int y = i / width;
            max_x = std::max(x, max_x);
            min_x = std::min(x, min_x);
            max_y = std::max(y, max_y);
            min_y = std::min(y, min_y);
        }
        if(max_x - min_x < max_y - min_y) {
            if(run_mode == 0 || run_mode == 2 || run_mode > 2) {
                std::pair<int,int> p1(width, height);
                std::pair<int,int> p2(0,0);
                for(auto i: lineid_list) {
                    int x = i % width;
                    int y = i / width;
                    if(p1.second > y) {
                        p1.first = x;
                        p1.second = y;
                    }
                    if(p2.second < y) {
                        p2.first = x;
                        p2.second = y;
                    }
                }
                float angle = atan2(p2.second - p1.second, p2.first - p1.first);
                angle_list.push_back(angle);
                valid_lineid_lists.push_back(lineid_list);
            }
        }
        else {
            if(run_mode == 0 || run_mode == 1 || run_mode > 2) {
                std::pair<int,int> p1(width, height);
                std::pair<int,int> p2(0,0);
                for(auto i: lineid_list) {
                    int x = i % width;
                    int y = i / width;
                    if(p1.first > x) {
                        p1.first = x;
                        p1.second = y;
                    }
                    if(p2.first < x) {
                        p2.first = x;
                        p2.second = y;
                    }
                }
                float angle = atan2(p2.second - p1.second, p2.first - p1.first);
                angle_list.push_back(angle);
                valid_lineid_lists.push_back(lineid_list);
            }
        }
    }
    std::fill(lineid_map.begin(), lineid_map.end(), -1);
    for (int line_id = 0; line_id < valid_lineid_lists.size(); line_id++) {
        for(const auto i: valid_lineid_lists[line_id]) {
            angle_map[i] = angle_list[line_id];
            lineid_map[i] = line_id;
        }
    }
    return (int)valid_lineid_lists.size();
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
