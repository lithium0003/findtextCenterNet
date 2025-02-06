#include "after_search.h"
#include "ruby_search.h"
#include "number_unbind.h"
#include "make_block.h"
#include "search_loop.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>

// 短いチェーンは方向を修正しておく
void fix_shortchain(
    std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() < 3) {
            int id1 = line_box_chain[chainid].front();
            int id2 = line_box_chain[chainid].back();
            float diffx = fabs(boxes[id1].cx - boxes[id2].cx);
            float diffy = fabs(boxes[id1].cy - boxes[id2].cy);
            if(diffx > diffy) {
                // 横書き
                for(auto boxid: line_box_chain[chainid]) {
                    boxes[boxid].direction = 0;
                }
            }
            else {
                // 縦書き
                for(auto boxid: line_box_chain[chainid]) {
                    boxes[boxid].direction = M_PI_2;
                }
            }
        }
    }
}

// chain id を登録する
void register_chainid(
    std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        for(auto boxid: line_box_chain[chainid]) {
            boxes[boxid].idx = chainid;
            if (fabs(boxes[boxid].direction) < M_PI_4) {
                boxes[boxid].subtype &= ~1;
            }
            else {
                boxes[boxid].subtype |= 1;
            }
        }
    }
}

// 飛んでる番号があるので振り直す
int renumber_chain(
    std::vector<charbox> &boxes)
{
    std::vector<int> chain_remap;
    for(const auto &box: boxes) {
        if(box.idx < 0) continue;
        if(std::find(chain_remap.begin(), chain_remap.end(), box.idx) == chain_remap.end()) {
            chain_remap.push_back(box.idx);
        }
    }
    std::sort(chain_remap.begin(), chain_remap.end());
    for(auto &box: boxes) {
        if(box.idx < 0) continue;
        int id = std::distance(chain_remap.begin(), std::find(chain_remap.begin(), chain_remap.end(), box.idx));
        box.idx = id;
    }
    return chain_remap.size();
}

// 行の順番に番号を振り直す
int renumber_id(
    int id_max,
    std::vector<charbox> &boxes)
{
    struct lineparam {
        int d;
        float cx1;
        float cy1;
        float cx2;
        float cy2;
        int count;
        float size;
        int doubleline1;
        int doubleline2;
        int doubleline;
    };

    std::cerr << "renumber id" << std::endl;
    std::vector<lineparam> lineparams(id_max);
    std::vector<int> chain_remap;
    for(auto &box: boxes) {
        if(box.idx < 0) continue;
        if(std::find(chain_remap.begin(), chain_remap.end(), box.idx) == chain_remap.end()) {
            chain_remap.push_back(box.idx);
        }

        if((box.subtype & (2+4)) == 2+4) continue;
        if((box.subtype & 32) == 32) continue;
        if((box.subtype & 1) == 0) {
            // 横書き
            lineparams[box.idx].d = 0;
            // 一番左端の文字の座標を取る
            if(lineparams[box.idx].count == 0 || box.cx < lineparams[box.idx].cx1) {
                lineparams[box.idx].cx1 = box.cx;
                lineparams[box.idx].cy1 = box.cy;
            }
            // 一番右端の文字の座標を取る
            if(lineparams[box.idx].count == 0 || box.cx > lineparams[box.idx].cx2) {
                lineparams[box.idx].cx2 = box.cx;
                lineparams[box.idx].cy2 = box.cy;
            }
            if (box.double_line == 1) {
                lineparams[box.idx].doubleline1++;    
            }
            else if (box.double_line == 2) {
                lineparams[box.idx].doubleline2++;    
            }
            lineparams[box.idx].size = std::max(lineparams[box.idx].size, std::max(box.w, box.h));
        }
        else {
            // 縦書き
            lineparams[box.idx].d = 1;
            // 一番上端の座標を取る
            if(lineparams[box.idx].count == 0 || box.cy < lineparams[box.idx].cy1) {
                lineparams[box.idx].cx1 = box.cx;
                lineparams[box.idx].cy1 = box.cy;
            }
            // 一番下端の座標を取る
            if(lineparams[box.idx].count == 0 || box.cy > lineparams[box.idx].cy2) {
                lineparams[box.idx].cx2 = box.cx;
                lineparams[box.idx].cy2 = box.cy;
            }
            if (box.double_line == 1) {
                lineparams[box.idx].doubleline1++;    
            }
            else if (box.double_line == 2) {
                lineparams[box.idx].doubleline2++;    
            }
            lineparams[box.idx].size = std::max(lineparams[box.idx].size, std::max(box.w, box.h));
        }
        lineparams[box.idx].count++;
    }
    for(auto &p: lineparams) {
        if(p.doubleline1 > p.doubleline2) {
            if(p.doubleline1 > p.count / 2) {
                p.doubleline = 1;
            }
        }
        else {
            if(p.doubleline2 > p.count / 2) {
                p.doubleline = 2;
            }
        }
    }
    
    std::sort(chain_remap.begin(), chain_remap.end(), [&](const auto a, const auto b){
        // 横書きを優先する
        if(lineparams[a].d != lineparams[b].d)
            return lineparams[a].d < lineparams[b].d;

        if (lineparams[a].doubleline + lineparams[b].doubleline > 0) {
            float s = std::max(lineparams[a].size, lineparams[b].size);
            if(lineparams[a].d == 0) {
                // 横書き
                // 同じ行なら、左から
                if (fabs(lineparams[a].cy1 - lineparams[b].cy1) < s) {
                    // 左にある方が先
                    return lineparams[a].cx1 < lineparams[b].cx1;
                }
            }
            else {
                // 縦書き
                // 同じ行なら、上から
                if (fabs(lineparams[a].cx1 - lineparams[b].cx1) < s) {
                    // 上にある方が先
                    return lineparams[a].cy1 < lineparams[b].cy1;
                }
            }
        }

        float s = std::max(lineparams[a].size, lineparams[b].size) * 0.5;
        if(lineparams[a].d == 0) {
            // 横書き
            // 行が重なっている場合、縦方向にソート
            if(lineparams[a].cx1 < lineparams[b].cx2 + s || lineparams[b].cx1 < lineparams[a].cx2 + s) {
                // 上にある行が先
                return lineparams[a].cy1 < lineparams[b].cy1;
            }
            // 左にある方が先
            return lineparams[a].cx1 < lineparams[b].cx1;
        }
        else {
            // 縦書き
            // 行が重なっている場合、縦方向にソート
            if(lineparams[a].cy1 < lineparams[b].cy2 + s || lineparams[b].cy1 < lineparams[a].cy2 + s) {
                // 右にある行が先
                return lineparams[a].cx1 > lineparams[b].cx1;
            }
            // 上にある方が先
            return lineparams[a].cy1 < lineparams[b].cy1;
        }
    });
    for(auto &box: boxes) {
        if(box.idx < 0) continue;
        int id = std::distance(chain_remap.begin(), std::find(chain_remap.begin(), chain_remap.end(), box.idx));
        box.idx = id;
    }
    return chain_remap.size();
}

// ゴミっぽい外れの小さいboxを外す
void filter_box(
    int id_max,
    std::vector<charbox> &boxes)
{
    if(boxes.size() < 20) return;

    std::vector<std::vector<int>> linecount(id_max);
    std::vector<float> sizes;
    for(auto &box: boxes) {
        if(box.idx < 0) continue;
        linecount[box.idx].push_back(box.id);
        sizes.push_back(std::max(box.w, box.h));
    }
    float mean1_size = std::reduce(sizes.begin(), sizes.end()) / std::max(1.0, double(sizes.size()));
    std::vector<float> lsizes;
    std::copy_if(sizes.begin(), sizes.end(), std::back_inserter(lsizes), [mean1_size](auto x){
        return x > mean1_size;
    });
    float mean2_size = std::reduce(lsizes.begin(), lsizes.end()) / std::max(1.0, double(lsizes.size()));

    for(auto chain: linecount) {
        if(chain.size() != 1) continue;
        int idx = chain.front();
        if(std::max(boxes[idx].w, boxes[idx].h) > 25) continue;
        float cx = boxes[idx].cx;
        float cy = boxes[idx].cy;

        std::vector<float> dist;
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            if(box.id == idx) continue;
            float d = sqrt((box.cx - cx) * (box.cx - cx) + (box.cy - cy) * (box.cy - cy));
            dist.push_back(d);
        }

        float min_dist = std::reduce(dist.begin(), dist.end(), INFINITY, [](auto acc, auto i){
            return std::min(acc, i);
        });

        if(min_dist > mean2_size * 5) {
            // std::cerr << min_dist << "," << idx << std::endl;
            boxes[idx].idx = -1;
        }
    }
}

void after_search(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage)
{
    fprintf(stderr, "after_search\n");

    fix_shortchain(boxes, line_box_chain);
    register_chainid(boxes, line_box_chain);

    // ルビの検索
    search_ruby(boxes, line_box_chain, lineblocker, idimage);

    int id_max = renumber_chain(boxes);

    id_max = number_unbind(boxes, lineblocker, idimage, id_max);
    std::cerr << "id max " << id_max << std::endl;

    filter_box(id_max, boxes);

    id_max = renumber_id(id_max, boxes);

    make_block(id_max, boxes, lineblocker);

    fprintf(stderr, "after_search done\n");
}