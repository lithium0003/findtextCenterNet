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
        int id = (int)std::distance(chain_remap.begin(), std::find(chain_remap.begin(), chain_remap.end(), box.idx));
        box.idx = id;
    }
    return int(chain_remap.size());
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

    make_block(boxes, lineblocker);

    fprintf(stderr, "after_search done\n");
}
