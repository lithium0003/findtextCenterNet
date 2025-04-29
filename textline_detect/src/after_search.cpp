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

int chain_line_force(
    int id_max,
    std::vector<charbox> &boxes)
{
    if(chain_line_ratio <= 0) return id_max;
    
    std::vector<std::vector<int>> line_box_chain(id_max);
    for(const auto &box: boxes) {
        if(box.idx < 0) continue;
        line_box_chain[box.idx].push_back(-1);
    }
    for(const auto &box: boxes) {
        if(box.idx < 0) continue;
        line_box_chain[box.idx][box.subidx] = box.id;
    }

    for(auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        float direction = boxes[it->front()].direction;
        float ax1 = boxes[it->front()].cx;
        float ay1 = boxes[it->front()].cy;
        float ax2 = boxes[it->back()].cx;
        float ay2 = boxes[it->back()].cy;
        for(auto bit = it->rbegin(); bit != it->rend(); ++bit) {
            if((boxes[*bit].subtype & (2+4)) == 2+4) {
                continue;
            }
            ax2 = boxes[*bit].cx;
            ay2 = boxes[*bit].cy;
            break;
        }
        float s1 = 0;
        for(auto i: *it) {
            s1 = std::max(s1, std::max(boxes[i].w, boxes[i].h));
        }
        std::vector<std::pair<std::vector<std::vector<int>>::iterator, float>> dist_map;
        for(auto it2 = line_box_chain.begin(); it2 != line_box_chain.end(); ++it2) {
            if(it == it2) continue;
            if(it2->size() > 2) {
                float direction2 = boxes[it2->front()].direction;
                if(fabs(direction) < M_PI_4 && fabs(direction2) > M_PI_4) continue;
                if(fabs(direction) > M_PI_4 && fabs(direction2) < M_PI_4) continue;
            }
            else if(it2->size() > 1) {
                // 横書きの2文字は、縦中横の可能性があるので通す
                float direction2 = boxes[it2->front()].direction;
                if(fabs(direction) < M_PI_4 && fabs(direction2) > M_PI_4) continue;
            }
            
            float bx1 = boxes[it2->front()].cx;
            float by1 = boxes[it2->front()].cy;
            float bx2 = boxes[it2->back()].cx;
            float by2 = boxes[it2->back()].cy;
            for(auto bit = it2->rbegin(); bit != it2->rend(); ++bit) {
                if((boxes[*bit].subtype & (2+4)) == 2+4) {
                    continue;
                }
                bx2 = boxes[*bit].cx;
                by2 = boxes[*bit].cy;
                break;
            }

            if(fabs(direction) < M_PI_4) {
                // 横書き
                if(abs(ay1 - by2) < s1 && ax1 > bx2 && ax1 - bx2 < s1 * chain_line_ratio) {
                    // b -> a
                    dist_map.emplace_back(it2, ax1 - bx2);
                }
                if(abs(ay2 - by1) < s1 && ax2 > bx1 && ax2 - bx1 < s1 * chain_line_ratio) {
                    // a -> b
                    dist_map.emplace_back(it2, bx1 - ax2);
                }
            }
            else {
                // 縦書き
                if(abs(ax1 - bx2) < s1 && ay1 > by2 && ay1 - by2 < s1 * chain_line_ratio) {
                    // b -> a
                    dist_map.emplace_back(it2, ay1 - by2);
                }
                if(abs(ax2 - bx1) < s1 && ay2 > by1 && ay2 - by1 < s1 * chain_line_ratio) {
                    // a -> b
                    dist_map.emplace_back(it2, by1 - ay2);
                }
            }
        }
        std::sort(dist_map.begin(), dist_map.end(), [](const auto a, const auto b){
            return fabs(a.second) < fabs(b.second);
        });
        if(dist_map.empty()) {
            ++it;
            continue;
        }
        auto it2 = dist_map.front().first;
        auto d = dist_map.front().second;
        if(d < 0) {
            // a -> b
            std::copy(it2->begin(), it2->end(), std::back_inserter(*it));
            boxes[it2->front()].subtype |= 8 + 512;
            if(fabs(direction) < M_PI_4) {
                for(auto i: *it) {
                    boxes[i].subtype &= ~1;
                }
            }
            else {
                for(auto i: *it) {
                    boxes[i].subtype |= 1;
                }
            }
            line_box_chain.erase(it2);
            ++it;
        }
        else {
            // b -> a
            std::copy(it->begin(), it->end(), std::back_inserter(*it2));
            boxes[it->front()].subtype |= 8 + 512;
            if(fabs(direction) < M_PI_4) {
                for(auto i: *it2) {
                    boxes[i].subtype &= ~1;
                }
            }
            else {
                for(auto i: *it2) {
                    boxes[i].subtype |= 1;
                }
            }
            it = line_box_chain.erase(it);
        }
    }
    
    id_max = (int)line_box_chain.size();
    for(int lineid = 0; lineid < id_max; lineid++) {
        for(int subid = 0; subid < line_box_chain[lineid].size(); subid++) {
            int boxid = line_box_chain[lineid][subid];
            boxes[boxid].idx = lineid;
            boxes[boxid].subidx = subid;
        }
    }
    return id_max;
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

    id_max = chain_line_force(id_max, boxes);
    std::cerr << "id max " << id_max << std::endl;

    make_block(boxes, lineblocker);

    fprintf(stderr, "after_search done\n");
}
