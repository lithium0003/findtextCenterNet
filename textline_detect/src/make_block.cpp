#include "make_block.h"
#include "search_loop.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <cassert>

struct lineparam {
    int d;
    int doubleline;
    int count;
    float size;
};

// 行を処理する
void process_line(
    int id_max,
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &chain_next,
    std::vector<std::vector<int>> &chain_prev,
    const std::vector<int> &chainid_map,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<lineparam> &lineparams,
    const std::vector<bool> &lineblocker)
{
    const double scanwidth_next_block = 0.5 + allowwidth_next_block;

    for(int chainid = 0; chainid < id_max; chainid++) {
        //std::cerr << "chain " << chainid << std::endl;

        if(lineparams[chainid].d == 2 || (lineparams[chainid].d == 0 && run_mode == 1)) {
            // 横書き
            float s_s = 0;
            for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                s_s = std::max(s_s, std::max(boxes[line_box_chain[chainid][i]].w, boxes[line_box_chain[chainid][i]].h));
            }

            float cx1 = -1;
            float cy1 = -1;
            float cx2 = -1;
            float cy2 = -1;

            for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                float s = std::max(boxes[line_box_chain[chainid][i]].w, boxes[line_box_chain[chainid][i]].h);

                if(fabs(s - s_s)/std::min(s, s_s) > 0.5) continue;
                if((boxes[line_box_chain[chainid][i]].subtype & (2+4)) == 2+4) continue;
                if(boxes[line_box_chain[chainid][i]].double_line > 0) continue;

                if(cx1 < 0 && cy1 < 0) {
                    cx1 = boxes[line_box_chain[chainid][i]].cx - boxes[line_box_chain[chainid][i]].w / 2;
                    cy1 = boxes[line_box_chain[chainid][i]].cy;
                }
                cx2 = boxes[line_box_chain[chainid][i]].cx + boxes[line_box_chain[chainid][i]].w / 2;
                cy2 = boxes[line_box_chain[chainid][i]].cy;
            }
            
            if(cx2 - cx1 < scale) continue;
            
            float a = (cy2 - cy1)/(cx2 - cx1);
            
            for(int x = (cx1 + cx2)/2; x < cx2 + s_s; x++) {
                int y = a * (x - cx1) + cy1;
                int xi = x / scale;
                int yi = y / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) break;
                
                for(int yp = yi; yp < yi + s_s / scale * scanwidth_next_block; yp++) {
                    if(yp < 0 || yp >= height) continue;

                    if(lineblocker[yp * width + xi]) break;
                    int other_chain = chainid_map[yp * width + xi];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 1) break;
                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::min(lineparams[chainid].size, lineparams[other_chain].size) > allow_sizediff) continue;

                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        chain_next[chainid].push_back(other_chain);
                        chain_prev[other_chain].push_back(chainid);
                        break;
                    }
                }
            }
            for(int x = (cx1 + cx2)/2; x > cx1 - s_s; x--) {
                int y = a * (x - cx1) + cy1;
                int xi = x / scale;
                int yi = y / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) break;
                
                for(int yp = yi; yp < yi + s_s / scale * scanwidth_next_block; yp++) {
                    if(yp < 0 || yp >= height) continue;

                    if(lineblocker[yp * width + xi]) break;
                    int other_chain = chainid_map[yp * width + xi];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 1) break;
                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::min(lineparams[chainid].size, lineparams[other_chain].size) > allow_sizediff) continue;

                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        chain_next[chainid].push_back(other_chain);
                        chain_prev[other_chain].push_back(chainid);
                        break;
                    }
                }
            }
        }
        else if (lineparams[chainid].d == 1 || (lineparams[chainid].d == 0 && run_mode == 2)){
            // 縦書き
            float s_s = 0;
            for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                s_s = std::max(s_s, std::max(boxes[line_box_chain[chainid][i]].w, boxes[line_box_chain[chainid][i]].h));
            }

            float cx1 = -1;
            float cy1 = -1;
            float cx2 = -1;
            float cy2 = -1;

            for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                float s = std::max(boxes[line_box_chain[chainid][i]].w, boxes[line_box_chain[chainid][i]].h);

                if(fabs(s - s_s)/std::min(s, s_s) > 0.5) continue;
                if((boxes[line_box_chain[chainid][i]].subtype & (2+4)) == 2+4) continue;
                if(boxes[line_box_chain[chainid][i]].double_line > 0) continue;

                if(cx1 < 0 && cy1 < 0) {
                    cx1 = boxes[line_box_chain[chainid][i]].cx;
                    cy1 = boxes[line_box_chain[chainid][i]].cy - boxes[line_box_chain[chainid][i]].h / 2;
                }
                cx2 = boxes[line_box_chain[chainid][i]].cx;
                cy2 = boxes[line_box_chain[chainid][i]].cy + boxes[line_box_chain[chainid][i]].h / 2;
            }
            
            if(cy2 - cy1 < scale) continue;
            
            float a = (cx2 - cx1)/(cy2 - cy1);
            
            for(int y = (cy1 + cy2)/2; y < cy2 + s_s; y++) {
                int x = a * (y - cy1) + cx1;
                int xi = x / scale;
                int yi = y / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) break;
                
                for(int xp = xi; xp > xi - s_s / scale * scanwidth_next_block; xp--) {
                    if(xp < 0 || xp >= width) continue;

                    if(lineblocker[yi * width + xp]) break;
                    int other_chain = chainid_map[yi * width + xp];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 2) break;

                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > allow_sizediff)
                        continue;

                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        chain_next[chainid].push_back(other_chain);
                        chain_prev[other_chain].push_back(chainid);
                        break;
                    }
                }
            }
            for(int y = (cy1 + cy2)/2; y > cy1 - s_s; y--) {
                int x = a * (y - cy1) + cx1;
                int xi = x / scale;
                int yi = y / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) break;
                
                for(int xp = xi; xp > xi - s_s / scale * scanwidth_next_block; xp--) {
                    if(xp < 0 || xp >= width) continue;

                    if(lineblocker[yi * width + xp]) break;
                    int other_chain = chainid_map[yi * width + xp];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 2) break;

                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > allow_sizediff)
                        continue;

                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        chain_next[chainid].push_back(other_chain);
                        chain_prev[other_chain].push_back(chainid);
                        break;
                    }
                }
            }
        }
        else {
            // 短いのでどちらか不明
        }
    }
}

// ブロックに含まれるchainの検索
std::vector<std::vector<int>> block_chain_search(
    int id_max,
    const std::vector<std::vector<int>> &chain_next,
    const std::vector<std::vector<int>> &chain_prev)
{
    std::vector<std::vector<int>> block_chain;
    std::cerr << "block chain check" << std::endl;
    std::vector<int> chain_root;
    for(int cur_id = 0; cur_id < id_max; cur_id++){
        if(chain_prev[cur_id].empty()) {
            if (std::find(chain_root.begin(), chain_root.end(), cur_id) == chain_root.end()) {
                chain_root.push_back(cur_id);
            }
            continue;
        }
    }


    for(auto cur_id: chain_root) {
        std::vector<int> stack;
        stack.push_back(cur_id);
        std::vector<int> tmp_block;
        std::vector<int> done_block;
        for(auto chain: block_chain) {
            std::copy(chain.begin(), chain.end(), std::back_inserter(done_block));
        }
        while(stack.size() > 0) {
            auto j = stack.back();
            stack.pop_back();

            if(std::find(done_block.begin(), done_block.end(), j) != done_block.end()) continue;
            if(std::find(tmp_block.begin(), tmp_block.end(), j) != tmp_block.end()) continue;

            tmp_block.push_back(j);
            for(const auto chainid: chain_next[j]) {
                if(std::find(stack.begin(), stack.end(), chainid) == stack.end()) {
                    stack.push_back(chainid);
                }
            }
        }
        std::sort(tmp_block.begin(), tmp_block.end());
        tmp_block.erase(std::unique(tmp_block.begin(), tmp_block.end()), tmp_block.end());
        block_chain.push_back(tmp_block);
    }

    return block_chain;
}

bool rechain_search(
    std::vector<std::vector<int>> &line_box_chain,
    std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &chain_next,
    const std::vector<std::vector<int>> &chain_prev)
{
    bool ret = true;
    if(std::count_if(chain_next.begin(), chain_next.end(), [](const auto x){ return x.size() > 1; }) > 0) {
        for(int i = 0; i < chain_next.size(); i++) {
            if(chain_next[i].size() <= 1) {
                continue;
            }
            
            std::vector<int> agg_idx;
            agg_idx.push_back(i);
            std::vector<int> tmp;
            std::copy(chain_next[i].begin(), chain_next[i].end(), std::back_inserter(tmp));
            while(!tmp.empty()) {
                int j = tmp.back();
                tmp.pop_back();
                if(std::find(agg_idx.begin(), agg_idx.end(), j) == agg_idx.end()) {
                    agg_idx.push_back(j);
                    std::copy(chain_next[j].begin(), chain_next[j].end(), std::back_inserter(tmp));
                }
            }
            
            std::sort(agg_idx.begin(), agg_idx.end());
            agg_idx.erase(std::unique(agg_idx.begin(), agg_idx.end()), agg_idx.end());
            for(int j = 0; j < agg_idx.size(); j++) {
                for(int k = 0; k < agg_idx.size(); k++) {
                    if (j == k) continue;
                    int n = agg_idx[j];
                    int m = agg_idx[k];
                    if (line_box_chain[n].empty()) continue;
                    if (line_box_chain[m].empty()) continue;
                    int n_i = line_box_chain[n].front();
                    int m_i = line_box_chain[m].front();
                    float size = 0.0f;
                    size = std::transform_reduce(
                        line_box_chain[n].begin(),
                        line_box_chain[n].end(),
                        size,
                        [&](float acc, float i) { return std::max(acc, i); },
                        [&](int x) { return std::max(boxes[x].w,boxes[x].h); });
                    size = std::transform_reduce(
                        line_box_chain[m].begin(),
                        line_box_chain[m].end(),
                        size,
                        [&](float acc, float i) { return std::max(acc, i); },
                        [&](int x) { return std::max(boxes[x].w,boxes[x].h); });
                    if((boxes[n_i].subtype & 1) == (boxes[m_i].subtype & 1)) {
                        if((boxes[n_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                    else if (line_box_chain[n].size() < 2) {
                        if((boxes[m_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                    else if (line_box_chain[m].size() < 2) {
                        if((boxes[n_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if(!ret) return ret;
    
    if(std::count_if(chain_prev.begin(), chain_prev.end(), [](const auto x){ return x.size() > 1; }) > 0) {
        for(int i = 0; i < chain_prev.size(); i++) {
            if(chain_prev[i].size() <= 1) {
                continue;
            }
            
            std::vector<int> agg_idx;
            agg_idx.push_back(i);
            std::vector<int> tmp;
            std::copy(chain_prev[i].begin(), chain_prev[i].end(), std::back_inserter(tmp));
            while(!tmp.empty()) {
                int j = tmp.back();
                tmp.pop_back();
                if(std::find(agg_idx.begin(), agg_idx.end(), j) == agg_idx.end()) {
                    agg_idx.push_back(j);
                    std::copy(chain_prev[j].begin(), chain_prev[j].end(), std::back_inserter(tmp));
                }
            }
            
            std::sort(agg_idx.begin(), agg_idx.end());
            agg_idx.erase(std::unique(agg_idx.begin(), agg_idx.end()), agg_idx.end());
            for(int j = 0; j < agg_idx.size(); j++) {
                for(int k = 0; k < agg_idx.size(); k++) {
                    if (j == k) continue;
                    int n = agg_idx[j];
                    int m = agg_idx[k];
                    if (line_box_chain[n].empty()) continue;
                    if (line_box_chain[m].empty()) continue;
                    int n_i = line_box_chain[n].front();
                    int m_i = line_box_chain[m].front();
                    float size = 0.0f;
                    size = std::transform_reduce(
                        line_box_chain[n].begin(),
                        line_box_chain[n].end(),
                        size,
                        [&](float acc, float i) { return std::max(acc, i); },
                        [&](int x) { return std::max(boxes[x].w,boxes[x].h); });
                    size = std::transform_reduce(
                        line_box_chain[m].begin(),
                        line_box_chain[m].end(),
                        size,
                        [&](float acc, float i) { return std::max(acc, i); },
                        [&](int x) { return std::max(boxes[x].w,boxes[x].h); });
                    if((boxes[n_i].subtype & 1) == (boxes[m_i].subtype & 1)) {
                        if((boxes[n_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                    else if (line_box_chain[n].size() < 2) {
                        if((boxes[m_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                    else if (line_box_chain[m].size() < 2) {
                        if((boxes[n_i].subtype & 1) == 0) {
                            // 横書き
                            if(boxes[line_box_chain[n].back()].cx < boxes[line_box_chain[m].front()].cx && fabs(boxes[line_box_chain[n].back()].cy - boxes[line_box_chain[m].front()].cy) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cx < boxes[line_box_chain[n].front()].cx && fabs(boxes[line_box_chain[m].back()].cy - boxes[line_box_chain[n].front()].cy) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                        else {
                            // 縦書き
                            if(boxes[line_box_chain[n].back()].cy < boxes[line_box_chain[m].front()].cy && fabs(boxes[line_box_chain[n].back()].cx - boxes[line_box_chain[m].front()].cx) < size) {
                                boxes[line_box_chain[m].front()].subtype |= 8;
                                std::copy(line_box_chain[m].begin(),line_box_chain[m].end(), std::back_inserter(line_box_chain[n]));
                                line_box_chain[m].clear();
                                ret = false;
                                continue;
                            }
                            if(boxes[line_box_chain[m].back()].cy < boxes[line_box_chain[n].front()].cy && fabs(boxes[line_box_chain[m].back()].cx - boxes[line_box_chain[n].front()].cx) < size) {
                                boxes[line_box_chain[n].front()].subtype |= 8;
                                std::copy(line_box_chain[n].begin(),line_box_chain[n].end(), std::back_inserter(line_box_chain[m]));
                                line_box_chain[n].clear();
                                ret = false;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return ret;
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
        int section;
        int secidx;
        int doubleline1;
        int doubleline2;
        int doubleline;
    };

    struct rect {
        float cx1;
        float cx2;
        float cy1;
        float cy2;
    };
    
    std::cerr << "renumber id" << std::endl;
    std::vector<lineparam> lineparams;
    std::vector<int> chain_remap;
    lineparams.resize(id_max);
    int major_direction = 0;
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
            major_direction++;
            if(lineparams[box.idx].count == 0 || box.cx - box.w/2 < lineparams[box.idx].cx1) {
                lineparams[box.idx].cx1 = box.cx - box.w/2;
            }
            if(lineparams[box.idx].count == 0 || box.cy - box.h/2 < lineparams[box.idx].cy1) {
                lineparams[box.idx].cy1 = box.cy - box.h/2;
            }
            if(lineparams[box.idx].count == 0 || box.cx + box.w/2 > lineparams[box.idx].cx2) {
                lineparams[box.idx].cx2 = box.cx + box.w/2;
            }
            if(lineparams[box.idx].count == 0 || box.cy + box.h/2 > lineparams[box.idx].cy2) {
                lineparams[box.idx].cy2 = box.cy + box.h/2;
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
            major_direction--;
            if(lineparams[box.idx].count == 0 || box.cx - box.w/2 < lineparams[box.idx].cx1) {
                lineparams[box.idx].cx1 = box.cx - box.w/2;
            }
            if(lineparams[box.idx].count == 0 || box.cy - box.h/2 < lineparams[box.idx].cy1) {
                lineparams[box.idx].cy1 = box.cy - box.h/2;
            }
            if(lineparams[box.idx].count == 0 || box.cx + box.w/2 > lineparams[box.idx].cx2) {
                lineparams[box.idx].cx2 = box.cx + box.w/2;
            }
            if(lineparams[box.idx].count == 0 || box.cy + box.h/2 > lineparams[box.idx].cy2) {
                lineparams[box.idx].cy2 = box.cy + box.h/2;
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
        if(p.count == 1) {
            p.d = (major_direction >= 0)? 0: 1;
        }
    }
    
    {
        std::vector<rect> section_param;
        int section = 0;
        section_param.emplace_back(width*scale,0,height*scale,0);

        std::sort(chain_remap.begin(), chain_remap.end());
        // 横書きを優先する
        auto it3 = std::partition(chain_remap.begin(), chain_remap.end(), [&](const auto x){
            return lineparams[x].d == 0;
        });
        // 横書き
        auto it1 = chain_remap.begin();
        auto it2 = it3;
        while(it1 != it2) {
            std::sort(it1, it2, [&](const auto a, const auto b){
                // 上を先に
                return lineparams[a].cy1 < lineparams[b].cy1;
            });
            // y方向にほぼ完全に重なっているブロックを探索する
            float cy1 = lineparams[*it1].cy1 - lineparams[*it1].size * float(allowwidth_next_block);
            float cy2 = lineparams[*it1].cy2 + lineparams[*it1].size * float(allowwidth_next_block);
            auto it4 = std::partition(it1, it2, [&](const auto x){
                return std::min(cy2, lineparams[x].cy2) - std::max(cy1, lineparams[x].cy1) > 0;
            });
            // yで重なっているブロックはいない
            if(it4 == it1) {
                int s = section;
                lineparams[*it1].section = s;
                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it1].cx1);
                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it1].cx2);
                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it1].cy1);
                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it1].cy2);
                ++it1;
                continue;
            }
            // これらのブロックのxの範囲を探索
            float cx1 = lineparams[*it1].cx1;
            float cx2 = lineparams[*it1].cx2;
            for(auto it5 = it1; it5 != it4; ++it5) {
                cx1 = std::min(cx1, lineparams[*it5].cx1);
                cx2 = std::max(cx2, lineparams[*it5].cx2);
            }
            std::sort(it1, it2, [&](const auto a, const auto b){
                // 左を先に
                return lineparams[a].cx1 < lineparams[b].cx1;
            });
            // このブロックにx座標が含まれるブロックを検索
            auto it5 = std::partition(it1, it2, [&](const auto x){
                return lineparams[x].cx1 <= cx2 && lineparams[x].cx2 >= cx1;
            });
            auto it52 = it5;
            do {
                it52 = it5;
                // これらのブロックのxの範囲を探索
                for(auto it53 = it1; it53 != it5; ++it53) {
                    cx1 = std::min(cx1, lineparams[*it53].cx1);
                    cx2 = std::max(cx2, lineparams[*it53].cx2);
                }
                // このブロックにx座標が含まれるブロックを検索
                it5 = std::partition(it1, it2, [&](const auto x){
                    return lineparams[x].cx1 < cx2 && lineparams[x].cx2 > cx1;
                });
            } while(it52 != it5);
            if(it5 == it1) {
                int s = section;
                lineparams[*it1].section = s;
                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it1].cx1);
                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it1].cx2);
                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it1].cy1);
                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it1].cy2);
                ++it1;
                continue;
            }
            // 含まれているブロック内で検索
            int block_section = section;
            auto it6 = it1;
            while(it6 != it5) {
                std::sort(it6, it5, [&](const auto a, const auto b){
                    // 上を先に
                    return lineparams[a].cy1 < lineparams[b].cy1;
                });
                cy1 = lineparams[*it6].cy1;
                cy2 = lineparams[*it6].cy2;
                // y方向に重なっている行を探索
                auto it7 = std::partition(it6, it5, [&](const auto x){
                    return std::min(cy2, lineparams[x].cy2) - std::max(cy1, lineparams[x].cy1) > 0;
                });
                if(it7 == it6) {
                    int s = block_section;
                    lineparams[*it6].section = s;
                    section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it6].cx1);
                    section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it6].cx2);
                    section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it6].cy1);
                    section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it6].cy2);
                    ++it6;
                    continue;
                }
                else if(std::distance(it6, it7) > 1) {
                    // yで重なっている行を左からソート
                    std::sort(it6, it7, [&](const auto a, const auto b){
                        // 左を先に
                        return lineparams[a].cx1 < lineparams[b].cx1;
                    });
                    if(section_param.size()-1 == section) {
                        // 段を追加
                        block_section = (int)section_param.size();
                        section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                    }
                    for(auto it8 = it6; it8 != it7; ++it8) {
                        bool found = false;
                        for(int s = block_section; s < section_param.size(); s++) {
                            // 横に重なっている
                            if (std::min(section_param[s].cx2, lineparams[*it8].cx2) - std::max(section_param[s].cx1, lineparams[*it8].cx1) > 0) {
                                lineparams[*it8].section = s;
                                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it8].cx1);
                                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it8].cx2);
                                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it8].cy1);
                                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it8].cy2);
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            // 段を追加
                            lineparams[*it8].section = (int)section_param.size();
                            section_param.emplace_back(lineparams[*it8].cx1, lineparams[*it8].cx2, lineparams[*it8].cy1, lineparams[*it8].cy2);
                        }
                    }
                }
                else {
                    if(section_param.size()-1 == section) {
                        // 段を追加
                        block_section = (int)section_param.size();
                        lineparams[*it6].section = block_section;
                        section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                    }
                    else {
                        bool found = false;
                        for(int s = block_section; s < section_param.size(); s++) {
                            // 横に重なっている
                            if (std::min(section_param[s].cx2, lineparams[*it6].cx2) - std::max(section_param[s].cx1, lineparams[*it6].cx1) > 0) {
                                lineparams[*it6].section = s;
                                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it6].cx1);
                                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it6].cx2);
                                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it6].cy1);
                                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it6].cy2);
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            // 段を消す
                            block_section = (int)section_param.size();
                            lineparams[*it6].section = block_section;
                            section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                        }
                    }
                }
                it6 = it7;
            }
            section = block_section;

            // 段をソートしておく
            std::vector<int> section_renum(section_param.size());
            std::iota(section_renum.begin(), section_renum.end(), 0);
            std::sort(section_renum.begin(), section_renum.end(), [&](const auto a, const auto b){
                // 左を先に
                return section_param[a].cx1 < section_param[b].cx1;
            });
            auto sit1 = section_renum.begin();
            while(sit1 != section_renum.end()) {
                float sx1 = section_param[*sit1].cx1;
                float sx2 = section_param[*sit1].cx2;
                // x方向に重なっている行を探索
                auto sit2 = std::partition(sit1, section_renum.end(), [&](const auto x){
                    return std::min(sx2, section_param[x].cx2) - std::max(sx1, section_param[x].cx1) > 0;
                });
                if (sit2 == sit1) {
                    ++sit1;
                    continue;
                }
                if(std::distance(sit1, sit2) > 1) {
                    std::sort(sit1, sit2, [&](const auto a, const auto b){
                        // 上を先に
                        return section_param[a].cy1 < section_param[b].cy1;
                    });
                }
                sit1 = sit2;
            }
            std::sort(section_param.begin(), section_param.end(), [](const auto a, const auto b){
                // 左を先に
                return a.cx1 < b.cx1;
            });
            auto pit1 = section_param.begin();
            while(pit1 != section_param.end()) {
                float sx1 = pit1->cx1;
                float sx2 = pit1->cx2;
                // x方向に重なっている行を探索
                auto pit2 = std::partition(pit1, section_param.end(), [&](const auto x){
                    return std::min(sx2, x.cx2) - std::max(sx1, x.cx1) > 0;
                });
                if (pit2 == pit1) {
                    ++pit1;
                    continue;
                }
                if(std::distance(pit1, pit2) > 1) {
                    std::sort(pit1, pit2, [](const auto a, const auto b){
                        // 上を先に
                        return a.cy1 < b.cy1;
                    });
                }
                pit1 = pit2;
            }

            // 番号を振り直す
            for(auto &lp: lineparams) {
                auto sit = std::find(section_renum.begin(), section_renum.end(), lp.section);
                assert(sit != section_renum.end());
                lp.section = (int)std::distance(section_renum.begin(), sit);
            }

            // 表か段組かを判定する
            // 段組なら、縦書きなので右端が一致しているはず
            std::vector<float> sec_bottom(section_param.size());
            for(auto cit = it1; cit != it5; ++cit) {
                int s = lineparams[*cit].section;
                sec_bottom[s] = std::max(sec_bottom[s], lineparams[*cit].cy2);
            }
            std::vector<float> valid_sec_bottom;
            std::copy_if(sec_bottom.begin(), sec_bottom.end(), std::back_inserter(valid_sec_bottom), [](auto x) { return x > 0; });
            if(valid_sec_bottom.size() > 1) {
                float s = lineparams[*it1].size * 2;
                float b = std::reduce(valid_sec_bottom.begin(), valid_sec_bottom.end(), 0.0f, [](auto acc, auto x){ return std::max(acc, x); });
                int c = (int)std::count_if(valid_sec_bottom.begin(), valid_sec_bottom.end(), [b,s](auto x){ return fabs(b-x)<s*2; });
                
                if(c > 1) {
                    // 多分段組
                    std::sort(it1, it5, [&](const auto a, const auto b){
                        // 上を先に
                        return lineparams[a].cy1 < lineparams[b].cy1;
                    });
                    std::stable_sort(it1, it5, [&](const auto a, const auto b){
                        // 段の順にソートする
                        return lineparams[a].section < lineparams[b].section;
                    });
                    for(auto ait = it1; ait != it5; ++ait) {
                        lineparams[*ait].secidx = lineparams[*ait].section;
                    }
                }
                else {
                    // 多分表が混ざっている
                    std::sort(it1, it5, [&](const auto a, const auto b){
                        // 段の順にソートする
                        return lineparams[a].section < lineparams[b].section;
                    });
                    int secidx = lineparams[*it1].section;
                    auto it8 = it1;
                    while(it8 != it5) {
                        // 最初の段でまずソート
                        std::sort(it8, it5, [&](const auto a, const auto b){
                            // 段の順にソートする
                            return lineparams[a].section < lineparams[b].section;
                        });
                        auto it7 = std::partition(it8, it5, [&](const auto x){
                            return lineparams[*it8].section == lineparams[x].section;
                        });
                        std::sort(it8, it7, [&](const auto a, const auto b){
                            // 上を先に
                            return lineparams[a].cy1 < lineparams[b].cy1;
                        });
                        if(lineparams[*it8].section > secidx) {
                            for(; it8 != it7; ++it8) {
                                // yに重なっている、いっこ上の段があるはず
                                auto it9 = std::find_if(it1, it8, [&](const auto x) {
                                    return lineparams[*it8].section == lineparams[x].section + 1 && std::min(lineparams[*it8].cy2, lineparams[x].cy2) - std::max(lineparams[*it8].cy1, lineparams[x].cy1) > 0;
                                });
                                if (it9 == it8) {
                                    // ないので末尾に足す
                                }
                                else {
                                    // その後ろに追加する
                                    auto tmp = *it8;
                                    std::copy_backward(it9+1, it8, it8+1);
                                    *(it9+1) = tmp;
                                }
                            }
                        }
                        else {
                            it8 = it7;
                        }
                    }
                    for(auto ait = it1; ait != it5; ++ait) {
                        lineparams[*ait].secidx = secidx;
                    }
                }
            }
            else {
                // 普通に上からソート
                std::sort(it1, it5, [&](const auto a, const auto b){
                    // 上を先に
                    return lineparams[a].cy1 < lineparams[b].cy1;
                });
                for(auto ait = it1; ait != it5; ++ait) {
                    lineparams[*ait].secidx = lineparams[*ait].section;
                }
            }
            
            section = (int)section_param.size();
            section_param.emplace_back(width*scale,0,height*scale,0);
            it1 = it5;
        }
        it1 = it3;
        it2 = chain_remap.end();
        int section2 = section;
        // 縦書き
        while(it1 != it2) {
            std::sort(it1, it2, [&](const auto a, const auto b){
                // 右を先に
                return lineparams[a].cx2 > lineparams[b].cx2;
            });
            // x方向に重なっているブロックを探索する
            float cx1 = lineparams[*it1].cx1 - lineparams[*it1].size * float(allowwidth_next_block);
            float cx2 = lineparams[*it1].cx2 + lineparams[*it1].size * float(allowwidth_next_block);
            auto it4 = std::partition(it1, it2, [&](const auto x){
                return std::min(cx2, lineparams[x].cx2) - std::max(cx1, lineparams[x].cx1) > 0;
            });
            // xで重なっているブロックはいない
            if(it4 == it1) {
                int s = section;
                lineparams[*it1].section = s;
                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it1].cx1);
                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it1].cx2);
                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it1].cy1);
                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it1].cy2);
                ++it1;
                continue;
            }
            // これらのブロックのyの範囲を探索
            float cy1 = lineparams[*it1].cy1;
            float cy2 = lineparams[*it1].cy2;
            for(auto it5 = it1; it5 != it4; ++it5) {
                cy1 = std::min(cy1, lineparams[*it5].cy1);
                cy2 = std::max(cy2, lineparams[*it5].cy2);
            }
            std::sort(it1, it2, [&](const auto a, const auto b){
                // 上を先に
                return lineparams[a].cy1 < lineparams[b].cy1;
            });
            // このブロックにy座標が含まれるブロックを検索
            auto it5 = std::partition(it1, it2, [&](const auto x){
                return lineparams[x].cy1 <= cy2 && lineparams[x].cy2 >= cy1;
            });
            auto it52 = it5;
            do {
                it52 = it5;
                // これらのブロックのyの範囲を探索
                for(auto it53 = it1; it53 != it5; ++it53) {
                    cy1 = std::min(cy1, lineparams[*it53].cy1);
                    cy2 = std::max(cy2, lineparams[*it53].cy2);
                }
                // このブロックにy座標が含まれるブロックを検索
                it5 = std::partition(it1, it2, [&](const auto x){
                    return lineparams[x].cy1 <= cy2 && lineparams[x].cy2 >= cy1;
                });
            } while(it52 != it5);
            if(it5 == it1) {
                int s = section;
                lineparams[*it1].section = s;
                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it1].cx1);
                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it1].cx2);
                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it1].cy1);
                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it1].cy2);
                ++it1;
                continue;
            }
            // 含まれているブロック内で検索
            int block_section = section;
            auto it6 = it1;
            while(it6 != it5) {
                std::sort(it6, it5, [&](const auto a, const auto b){
                    // 右を先に
                    return lineparams[a].cx2 > lineparams[b].cx2;
                });
                cx1 = lineparams[*it6].cx1;
                cx2 = lineparams[*it6].cx2;
                // x方向に重なっている行を探索
                auto it7 = std::partition(it6, it5, [&](const auto x){
                    return std::min(cx2, lineparams[x].cx2) - std::max(cx1, lineparams[x].cx1) > 0;
                });
                if(it7 == it6) {
                    int s = block_section;
                    lineparams[*it6].section = s;
                    section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it6].cx1);
                    section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it6].cx2);
                    section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it6].cy1);
                    section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it6].cy2);
                    ++it6;
                    continue;
                }
                if(std::distance(it6, it7) > 1) {
                    // xで重なっている行を上からソート
                    std::sort(it6, it7, [&](const auto a, const auto b){
                        // 上を先に
                        return lineparams[a].cy1 < lineparams[b].cy1;
                    });
                    if(section_param.size()-1 == section) {
                        // 段を追加
                        block_section = (int)section_param.size();
                        section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                    }
                    for(auto it8 = it6; it8 != it7; ++it8) {
                        bool found = false;
                        for(int s = block_section; s < section_param.size(); s++) {
                            // 縦に重なっている
                            if (std::min(section_param[s].cy2, lineparams[*it8].cy2) - std::max(section_param[s].cy1, lineparams[*it8].cy1) > 0) {
                                lineparams[*it8].section = s;
                                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it8].cx1);
                                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it8].cx2);
                                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it8].cy1);
                                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it8].cy2);
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            // 段を追加
                            lineparams[*it8].section = (int)section_param.size();
                            section_param.emplace_back(lineparams[*it8].cx1, lineparams[*it8].cx2, lineparams[*it8].cy1, lineparams[*it8].cy2);
                        }
                    }
                }
                else {
                    if(section_param.size()-1 == section) {
                        // 段を追加
                        block_section = (int)section_param.size();
                        lineparams[*it6].section = block_section;
                        section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                    }
                    else {
                        bool found = false;
                        for(int s = block_section; s < section_param.size(); s++) {
                            if (std::min(section_param[s].cy2, lineparams[*it6].cy2) - std::max(section_param[s].cy1, lineparams[*it6].cy1) > 0) {
                                lineparams[*it6].section = s;
                                section_param[s].cx1 = std::min(section_param[s].cx1, lineparams[*it6].cx1);
                                section_param[s].cx2 = std::max(section_param[s].cx2, lineparams[*it6].cx2);
                                section_param[s].cy1 = std::min(section_param[s].cy1, lineparams[*it6].cy1);
                                section_param[s].cy2 = std::max(section_param[s].cy2, lineparams[*it6].cy2);
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            // 段を消す
                            block_section = (int)section_param.size();
                            lineparams[*it6].section = block_section;
                            section_param.emplace_back(lineparams[*it6].cx1, lineparams[*it6].cx2, lineparams[*it6].cy1, lineparams[*it6].cy2);
                        }
                    }
                }
                it6 = it7;
            }
            section = block_section;

            // 段をソートしておく
            std::vector<int> section_renum(section_param.size());
            std::iota(section_renum.begin(), section_renum.end(), 0);
            std::sort(section_renum.begin()+section2, section_renum.end(), [&](const auto a, const auto b){
                // 上を先に
                return section_param[a].cy1 < section_param[b].cy1;
            });
            auto sit1 = section_renum.begin()+section2;
            while(sit1 != section_renum.end()) {
                float sy1 = section_param[*sit1].cy1;
                float sy2 = section_param[*sit1].cy2;
                // y方向に重なっている行を探索
                auto sit2 = std::partition(sit1, section_renum.end(), [&](const auto x){
                    return std::min(sy2, section_param[x].cy2) - std::max(sy1, section_param[x].cy1) > 0;
                });
                if (sit2 == sit1) {
                    ++sit1;
                    continue;
                }
                if(std::distance(sit1, sit2) > 1) {
                    std::sort(sit1, sit2, [&](const auto a, const auto b){
                        // 右を先に
                        return section_param[a].cx2 > section_param[b].cx2;
                    });
                }
                sit1 = sit2;
            }
            std::sort(section_param.begin()+section2, section_param.end(), [](const auto a, const auto b){
                // 上を先に
                return a.cy1 < b.cy1;
            });
            auto pit1 = section_param.begin()+section2;
            while(pit1 != section_param.end()) {
                float sy1 = pit1->cy1;
                float sy2 = pit1->cy2;
                // y方向に重なっている行を探索
                auto pit2 = std::partition(pit1, section_param.end(), [&](const auto x){
                    return std::min(sy2, x.cy2) - std::max(sy1, x.cy1) > 0;
                });
                if (pit2 == pit1) {
                    ++pit1;
                    continue;
                }
                if(std::distance(pit1, pit2) > 1) {
                    std::sort(pit1, pit2, [](const auto a, const auto b){
                        // 右を先に
                        return a.cx2 > b.cx2;
                    });
                }
                pit1 = pit2;
            }

            // 番号を振り直す
            for(auto &lp: lineparams) {
                auto sit = std::find(section_renum.begin(), section_renum.end(), lp.section);
                assert(sit != section_renum.end());
                lp.section = (int)std::distance(section_renum.begin(), sit);
            }

            // 表か段組かを判定する
            // 段組なら、縦書きなので右端が一致しているはず
            std::vector<float> sec_right(1+section_param.size());
            for(auto cit = it1; cit != it5; ++cit) {
                int s = lineparams[*cit].section;
                sec_right[s] = std::max(sec_right[s], lineparams[*cit].cx2);
            }
            std::vector<float> valid_sec_right;
            std::copy_if(sec_right.begin(), sec_right.end(), std::back_inserter(valid_sec_right), [](auto x) { return x > 0; });
            if(valid_sec_right.size() > 1) {
                float s = lineparams[*it1].size * 2;
                float r = std::reduce(valid_sec_right.begin(), valid_sec_right.end(), 0.0f, [](auto acc, auto x){ return std::max(acc, x); });
                int c = (int)std::count_if(valid_sec_right.begin(), valid_sec_right.end(), [r,s](auto x){ return fabs(r-x)<s*2; });
                
                if(c > 1) {
                    // 多分段組
                    std::sort(it1, it5, [&](const auto a, const auto b){
                        // 右を先に
                        return lineparams[a].cx2 > lineparams[b].cx2;
                    });
                    std::stable_sort(it1, it5, [&](const auto a, const auto b){
                        // 段の順にソートする
                        return lineparams[a].section < lineparams[b].section;
                    });
                    for(auto ait = it1; ait != it5; ++ait) {
                        lineparams[*ait].secidx = lineparams[*ait].section;
                    }
                }
                else {
                    // 多分表が混ざっている
                    std::sort(it1, it5, [&](const auto a, const auto b){
                        // 段の順にソートする
                        return lineparams[a].section < lineparams[b].section;
                    });
                    int secidx = lineparams[*it1].section;
                    auto it8 = it1;
                    while(it8 != it5) {
                        // 最初の段でまずソート
                        std::sort(it8, it5, [&](const auto a, const auto b){
                            // 段の順にソートする
                            return lineparams[a].section < lineparams[b].section;
                        });
                        auto it7 = std::partition(it8, it5, [&](const auto x){
                            return lineparams[*it8].section == lineparams[x].section;
                        });
                        std::sort(it8, it7, [&](const auto a, const auto b){
                            // 右を先に
                            return lineparams[a].cx2 > lineparams[b].cx2;
                        });
                        if(lineparams[*it8].section > secidx) {
                            for(; it8 != it7; ++it8) {
                                // xに重なっている、いっこ上の段があるはず
                                auto it9 = std::find_if(it1, it8, [&](const auto x) {
                                    return lineparams[*it8].section == lineparams[x].section + 1 && std::min(lineparams[*it8].cx2, lineparams[x].cx2) - std::max(lineparams[*it8].cx1, lineparams[x].cx1) > 0;
                                });
                                if (it9 == it8) {
                                    // ないので末尾に足す
                                }
                                else {
                                    // その後ろに追加する
                                    auto tmp = *it8;
                                    std::copy_backward(it9+1, it8, it8+1);
                                    *(it9+1) = tmp;
                                }
                            }
                        }
                        else {
                            it8 = it7;
                        }
                    }
                    for(auto ait = it1; ait != it5; ++ait) {
                        lineparams[*ait].secidx = secidx;
                    }
                }
            }
            else {
                // 普通に右からソート
                std::sort(it1, it5, [&](const auto a, const auto b){
                    // 右を先に
                    return lineparams[a].cx2 > lineparams[b].cx2;
                });
                for(auto ait = it1; ait != it5; ++ait) {
                    lineparams[*ait].secidx = lineparams[*ait].section;
                }
            }
            
            section = (int)section_param.size();
            section_param.emplace_back(width*scale,0,height*scale,0);
            it1 = it5;
        }
        
        // Section marge
        std::vector<int> section_sizeidx(section_param.size());
        std::iota(section_sizeidx.begin(), section_sizeidx.end(), 0);
        std::vector<int> section_idx(section_param.size());
        std::iota(section_idx.begin(), section_idx.end(), 0);
        // 大きい順に探す
        std::sort(section_sizeidx.begin(), section_sizeidx.end(), [&](const auto a, const auto b){
            return std::max(0.0f, section_param[a].cx2 - section_param[a].cx1) * std::max(0.0f, section_param[a].cy2 - section_param[a].cy1) > std::max(0.0f, section_param[b].cx2 - section_param[b].cx1) * std::max(0.0f, section_param[b].cy2 - section_param[b].cy1);
        });
        for(auto sidx: section_sizeidx) {
            if (std::max(0.0f, section_param[sidx].cx2 - section_param[sidx].cx1) * std::max(0.0f, section_param[sidx].cy2 - section_param[sidx].cy1) == 0) {
                break;
            }
            for(auto it = section_idx.begin(); it != section_idx.end();) {
                if (std::max(0.0f, section_param[*it].cx2 - section_param[*it].cx1) * std::max(0.0f, section_param[*it].cy2 - section_param[*it].cy1) == 0) {
                    ++it;
                    continue;
                }
                float area1 = (section_param[sidx].cx2 - section_param[sidx].cx1) * (section_param[sidx].cy2 - section_param[sidx].cy1);
                float area2 = (section_param[*it].cx2 - section_param[*it].cx1) * (section_param[*it].cy2 - section_param[*it].cy1);
                float inter_area = (std::min(section_param[sidx].cx2, section_param[*it].cx2) - std::max(section_param[sidx].cx1, section_param[*it].cx1)) * (std::min(section_param[sidx].cy2, section_param[*it].cy2) - std::max(section_param[sidx].cy1, section_param[*it].cy1));
                if(inter_area > std::min(area1, area2) * 0.25) {
                    // 重なっているのでまとめる
                    for(auto &lp: lineparams) {
                        if(lp.secidx == *it) {
                            lp.secidx = sidx;
                        }
                    }
                    it = section_idx.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        
        // Renumber secidx
        std::vector<int> section_renum;
        for(auto &lp: lineparams) {
            if(std::find(section_renum.begin(), section_renum.end(), lp.secidx) == section_renum.end()) {
                section_renum.push_back(lp.secidx);
            }
        }
        std::sort(section_renum.begin(), section_renum.end());
        // 番号を振り直す
        for(auto &lp: lineparams) {
            auto sit = std::find(section_renum.begin(), section_renum.end(), lp.secidx);
            assert(sit != section_renum.end());
            lp.secidx = (int)std::distance(section_renum.begin(), sit);
        }
    }
    
    for(auto &box: boxes) {
        if(box.idx < 0) continue;
        auto it = std::find(chain_remap.begin(), chain_remap.end(), box.idx);
        assert(it != chain_remap.end());
        int id = (int)std::distance(chain_remap.begin(), it);
        box.idx = id;
        box.section = lineparams[*it].secidx;
    }
    return int(chain_remap.size());
}

int renumber_chain(std::vector<charbox> &boxes);

// ブロックの形成
void make_block(
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker)
{
    int id_max;
    std::vector<std::vector<int>> chain_next;
    std::vector<std::vector<int>> chain_prev;
    std::vector<std::vector<int>> line_box_chain;
    while(true) {
        id_max = renumber_chain(boxes);
        id_max = renumber_id(id_max, boxes);

        chain_next.clear();
        chain_prev.clear();
        chain_next.resize(id_max);
        chain_prev.resize(id_max);
        line_box_chain.clear();
        line_box_chain.resize(id_max);

        std::vector<lineparam> lineparams(id_max);
        for(const auto &box: boxes) {
            if(box.idx < 0) continue;
            if((box.subtype & (2+4)) == 2+4) continue;
            if((box.subtype & 32) == 32) continue;

            line_box_chain[box.idx].push_back(box.id);
            lineparams[box.idx].size = std::max(lineparams[box.idx].size, std::max(box.w, box.h));
            lineparams[box.idx].count++;
            if((box.subtype & 1) == 0) {
                // 横書き
                if(line_box_chain[box.idx].size() > 1) {
                    lineparams[box.idx].d = 2;
                }
                else {
                    lineparams[box.idx].d = 0;
                }
            }
            else {
                // 縦書き
                if(line_box_chain[box.idx].size() > 1) {
                    lineparams[box.idx].d = 1;
                }
                else {
                    lineparams[box.idx].d = 0;
                }
            }
        }
        for(auto &chain: line_box_chain) {
            std::sort(chain.begin(), chain.end(), [&](const auto a, const auto b){
                return boxes[a].subidx < boxes[b].subidx;
            });
        }

        for(auto &chain: line_box_chain) {
            int count = 0;
            int count1 = 0;
            int count2 = 0;
            int chainid = -1;
            if (chain.size() == 0) continue;
            for(auto boxid: chain) {
                chainid = boxes[boxid].idx;
                if(chainid < 0) continue;
                if(boxes[boxid].double_line == 0) {
                    if(count1 > 1 || count2 > 1) {
                        count++;
                    }
                    count1 = count2 = 0;
                }
                if(boxes[boxid].double_line == 1) {
                    count1++;
                }
                if(boxes[boxid].double_line == 2) {
                    count2++;
                }
            }
            if(count1 > 1 || count2 > 1) {
                count++;
            }
            lineparams[chainid].doubleline = count;
        }
        
        std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, lineblocker, 1.0, 0);
        process_line(id_max, boxes, chain_next, chain_prev, chainid_map, line_box_chain, lineparams, lineblocker);

        line_box_chain.clear();
        line_box_chain.resize(id_max);
        for(const auto &box: boxes) {
            if(box.idx < 0) continue;

            line_box_chain[box.idx].push_back(box.id);
        }
        for(auto &chain: line_box_chain) {
            std::sort(chain.begin(), chain.end(), [&](const auto a, const auto b){
                return boxes[a].subidx < boxes[b].subidx;
            });
        }
        
        if(rechain_search(line_box_chain, boxes, chain_next, chain_prev)) {
            break;
        }
        
        // ないchainを消す
        for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
            if (it->empty()) {
                it = line_box_chain.erase(it);
            }
            else {
                ++it;
            }
        }

        // chain idを振り直す
        for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
            for(int bidx = 0; bidx < line_box_chain[chainid].size(); bidx++) {
                int boxid = line_box_chain[chainid][bidx];
                boxes[boxid].idx = chainid;
                boxes[boxid].subidx = bidx;
            }
        }
    }
    
    auto block_chain = block_chain_search(id_max, chain_next, chain_prev);

    std::vector<int> block_idx(block_chain.size());
    std::iota(block_idx.begin(), block_idx.end(), 0);
    struct blockparam {
        int d;
        int p;
        int sec;
        int count;
        float size;
        float x_min;
        float x_max;
        float y_min;
        float y_max;
    };
    std::vector<blockparam> blockparams(block_chain.size());
    for(auto &p: blockparams) {
        p.p = 0;
        p.count = 0;
        p.size = 0;
        p.sec = 0;
        p.x_min = width * scale;
        p.y_min = height * scale;
        p.x_max = 0;
        p.y_max = 0;
    }
    std::vector<int> blockid_of_chain(id_max, -1);
    for(int i = 0; i < block_chain.size(); i++) {
        for(const auto c: block_chain[i]) {
            blockid_of_chain[c] = i;
        }
    }

    for(const auto &box: boxes) {
        if(box.idx < 0) continue;
        if((box.subtype & 32) == 32) continue;

        int block = blockid_of_chain[box.idx];
        blockparams[block].d = ((box.subtype & 1) == 0) ? 0: 1;
        blockparams[block].count++;
        blockparams[block].size = std::max(blockparams[block].size, std::max(box.w, box.h));
        blockparams[block].sec = box.section;
        if(blockparams[block].x_min > box.cx - box.w/2)
            blockparams[block].x_min = box.cx - box.w/2;
        if(blockparams[block].y_min > box.cy - box.h/2)
            blockparams[block].y_min = box.cy - box.h/2;
        if(blockparams[block].x_max < box.cx + box.w/2)
            blockparams[block].x_max = box.cx + box.w/2;
        if(blockparams[block].y_max < box.cy + box.h/2)
            blockparams[block].y_max = box.cy + box.h/2;
    }
    for(auto &p: blockparams) {
        if(p.x_min > p.x_max) std::swap(p.x_min, p.x_max);
        if(p.y_min > p.y_max) std::swap(p.y_min, p.y_max);
    }

    // 縦書きか横書きか決める
    float orientation_score = 0;
    for(auto blockid: block_idx) {
        if(blockparams[blockid].d == 0) {
            // 横書きに投票
            orientation_score += (blockparams[blockid].x_max - blockparams[blockid].x_min) * (blockparams[blockid].y_max - blockparams[blockid].y_min);
        }
        else {
            // 縦書きに投票
            orientation_score -= (blockparams[blockid].x_max - blockparams[blockid].x_min) * (blockparams[blockid].y_max - blockparams[blockid].y_min);
        }
    }
    // 縦に分割できる見開きかどうか
    if (page_divide) {
        // 縦書きか横書きか、メインの方向のブロックのみを使用する
        std::vector<int> target_block_idx;
        std::copy_if(block_idx.begin(), block_idx.end(), std::back_inserter(target_block_idx), [&](const auto x){
            return (orientation_score >= 0) ? blockparams[x].d == 0 : blockparams[x].d == 1;
        });
        std::sort(target_block_idx.begin(), target_block_idx.end(), [&](const int a, const int b){
            return blockparams[a].x_min < blockparams[b].x_min;
        });

        float page_div_x = width * scale / 2;
        std::vector<std::pair<float,float>> div_points;
        for(auto it1 = target_block_idx.begin(); it1 != target_block_idx.end(); ++it1) {
            auto it2 = std::find_if(it1+1, target_block_idx.end(), [&](const auto x){
                return std::min(blockparams[x].x_max,blockparams[*it1].x_max) - std::max(blockparams[x].x_min,blockparams[*it1].x_min) <= 0;
            });
            if(it2 != target_block_idx.end()) {
                if(blockparams[*it2].x_min > blockparams[*it1].x_max) {
                    div_points.emplace_back(blockparams[*it1].x_max, blockparams[*it2].x_min);
                }
            }
        }
        
        if(div_points.empty()) {
            page_div_x = 0;
        }
        else {
            auto calc_dist = [&](std::pair<float,float> x){
                float d1 = page_div_x - x.first;
                float d2 = page_div_x - x.second;
                if(d1 * d2 > 0) {
                    return std::min(fabs(d1), fabs(d2));
                }
                else {
                    return 0.0f;
                }
            };
            std::sort(div_points.begin(), div_points.end(), [&](const auto a, const auto b){
                return calc_dist(a) < calc_dist(b);
            });
            
            if(div_points.front().first <= page_div_x && page_div_x <= div_points.front().second) {
                // keep page_div_x
            }
            else {
                page_div_x = (div_points.front().first + div_points.front().second) / 2;
            }
            
            // 真ん中から外れてるので、なかったことにする
            if(fabs(page_div_x - width * scale / 2) > width * scale / 10) {
                page_div_x = 0;
            }
        }

        if (orientation_score >= 0) {
            // 横書きなので、左から
            for(auto blockid: block_idx) {
                if(blockparams[blockid].x_min > page_div_x) {
                    blockparams[blockid].p = 1;
                }
                else {
                    blockparams[blockid].p = 0;
                }
            }
        }
        else {
            // 縦書きなので、右から
            for(auto blockid: block_idx) {
                if(blockparams[blockid].x_min > page_div_x) {
                    blockparams[blockid].p = 0;
                }
                else {
                    blockparams[blockid].p = 1;
                }
            }
        }
    }

    // 段順にソートしておく
    std::sort(block_idx.begin(), block_idx.end(), [&](const int a, const int b){
        return blockparams[a].sec < blockparams[b].sec;
    });
    // ページ順にソートしておく
    std::stable_sort(block_idx.begin(), block_idx.end(), [&](const int a, const int b){
        return blockparams[a].p < blockparams[b].p;
    });

    {
        auto st = block_idx.begin();
        while(st != block_idx.end()) {
            auto ed = std::partition(st, block_idx.end(), [&](const auto x){
                return blockparams[x].p == blockparams[*st].p && blockparams[x].sec == blockparams[*st].sec;
            });
            if (orientation_score < 0) {
                // 縦書き
                std::sort(st, ed, [&](const int a, const int b){
                    // 右から
                    return blockparams[a].x_max > blockparams[b].x_max;
                });

                auto it1 = st;
                while(it1 != ed) {
                    float x_min = blockparams[*it1].x_min;
                    float x_max = blockparams[*it1].x_max;
                    auto it2 = std::partition(it1, ed, [&](const auto x){
                        return std::min(x_max, blockparams[x].x_max) - std::max(x_min, blockparams[x].x_min) > 0;
                    });
                    std::sort(it1, it2, [&](const int a, const int b){
                        // 上から
                        return blockparams[a].y_min < blockparams[b].y_min;
                    });

                    it1 = it2;
                }
            }
            else {
                // 横書き
                std::sort(st, ed, [&](const int a, const int b){
                    // 上から
                    return blockparams[a].y_min < blockparams[b].y_min;
                });

                auto it1 = st;
                while(it1 != ed) {
                    float y_min = blockparams[*it1].y_min;
                    float y_max = blockparams[*it1].y_max;
                    auto it2 = std::partition(it1, ed, [&](const auto x){
                        return std::min(y_max, blockparams[x].y_max) - std::max(y_min, blockparams[x].y_min) > 0;
                    });
                    std::sort(it1, it2, [&](const int a, const int b){
                        // 左から
                        return blockparams[a].x_min < blockparams[b].x_min;
                    });

                    it1 = it2;
                }
            }
            st = ed;
        }
    }
    
    // idを振る
    {
        std::cerr << "id renumber " << block_idx.size() << std::endl;
        std::vector<int> chain_remap(id_max);
        std::fill(chain_remap.begin(), chain_remap.end(), -1);
        std::vector<int> chain_remap_page(id_max);
        int renum = 0;
        for(int i = 0; i < block_idx.size(); i++) {
            for(const auto chainid: block_chain[block_idx[i]]) {
                chain_remap[chainid] = renum;
                chain_remap_page[chainid] = blockparams[block_idx[i]].p;
            }
            renum++;
        }
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            box.block = chain_remap[box.idx];
            box.page = chain_remap_page[box.idx];
        }
    }

    // idxを修正
    {
        std::cerr << "fix idx" << std::endl;
        std::vector<std::vector<int>> idx_in_block(block_idx.size());
        for(const auto &box: boxes) {
            if(box.idx < 0) continue;
            if(box.block < 0) continue;
            idx_in_block[box.block].push_back(box.idx);
        }
        for(auto &list: idx_in_block) {
            if(list.size() < 2) continue;
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            if(box.block < 0) continue;
            auto it = std::find(idx_in_block[box.block].begin(), idx_in_block[box.block].end(), box.idx);
            box.idx = int(std::distance(idx_in_block[box.block].begin(), it));
        }
    }

    std::vector<charbox> result;
    for(auto box: boxes) {
        if(box.block < 0) continue;
        if(box.idx < 0) continue;
        result.push_back(box);
    }
    boxes = result;

    // 順にソートしておく
    {
        std::cerr << "sort idx" << std::endl;
        std::sort(boxes.begin(), boxes.end(), [](const auto a, const auto b){
            if(a.block == b.block) {
                if(a.idx == b.idx) {
                    if(a.subidx == b.subidx) {
                        return a.subtype < b.subtype;
                    }
                    return a.subidx < b.subidx;
                }
                return a.idx < b.idx;
            }
            return a.block < b.block;
        });
    }

    // 割注を処理する
    {
        std::cerr << "process split" << std::endl;
        int i = 0;
        std::vector<int> tmp;
        std::vector<std::vector<int>> double_idx;
        int block_idx = -1;
        int line_idx = -1;
        for(auto box: boxes) {
            if(block_idx != box.block || line_idx != box.idx) {
                if(tmp.size() > 2) {
                    double_idx.push_back(tmp);
                }
                tmp.clear();
            }
            block_idx = box.block;
            line_idx = box.idx;
            if(box.double_line > 0) {
                tmp.push_back(i);
            }
            else {
                if(tmp.size() > 2) {
                    double_idx.push_back(tmp);
                }
                tmp.clear();
            }
            i++;
        }
        if(tmp.size() > 2) {
            double_idx.push_back(tmp);
        }
        tmp.clear();
        for(auto idxlist: double_idx) {
            std::vector<int> sortidx(idxlist.size());
            std::iota(sortidx.begin(), sortidx.end(), 0);
            if((boxes[idxlist.front()].subtype & 1) == 0) {
                std::sort(sortidx.begin(), sortidx.end(), [idxlist, boxes](auto a, auto b){
                    if(a == b) return false;
                    if (boxes[idxlist[a]].double_line == boxes[idxlist[b]].double_line) {
                        return boxes[idxlist[a]].cx < boxes[idxlist[b]].cx;
                    }
                    return boxes[idxlist[a]].double_line < boxes[idxlist[b]].double_line;
                });
            }
            else {
                std::sort(sortidx.begin(), sortidx.end(), [idxlist, boxes](auto a, auto b){
                    if(a == b) return false;
                    if (boxes[idxlist[a]].double_line == boxes[idxlist[b]].double_line) {
                        return boxes[idxlist[a]].cy < boxes[idxlist[b]].cy;
                    }
                    return boxes[idxlist[a]].double_line < boxes[idxlist[b]].double_line;
                });
            }
            std::vector<int> subidx;
            for(auto id: idxlist) {
                subidx.push_back(boxes[id].subidx);
            }
            std::sort(subidx.begin(), subidx.end());
            for(int j = 0; j < sortidx.size(); j++) {
                boxes[idxlist[sortidx[j]]].subidx = subidx[j];
            }
        }
    }

    // 順にソートしておく
    {
        std::cerr << "sort idx" << std::endl;
        std::sort(boxes.begin(), boxes.end(), [](const auto a, const auto b){
            if(a.block == b.block) {
                if(a.idx == b.idx) {
                    if(a.subidx == b.subidx) {
                        return a.subtype < b.subtype;
                    }
                    return a.subidx < b.subidx;
                }
                return a.idx < b.idx;
            }
            return a.block < b.block;
        });
    }
}
