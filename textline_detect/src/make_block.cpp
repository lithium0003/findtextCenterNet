#include "make_block.h"
#include "search_loop.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <cmath>
#include <iostream>

struct lineparam {
    int d;
    int doubleline;
    int count;
    float size;
};

static const double allow_sizediff = 0.5;
static const double scanwidth_next_block = 1.0 + allowwidth_next_block;

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
    for(int chainid = 0; chainid < id_max; chainid++) {
        //std::cerr << "chain " << chainid << std::endl;
        if(line_box_chain[chainid].size() < 2) continue;

        if(chain_next[chainid].size() > 0) continue;

        std::vector<int> x;
        std::vector<int> y;
        float direction;
        double w, h;
        make_track_line(x,y, direction, w, h, boxes, line_box_chain, lineblocker, chainid, 1);
        if(x.size() == 0 || y.size() == 0) continue;
        double s_s = std::max(w, h);

        if(lineparams[chainid].d == 2) {
            // 横書き
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) continue;

                for(int yp = yi; yp < yi + s_s / scale * scanwidth_next_block; yp++) {
                    if(yp < 0 || yp >= height) continue;
                    //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * scale, yp * scale);
                    if(lineblocker[yp * width + xi]) break;
                    int other_chain = chainid_map[yp * width + xi];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 1) break;
                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / (0.5 * (lineparams[chainid].size + lineparams[other_chain].size)) > allow_sizediff) continue;

                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        //fprintf(stderr, "hori %d -> %d\n", chainid, other_chain);

                        float direction2;
                        double w2,h2;
                        float start_cx2, start_cy2, end_cx2, end_cy2;
                        search_chain(line_box_chain[other_chain], boxes, direction2, w2, h2, start_cx2, start_cy2, end_cx2, end_cy2);
                        float direction1;
                        double w1,h1;
                        float start_cx1, start_cy1, end_cx1, end_cy1;
                        search_chain(line_box_chain[chainid], boxes, direction1, w1, h1, start_cx1, start_cy1, end_cx1, end_cy1);

                        if((start_cy2 - std::max(h1,h2) - start_cy1) < std::max(h1,h2) * allowwidth_next_block) {
                            chain_next[chainid].push_back(other_chain);
                            chain_prev[other_chain].push_back(chainid);
                        }
                        else {
                            break;
                        }
                    }
                }
            }
        }
        else if (lineparams[chainid].d == 1){
            // 縦書き
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                if(lineblocker[yi * width + xi]) continue;

                for(int xp = xi; xp > xi - s_s / scale * scanwidth_next_block; xp--) {
                    if(xp < 0 || xp >= width) continue;
                    if(lineblocker[yi * width + xp]) break;
                    int other_chain = chainid_map[yi * width + xp];
                    if(other_chain < 0) continue;
                    if(other_chain == chainid) continue;
                    if(lineparams[other_chain].d == 2) break;

                    if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / (0.5 * (lineparams[chainid].size + lineparams[other_chain].size)) > allow_sizediff) continue;
                    
                    if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                        //fprintf(stderr, "vert %d -> %d\n", chainid, other_chain);

                        float direction2;
                        double w2,h2;
                        float start_cx2, start_cy2, end_cx2, end_cy2;
                        search_chain(line_box_chain[other_chain], boxes, direction2, w2, h2, start_cx2, start_cy2, end_cx2, end_cy2);
                        float direction1;
                        double w1,h1;
                        float start_cx1, start_cy1, end_cx1, end_cy1;
                        search_chain(line_box_chain[chainid], boxes, direction1, w1, h1, start_cx1, start_cy1, end_cx1, end_cy1);

                        if(((start_cx1 - std::max(w1,w2)/2) - (start_cx2 + std::max(w1,w2)/2)) < std::max(w1,w2) * allowwidth_next_block) {
                            chain_next[chainid].push_back(other_chain);
                            chain_prev[other_chain].push_back(chainid);
                        }
                        else {
                            break;
                        }
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
        std::vector<int> stack;
        for(const auto id: chain_prev[cur_id]) {
            stack.push_back(id);
        }
        std::vector<int> tmp_ids;
        while(stack.size() > 0) {
            auto j = stack.back();
            stack.pop_back();

            if(chain_prev[j].empty()) {
                    goto next_loop;
            }
            if(std::find(chain_root.begin(), chain_root.end(), j) != chain_root.end()) {
                goto next_loop;
            }

            if(std::find(tmp_ids.begin(), tmp_ids.end(), j) != tmp_ids.end()) continue;
            tmp_ids.push_back(j);
            for(const auto id: chain_prev[j]) {
                if(std::find(stack.begin(), stack.end(), id) == stack.end()) {
                    stack.push_back(id);
                }
            }
        }
        chain_root.push_back(cur_id);
    next_loop:
        ;
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

// 離れすぎている行をブロックから分離する
std::vector<std::vector<int>> split_block(
    int id_max,
    const std::vector<std::vector<int>> &block_chain,
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    std::vector<float> head_p(id_max);
    for(int chainid = 0; chainid < id_max; chainid++) {
        //std::cerr << "chain " << chainid << std::endl;
        if(line_box_chain[chainid].size() < 2) continue;

        int head_idx = line_box_chain[chainid].front();
        if ((boxes[head_idx].subtype & 1) == 0) {
            float cy = 0;
            int count = 0;
            for(auto idx: line_box_chain[chainid]) {
                if((boxes[idx].subtype & (2+4)) == 2+4) continue;
                cy += boxes[idx].cy;
                count++;
            }
            cy /= std::max(1,count);
            head_p[chainid] = cy;
        }
        else {
            float cx = 0;
            int count = 0;
            for(auto idx: line_box_chain[chainid]) {
                if((boxes[idx].subtype & (2+4)) == 2+4) continue;
                cx += boxes[idx].cx;
                count++;
            }
            cx /= std::max(1,count);
            head_p[chainid] = cx;
        }
    }

    std::vector<std::vector<int>> block_chain_result;
    for(auto block: block_chain) {
        if (block.size() < 4) {
            block_chain_result.push_back(block);
            continue;
        }
        // std::cerr << block_chain_result.size() << std::endl;

        std::vector<float> block_head;
        for(auto i: block) {
            block_head.push_back(head_p[i]);
        }

        std::vector<float> head_diff;
        for(int i = 0; i < block_head.size() - 1; i++) {
            head_diff.push_back(fabs(block_head[i+1] - block_head[i]));
        }

        float mean_diff = std::reduce(head_diff.begin(), head_diff.end()) / head_diff.size();
        std::vector<int> break_point;
        for(int i = 1; i < block.size(); i++) {
            // std::cerr << head_diff[i-1] << "," << mean_diff * 2.5 << "," << block_head[i-1] << std::endl;
            if(head_diff[i-1] > mean_diff * 2.5) {
                // std::cerr << "break" << std::endl;
                break_point.push_back(i);
            }
        }
        int start = 0;
        for(int i = 0; i < break_point.size(); i++) {
            std::vector<int> half(block.begin() + start, block.begin() + break_point[i]);
            start = break_point[i];
            block_chain_result.push_back(half);
        }
        std::vector<int> remain(block.begin() + start, block.end());
        block_chain_result.push_back(remain);
    }

    return block_chain_result;
}

// ブロックの形成
void make_block(
    int id_max,
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker)
{
    std::vector<std::vector<int>> chain_next(id_max, std::vector<int>());
    std::vector<std::vector<int>> chain_prev(id_max, std::vector<int>());

    std::vector<std::vector<int>> line_box_chain(id_max, std::vector<int>());

    std::vector<lineparam> lineparams(id_max);
    for(const auto &box: boxes) {
        if(box.idx < 0) continue;
        if((box.subtype & (2+4)) == 2+4) continue;
        if((box.subtype & 32) == 32) continue;

        line_box_chain[box.idx].push_back(box.id);
        lineparams[box.idx].size = std::max(lineparams[box.idx].size, std::max(box.w, box.h));
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
        sort_chain(chain, boxes);
    }
    
    std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, lineblocker, 1.5, 1);

    // process_doubleline(id_max, boxes, chain_next, chain_prev, chainid_map, line_box_chain, lineparams, lineblocker);
    // process_normalline(id_max, boxes, chain_next, chain_prev, chainid_map, line_box_chain, lineparams, lineblocker);

    process_line(id_max, boxes, chain_next, chain_prev, chainid_map, line_box_chain, lineparams, lineblocker);

    auto block_chain = block_chain_search(id_max, chain_next, chain_prev);

    block_chain = split_block(id_max, block_chain, boxes, line_box_chain);

    std::vector<int> block_idx(block_chain.size());
    std::iota(block_idx.begin(), block_idx.end(), 0);
    struct blockparam {
        int d;
        int p;
        int count;
        float size;
        float x_min;
        float x_max;
        float y_min;
        float y_max;
    };
    std::vector<blockparam> blockparams(block_chain.size());
    for(auto &p: blockparams) {
        p.p = -1;
        p.count = 0;
        p.size = 0;
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
        if(blockparams[block].x_min > box.cx)
            blockparams[block].x_min = box.cx;
        if(blockparams[block].y_min > box.cy)
            blockparams[block].y_min = box.cy;
        if(blockparams[block].x_max < box.cx)
            blockparams[block].x_max = box.cx;
        if(blockparams[block].y_max < box.cy)
            blockparams[block].y_max = box.cy;
    }
    for(auto &p: blockparams) {
        if(p.x_min > p.x_max) std::swap(p.x_min, p.x_max);
        if(p.y_min > p.y_max) std::swap(p.y_min, p.y_max);
    }

    std::vector<blockparam> pageparams;
    std::sort(block_idx.begin(), block_idx.end(), [blockparams](const int a, const int b){
        // 大きいブロックから検索する
        return (blockparams[a].x_max - blockparams[a].x_min) * (blockparams[a].y_max - blockparams[a].y_min) 
                > (blockparams[b].x_max - blockparams[b].x_min) * (blockparams[b].y_max - blockparams[b].y_min);
    });
    for(auto blockid: block_idx) {
        int pageidx = -1;
        for(int i = 0; i < pageparams.size(); i++) {
            if(blockparams[blockid].x_max > pageparams[i].x_min && pageparams[i].x_max > blockparams[blockid].x_min) {
                pageidx = i;
                break;
            }
        }
        if(pageidx < 0) {
            pageparams.push_back({ 
                .d = blockparams[blockid].d == 0 ? blockparams[blockid].count: -blockparams[blockid].count,
                .count = blockparams[blockid].count,
                .x_min = blockparams[blockid].x_min,
                .x_max = blockparams[blockid].x_max,
                .y_min = blockparams[blockid].y_min,
                .y_max = blockparams[blockid].y_max});
        }
        else {
            pageparams[pageidx].d += blockparams[blockid].d == 0 ? blockparams[blockid].count: -blockparams[blockid].count;
            pageparams[pageidx].count += blockparams[blockid].count;
            pageparams[pageidx].x_min = std::min(pageparams[pageidx].x_min, blockparams[blockid].x_min);
            pageparams[pageidx].x_max = std::max(pageparams[pageidx].x_max, blockparams[blockid].x_max);
            pageparams[pageidx].y_min = std::min(pageparams[pageidx].y_min, blockparams[blockid].y_min);
            pageparams[pageidx].y_max = std::max(pageparams[pageidx].y_max, blockparams[blockid].y_max);
        }
    }
    std::vector<int> page_idx(pageparams.size());
    std::iota(page_idx.begin(), page_idx.end(), 0);
    std::sort(page_idx.begin(), page_idx.end(), [pageparams](const int a, const int b){
        bool direction = pageparams[a].count > pageparams[b].count ? pageparams[a].d > 0 : pageparams[b].d > 0;
        if(direction) {
            // 横書き
            // x座標でみて、ブロック全体が左右どちらかにある
            if(pageparams[a].x_max < pageparams[b].x_min) return true;
            if(pageparams[b].x_max < pageparams[a].x_min) return false;

            // y座標でみて、ブロック全体が上下どちらかにある
            if(pageparams[a].y_max < pageparams[b].y_min) return true;
            if(pageparams[b].y_max < pageparams[a].y_min) return false;

            // ブロックが重なっているので、上にある方を優先
            if(pageparams[a].y_min == pageparams[b].y_min) return a < b;
            return pageparams[a].y_min < pageparams[b].y_min;
        }
        else {
            // 縦書き
            // x座標でみて、ブロック全体が左右どちらかにある 縦書きなので右から
            if(pageparams[a].x_min > pageparams[b].x_max) return true;
            if(pageparams[b].x_min > pageparams[a].x_max) return false;

            // y座標でみて、ブロック全体が上下どちらかにある
            if(pageparams[a].y_max < pageparams[b].y_min) return true;
            if(pageparams[b].y_max < pageparams[a].y_min) return false;

            // ブロックが重なっているので、右にある方を優先
            if(pageparams[a].x_max == pageparams[b].x_max) return a < b;
            return pageparams[a].x_max > pageparams[b].x_max;
        }
    });

    for(auto &block: blockparams) {
        for(int i = 0; i < page_idx.size(); i++) {
            if(pageparams[page_idx[i]].x_min <= block.x_min && pageparams[page_idx[i]].x_max >= block.x_max 
                && pageparams[page_idx[i]].y_min <= block.y_min && pageparams[page_idx[i]].y_max >= block.y_max) {
                    block.p = i;
                    break;
            }
        }
    }

    std::iota(block_idx.begin(), block_idx.end(), 0);
    std::sort(block_idx.begin(), block_idx.end(), [blockparams](const int a, const int b){
        if(blockparams[a].p != blockparams[b].p) {
            // ページ順
            return blockparams[a].p < blockparams[b].p;
        }

        int d;
        if(blockparams[a].d == blockparams[b].d) {
            d = blockparams[a].d;
        }
        else {
            // 片方が短い
            d = blockparams[a].count > blockparams[b].count ? blockparams[a].d : blockparams[b].d;
        }
        if(d != 1) {
            // 横書き
            // y座標でみて、ブロック全体が上下どちらかにある
            if(blockparams[a].y_max < blockparams[b].y_min) return true;
            if(blockparams[b].y_max < blockparams[a].y_min) return false;

            // x座標でみて、ブロック全体が左右どちらかにある
            if(blockparams[a].x_max < blockparams[b].x_min) return true;
            if(blockparams[b].x_max < blockparams[a].x_min) return false;

            // ブロックが重なっているので、上にある方を優先
            if(blockparams[a].y_min == blockparams[b].y_min) return a < b;
            return blockparams[a].y_min < blockparams[b].y_min;
        }
        else {
            // 縦書き
            // y座標でみて、ブロック全体が上下どちらかにある
            if(blockparams[a].y_max < blockparams[b].y_min) return true;
            if(blockparams[b].y_max < blockparams[a].y_min) return false;

            // x座標でみて、ブロック全体が左右どちらかにある 縦書きなので右から
            if(blockparams[a].x_min > blockparams[b].x_max) return true;
            if(blockparams[b].x_min > blockparams[a].x_max) return false;

            // ブロックが重なっているので、右にある方を優先
            if(blockparams[a].x_max == blockparams[b].x_max) return a < b;
            return blockparams[a].x_max > blockparams[b].x_max;
        }
    });

    float main_block_size = 0;
    int max_block_count = 0;
    for(auto p: blockparams) {
        if(p.count > max_block_count) {
            main_block_size = p.size;
            max_block_count = p.count;
        }
    }
    std::cerr << "block size " << main_block_size << std::endl;

    // idを振る
    {
        std::cerr << "id renumber " << block_idx.size() << std::endl;
        std::vector<int> chain_remap(id_max);
        std::fill(chain_remap.begin(), chain_remap.end(), -1);
        int renum = 0;
        for(int i = 0; i < block_idx.size(); i++) {
            // if(ignore_small_size_block) {
            //     if(blockparams[block_idx[i]].size < main_block_size * ignore_small_size_block_ratio) {
            //         continue;
            //     }
            // }
            for(const auto chainid: block_chain[block_idx[i]]) {
                chain_remap[chainid] = renum;
            }
            renum++;
        }
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            box.block = chain_remap[box.idx];
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
            box.idx = std::distance(idx_in_block[box.block].begin(), it);
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
                    if (boxes[idxlist[a]].double_line == boxes[idxlist[b]].double_line) {
                        return boxes[idxlist[a]].cx < boxes[idxlist[b]].cx;
                    }
                    return boxes[idxlist[a]].double_line < boxes[idxlist[b]].double_line;
                });
            }
            else {
                std::sort(sortidx.begin(), sortidx.end(), [idxlist, boxes](auto a, auto b){
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
