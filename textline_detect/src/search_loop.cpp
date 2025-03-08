#include "search_loop.h"
#include "split_doubleline.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <iostream>
#include <cmath>

void sort_chain(
    std::vector<int> &chain,
    const std::vector<charbox> &boxes) 
{
    // chain内をソート
    if(fabs(boxes[chain.front()].direction) < M_PI_4) {
        // 横書き
        std::sort(chain.begin(), chain.end(), [&](auto a, auto b){
            return boxes[a].cx < boxes[b].cx;
        });
        auto it = chain.begin();
        while(it != chain.end()) {
            auto it2 = it+1;
            while(it2 != chain.end()) {
                //x方向に重なっている場合は、y方向にソートする
                if(boxes[*it].cx + boxes[*it].w/2 > boxes[*it2].cx) {
                    ++it2;
                    continue;
                }
                break;
            }
            if(std::distance(it, it2) > 1) {
                std::sort(it, it2, [&](auto a, auto b){
                    return boxes[a].cy < boxes[b].cy;
                });
            }
            it = it2;
        }
    }
    else {
        // 縦書き
        std::sort(chain.begin(), chain.end(), [&](auto a, auto b){
            return boxes[a].cy < boxes[b].cy;
        });
        auto it = chain.begin();
        while(it != chain.end()) {
            auto it2 = it+1;
            while(it2 != chain.end()) {
                //y方向に重なっている場合は、x方向にソートする
                if(boxes[*it].cy + boxes[*it].h/2 > boxes[*it2].cy) {
                    ++it2;
                    continue;
                }
                break;
            }
            if(std::distance(it, it2) > 1) {
                std::sort(it, it2, [&](auto a, auto b){
                    return boxes[a].cx < boxes[b].cx;
                });
            }
            it = it2;
        }
    }
}

void fix_chain_info(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain)
{
    // ないchainを消す
    for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        if (it->size() < 2) {
            it = line_box_chain.erase(it);
        }
        else {
            ++it;
        }
    }

    // chain内をソート
    for(auto &chain: line_box_chain) {
        sort_chain(chain, boxes);
        float direction, start_cx, start_cy, end_cx, end_cy;
        double w, h;
        search_chain(chain, boxes, direction, w, h, start_cx, start_cy, end_cx, end_cy);
        for(auto boxid: chain) {
            boxes[boxid].direction = direction;
        }
    }
}

void search_chain(
    const std::vector<int> &chain,
    const std::vector<charbox> &boxes,
    float &direction,
    double &w, double &h,
    float &start_cx, float &start_cy, 
    float &end_cx, float &end_cy)
{
    if(chain.empty()) {
        direction = 0;
        w = 0;
        h = 0;
        start_cx = -1;
        start_cy = -1;
        end_cx = -1;
        end_cy = -1;
        return;
    }
    std::vector<int> nonruby;
    w = 0;
    h = 0;
    for(int i = 0; i < chain.size(); i++) {
        if((boxes[chain[i]].subtype & (2+4)) == 2+4) continue;
        if(boxes[chain[i]].double_line > 0) continue;
        nonruby.push_back(chain[i]);
        w = std::max(w, double(boxes[chain[i]].w));
        h = std::max(h, double(boxes[chain[i]].h));
    }

    if(nonruby.empty()) {
        direction = 0;
        w = 0;
        h = 0;
        start_cx = -1;
        start_cy = -1;
        end_cx = -1;
        end_cy = -1;
        return;
    }
    direction = boxes[nonruby.front()].direction;

    if(fabs(direction) < M_PI_4) {
        // 横書き
        start_cy = 0;
        double sum = 0;
        for(int i = 0; i < nonruby.size(); i++) {
            double weight = boxes[nonruby[i]].w * boxes[nonruby[i]].h / (i + 1);
            start_cy += boxes[nonruby[i]].cy * weight;
            sum += weight;
        }
        start_cy /= sum;
        start_cx = boxes[nonruby.front()].cx;

        end_cy = 0;
        sum = 0;
        for(int i = 0; i < nonruby.size(); i++) {
            double weight = boxes[nonruby[i]].w * boxes[nonruby[i]].h / (nonruby.size() - i);
            end_cy += boxes[nonruby[i]].cy * weight;
            sum += weight;
        }
        end_cy /= sum;
        end_cx = boxes[nonruby.back()].cx;
        direction = atan2(end_cy - start_cy, end_cx - start_cx);
        if(direction > M_PI_2) {
            direction -= M_PI;
        }
    }
    else {
        // 縦書き
        start_cx = 0;
        double sum = 0;
        for(int i = 0; i < nonruby.size(); i++) {
            double weight = boxes[nonruby[i]].w * boxes[nonruby[i]].h / (i + 1);
            start_cx += boxes[nonruby[i]].cx * weight;
            sum += weight;
        }
        start_cx /= sum;
        start_cy = boxes[nonruby.front()].cy;

        end_cx = 0;
        sum = 0;
        for(int i = 0; i < nonruby.size(); i++) {
            double weight = boxes[nonruby[i]].w * boxes[nonruby[i]].h / (nonruby.size() - i);
            end_cx += boxes[nonruby[i]].cx * weight;
            sum += weight;
        }
        end_cx /= sum;
        end_cy = boxes[nonruby.back()].cy;
        direction = atan2(end_cy - start_cy, end_cx - start_cx);
    }
}

// 中心をトラックするboxidを軸に沿って返す
void find_linecenter_id(
    std::vector<int> &line_select_id,
    float &direction,
    double &w, double &h,
    float &start_cx, float &start_cy, float &end_cx, float &end_cy,
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    int chainid)
{
    //std::cerr << "find_linecenter_id" << std::endl;

    line_select_id.clear();

    search_chain(line_box_chain[chainid], boxes, direction, w, h, start_cx, start_cy, end_cx, end_cy);
    //fprintf(stderr, "%d direction %f start cx %f cy %f end cx %f cy %f w %f h %f\n", chainid, direction, start_cx, start_cy, end_cx, end_cy, w, h);

    float sum_x = 0;
    float sum_y = 0;
    int count = 0;
    for(const auto &boxid: line_box_chain[chainid]) {
        if((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
        if(boxes[boxid].double_line > 0) continue;
        if(std::max(boxes[boxid].w, boxes[boxid].h) / std::max(w, h) < 0.4) continue;
        sum_x += boxes[boxid].cx;
        sum_y += boxes[boxid].cy;
        count++;
    }
    if(count == 0) return;
    float center_x = sum_x / count;
    float center_y = sum_y / count;
    if(fabs(direction) < M_PI_4) {
        // 横書き
        for(const auto &boxid: line_box_chain[chainid]) {
            if((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
            if(boxes[boxid].double_line > 0) continue;
            if(std::max(boxes[boxid].w, boxes[boxid].h) / std::max(w, h) < 0.4) continue;
            float xi = boxes[boxid].cx - center_x;
            float yi = tan(direction) * xi + center_y;
            if(fabs(yi - boxes[boxid].cy) < float(std::max(w,h) / 2)) {
                line_select_id.push_back(boxid);
            }
        }
    }
    else {
        // 縦書き        
        for(const auto &boxid: line_box_chain[chainid]) {
            if((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
            if(boxes[boxid].double_line > 0) continue;
            if(std::max(boxes[boxid].w, boxes[boxid].h) / std::max(w, h) < 0.4) continue;
            float yi = boxes[boxid].cy - center_y;
            float xi = tan(M_PI_2 - direction) * yi + center_x;
            if(fabs(xi - boxes[boxid].cx) < float(std::max(w,h) / 2)) {
                line_select_id.push_back(boxid);
            }
        }
    }
}

// 中心位置のトラックを行う
void make_track_line(
    std::vector<int> &x,
    std::vector<int> &y,
    float &direction,
    double &w, double &h,
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    int chainid,
    int extra_len)
{
    //std::cerr << "make_track_line" << std::endl;

    x.clear();
    y.clear();

    float start_cx, start_cy, end_cx, end_cy;
    std::vector<int> line_select_id;
    find_linecenter_id(line_select_id, direction, w, h, start_cx, start_cy, end_cx, end_cy, boxes, line_box_chain, chainid);

    if(fabs(direction) < M_PI_4) {
        // 横書き
        std::vector<float> xi;
        std::vector<float> yi;
        float track_cy = -1;
        {
            if(line_select_id.size() > 0) {
                int j = line_select_id.front();
                track_cy = boxes[j].cy;
                xi.push_back(boxes[j].cx - boxes[j].w / 2);
                yi.push_back(track_cy);
            }
            else {
                xi.push_back(start_cx);
                yi.push_back(start_cy);
                xi.push_back(end_cx);
                yi.push_back(end_cy);
                return;
            }
        }

        for(const auto boxid: line_box_chain[chainid]) {
            if ((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
            if(std::find(line_select_id.begin(), line_select_id.end(), boxid) != line_select_id.end()) {
                track_cy = (track_cy + boxes[boxid].cy) / 2;
            }
            xi.push_back(boxes[boxid].cx);
            yi.push_back(track_cy);
        }

        {
            int j = line_select_id.back();
            track_cy = boxes[j].cy;
            xi.push_back(boxes[j].cx + boxes[j].w / 2);
            yi.push_back(track_cy);
        }

        if(extra_len > 0 && xi.size() >= 2) {
            double x1 = xi[0];
            double y1 = yi[0];
            double x2 = xi[1];
            double y2 = yi[1];
            double a = (y2 - y1) / (x2 - x1);
            for(int xp = x1; xp > x1 - extra_len * w * 2; xp -= scale) {
                int yp = (xp - x1) * a + y1;
                int xp1 = xp / scale;
                int yp1 = yp / scale;
                if (xp1 < 0 || xp1 >= width || yp1 < 0 || yp1 >= height) {
                    continue;
                }
                if (lineblocker[yp1 * width + xp1]) {
                    break;
                }
                x.push_back(xp);
                y.push_back(yp);
            }
            std::reverse(x.begin(), x.end());
            std::reverse(y.begin(), y.end());
        }

        for(int i = 0; i < xi.size() - 1; i++) {
            double x1 = xi[i];
            double y1 = yi[i];
            double x2 = xi[i+1];
            double y2 = yi[i+1];
            double a = (y2 - y1) / (x2 - x1);
            if(!isfinite(a)) continue;
            for(int xp = x1; xp < x2; xp += scale) {
                int yp = (xp - x1) * a + y1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }

        if(extra_len > 0 && xi.size() >= 2) {
            double x1 = xi[xi.size() - 2];
            double y1 = yi[xi.size() - 2];
            double x2 = xi[xi.size() - 1];
            double y2 = yi[xi.size() - 1];
            double a = (y2 - y1) / (x2 - x1);
            for(int xp = x2; xp < x2 + extra_len * w * 2; xp += scale) {
                int yp = (xp - x1) * a + y1;
                int xp1 = xp / scale;
                int yp1 = yp / scale;
                if (xp1 < 0 || xp1 >= width || yp1 < 0 || yp1 >= height) {
                    continue;
                }
                if (lineblocker[yp1 * width + xp1]) {
                    break;
                }
                x.push_back(xp);
                y.push_back(yp);
            }
        }
    }
    else {
        // 縦書き
        std::vector<float> xi;
        std::vector<float> yi;
        float track_cx = -1;
        {
            if(line_select_id.size() > 0) {
                int j = line_select_id.front();
                track_cx = boxes[j].cx;
                xi.push_back(track_cx);
                yi.push_back(boxes[j].cy - boxes[j].h / 2);
            }
            else {
                xi.push_back(start_cx);
                yi.push_back(start_cy);
                xi.push_back(end_cx);
                yi.push_back(end_cy);
                return;
            }
        }

        for(const auto boxid: line_box_chain[chainid]) {
            if ((boxes[boxid].subtype & (2+4)) == (2+4)) continue;
            if(std::find(line_select_id.begin(), line_select_id.end(), boxid) != line_select_id.end()) {
                track_cx = (track_cx + boxes[boxid].cx) / 2;
            }
            xi.push_back(track_cx);
            yi.push_back(boxes[boxid].cy);
        }

        {
            int j = line_select_id.back();
            track_cx = boxes[j].cx;
            xi.push_back(track_cx);
            yi.push_back(boxes[j].cy + boxes[j].h / 2);
        }

        if(extra_len > 0 && xi.size() >= 2) {
            double x1 = xi[0];
            double y1 = yi[0];
            double x2 = xi[1];
            double y2 = yi[1];
            double a = (x2 - x1) / (y2 - y1);
            for(int yp = y1; yp > y1 - extra_len * h * 2; yp -= scale) {
                int xp = (yp - y1) * a + x1;
                int xp1 = xp / scale;
                int yp1 = yp / scale;
                if (xp1 < 0 || xp1 >= width || yp1 < 0 || yp1 >= height) {
                    continue;
                }
                if (lineblocker[yp1 * width + xp1]) {
                    break;
                }
                x.push_back(xp);
                y.push_back(yp);
            }
            std::reverse(x.begin(), x.end());
            std::reverse(y.begin(), y.end());
        }

        for(int i = 0; i < xi.size() - 1; i++) {
            double x1 = xi[i];
            double y1 = yi[i];
            double x2 = xi[i+1];
            double y2 = yi[i+1];
            double a = (x2 - x1) / (y2 - y1);
            if(!isfinite(a)) continue;
            for(int yp = y1; yp < y2; yp += scale) {
                int xp = (yp - y1) * a + x1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }

        if(extra_len > 0 && xi.size() >= 2) {
            double x1 = xi[xi.size() - 2];
            double y1 = yi[xi.size() - 2];
            double x2 = xi[xi.size() - 1];
            double y2 = yi[xi.size() - 1];
            double a = (x2 - x1) / (y2 - y1);
            for(int yp = y2; yp < y2 + extra_len * h * 2; yp += scale) {
                int xp = (yp - y1) * a + x1;
                int xp1 = xp / scale;
                int yp1 = yp / scale;
                if (xp1 < 0 || xp1 >= width || yp1 < 0 || yp1 >= height) {
                    continue;
                }
                if (lineblocker[yp1 * width + xp1]) {
                    break;
                }
                x.push_back(xp);
                y.push_back(yp);
            }
        }
    }
}

std::vector<int> create_chainid_map(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    double ratio,
    int extra_len)
{
    //std::cerr << "create_chainid_map" << std::endl;
    std::vector<int> chainid_map(width*height, -1);

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() == 0) continue;

        std::vector<int> x;
        std::vector<int> y;
        float direction;
        double w, h;
        make_track_line(x,y, direction, w, h, boxes, line_box_chain, lineblocker, chainid, extra_len);
        double s_s = std::max(w, h);

        if(fabs(direction) < M_PI_4) {
            // 横書き
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int yp = yi - s_s / 3 / scale * ratio; yp < yi + s_s / 3 / scale * ratio; yp++) {
                    if(yp < 0 || yp >= height) continue;
                    if(chainid_map[yp * width + xi] < 0) {
                        chainid_map[yp * width + xi] = chainid;
                    }
                }
            }
        }
        else {
            // 縦書き
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int xp = xi - s_s / 3 / scale * ratio; xp < xi + s_s / 3 / scale * ratio; xp++) {
                    if(xp < 0 || xp >= width) continue;
                    if(chainid_map[yi * width + xp] < 0) {
                        chainid_map[yi * width + xp] = chainid;
                    }
                }
            }
        }
    }

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        for(auto boxid: line_box_chain[chainid]) {
            auto box = boxes[boxid];
            if ((box.subtype & (2+4)) == 2+4) continue;
            //fprintf(stderr, "chain %d box %d cx %f cy %f w %f h %f th %f\n", chainid, box.id, box.cx, box.cy, box.w, box.h, box.direction / M_PI * 180);
            int left = (box.cx - box.w / 3) / scale;
            int right = (box.cx + box.w / 3) / scale;
            int top = (box.cy - box.h / 3) / scale;
            int bottom = (box.cy + box.h / 3) / scale;
            for(int y = top; y < bottom; y++) {
                for(int x = left; x < right; x++) {
                    if(x < 0 || x >= width || y < 0 || y >= height) continue;
                    if(chainid_map[y * width + x] < 0) {
                        chainid_map[y * width + x] = chainid;
                    }
                }
            }
        }
    }

    return chainid_map;
}

void process_merge(
    std::vector<std::vector<int>> &line_box_chain,
    std::vector<int> &merge_chain)
{
    // chainの連結の処理
    std::vector<int> root_id(line_box_chain.size(), -1);
    int new_id = int(line_box_chain.size());
    for(int i = 0; i < merge_chain.size(); i++) {
        if(merge_chain[i] == -1) continue;
        
        std::vector<int> loop_check;
        int j = merge_chain[i];
        loop_check.push_back(j);
        while(std::find(loop_check.begin(), loop_check.end(), merge_chain[j]) == loop_check.end() && merge_chain[j] != -1) {
            j = merge_chain[j];
            loop_check.push_back(j);
        }
        if(merge_chain[j] != -1) {
            int k = -1;
            for(const auto c: loop_check) {
                if(root_id[c] < 0) continue;
                k = root_id[c];
                break;
            }
            if(k < 0) {
                line_box_chain.push_back(std::vector<int>());
                root_id.push_back(-1);
                root_id[i] = new_id;
                new_id = int(line_box_chain.size());
            }
            else {
                root_id[i] = k;
            }
        }
        else {
            root_id[i] = j;
        }
    }
    for(int i = 0; i < line_box_chain.size(); i++) {
        if(root_id[i] < 0) continue;
        std::copy(line_box_chain[i].begin(), line_box_chain[i].end(), std::back_inserter(line_box_chain[root_id[i]]));
        line_box_chain[i].clear();
    }

    // 重複を排除
    for(int i = 0; i < line_box_chain.size(); i++) {
        if(line_box_chain[i].size() < 2) continue;
        std::sort(line_box_chain[i].begin(), line_box_chain[i].end());
        line_box_chain[i].erase( std::unique(line_box_chain[i].begin(), line_box_chain[i].end()), line_box_chain[i].end());
    }
}

void combine_chains(
    const std::vector<bool> &lineblocker,
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain)
{
    fprintf(stderr, "combine_chains\n");
    while(true) {
        fprintf(stderr, "process\n");
        fix_chain_info(boxes, line_box_chain);

        // 暫定chain id のマップを作る
        std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, lineblocker);

        // 分割して認識されたchainを連結する。
        std::vector<int> merge_chain(line_box_chain.size(), -1);
        for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
            auto boxid1 = line_box_chain[chainid].front();
            auto boxid2 = line_box_chain[chainid].back();
            double w = std::transform_reduce(
                line_box_chain[chainid].begin(), 
                line_box_chain[chainid].end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].w; });
            double h = std::transform_reduce(
                line_box_chain[chainid].begin(), 
                line_box_chain[chainid].end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].h; });

            if(fabs(boxes[boxid1].direction) < M_PI_4) {
                // 横書き
                double start_cy0 = boxes[line_box_chain[chainid].back()].cy;
                for(int i = int(line_box_chain[chainid].size()) - 1; i >= 0; i--) {
                    int boxid = line_box_chain[chainid][i];
                    start_cy0 = 0.25 * start_cy0 + 0.75 * boxes[boxid].cy;
                }
                
                double end_cy0 = boxes[line_box_chain[chainid].front()].cy;
                for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                    int boxid = line_box_chain[chainid][i];
                    end_cy0 = 0.25 * end_cy0 + 0.75 * boxes[boxid].cy;
                }

                double space = std::transform_reduce(
                    line_box_chain[chainid].begin(), 
                    std::prev(line_box_chain[chainid].end()),
                    std::next(line_box_chain[chainid].begin()),
                    0.0,
                    [&](double a, double b) { return a + b; },
                    [&](int a, int b) {
                        if(boxes[b].cx - boxes[b].w / 2 > boxes[a].cx + boxes[a].w / 2)
                            return (boxes[b].cx - boxes[b].w / 2) - (boxes[a].cx + boxes[a].w / 2);
                        else
                            return 0.0f;
                    }) / (line_box_chain[chainid].size() - 1);

                if(space < 0)
                    space = 0;

                std::vector<int> other_chainid1;
                std::vector<int> done_chain;
                // 左側に検索
                for(int xs = 0; xs < (w + space) * 3.0; xs+=scale) {
                    int x = boxes[boxid1].cx - w / 2 - xs;
                    x /= scale;
                    if(x < 0 || x >= width) continue;
                    for(int yk = -5; yk <= 5; yk++) {
                        int y = start_cy0 + h / 8.0 * yk / 5;
                        y /= scale;
                        if(y < 0 || y >= height) continue;
                        if(lineblocker[y * width + x]) {
                            other_chainid1.clear();
                            goto find1;
                        }
                        int other_chainid = chainid_map[y * width + x];
                        if(other_chainid >= 0 && other_chainid != chainid && 
                                std::find(done_chain.begin(), done_chain.end(), other_chainid) == done_chain.end() &&
                                std::find(other_chainid1.begin(), other_chainid1.end(), other_chainid) == other_chainid1.end()) {

                            if(fabs(boxes[line_box_chain[other_chainid].back()].direction) < M_PI_4 && 
                                boxes[line_box_chain[other_chainid].back()].double_line == boxes[boxid1].double_line) {

                                double h1 = std::transform_reduce(
                                    line_box_chain[other_chainid].begin(), 
                                    line_box_chain[other_chainid].end(), 
                                    0.0, 
                                    [&](double acc, double i) { return std::max(acc, i); },
                                    [&](int x) { return boxes[x].h; });

                                double end_cy1 = boxes[line_box_chain[other_chainid].front()].cy;
                                for(int i = 0; i < line_box_chain[other_chainid].size(); i++) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    end_cy1 = 0.25 * end_cy0 + 0.75 * boxes[boxid].cy;
                                }
                                // fprintf(stderr, "chain1 %d -> %d test %f %f %f\n", other_chainid, chainid, start_cy0, end_cy1, std::max(h, h1));

                                if(fabs(start_cy0 - end_cy1) < std::max(h, h1)) {
                                    other_chainid1.push_back(other_chainid);
                                    continue;
                                }
                            }
                            done_chain.push_back(other_chainid);
                        } // if other_chainid
                    } // for yk
                    if(done_chain.size() > 0) {
                        break;
                    }
                } // for xs
                find1:
                if(other_chainid1.size() == 1) {
                    //fprintf(stderr, "chain1 %d -> %d\n", other_chainid1, chainid);
                    merge_chain[chainid] = other_chainid1[0];
                }

                // 末尾が連結可能か検索
                std::vector<int> other_chainid2;
                // 右側に検索
                for(int xs = 0; xs < (w + space) * 3.0; xs+=scale) {
                    int x = boxes[boxid2].cx + w / 2 + xs;
                    x /= scale;
                    if(x < 0 || x >= width) continue;
                    for(int yk = -5; yk <= 5; yk++) {
                        int y = end_cy0 + h / 8.0 * yk / 5;
                        y /= scale;
                        if(y < 0 || y >= height) continue;
                        if(lineblocker[y * width + x]) {
                            other_chainid2.clear();
                            goto find2;
                        }
                        int other_chainid = chainid_map[y * width + x];
                        if(other_chainid >= 0 && other_chainid != chainid && 
                                std::find(done_chain.begin(), done_chain.end(), other_chainid) == done_chain.end() &&
                                std::find(other_chainid2.begin(), other_chainid2.end(), other_chainid) == other_chainid2.end()) {

                            if(fabs(boxes[line_box_chain[other_chainid].back()].direction) < M_PI_4 &&
                                boxes[line_box_chain[other_chainid].front()].double_line == boxes[boxid2].double_line) {

                                double h1 = std::transform_reduce(
                                    line_box_chain[other_chainid].begin(), 
                                    line_box_chain[other_chainid].end(), 
                                    0.0, 
                                    [&](double acc, double i) { return std::max(acc, i); },
                                    [&](int x) { return boxes[x].h; });

                                double start_cy1 = boxes[line_box_chain[other_chainid].front()].cy;
                                for(int i = int(line_box_chain[other_chainid].size()) - 1; i >= 0; i--) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    start_cy1 = 0.25 * start_cy1 + 0.75 * boxes[boxid].cy;
                                }

                                //fprintf(stderr, "chain2 %d -> %d test %f %f %f\n", chainid, other_chainid2, end_cy0, start_cy1, std::max(h, h1));

                                if(fabs(end_cy0 - start_cy1) < std::max(h, h1)) {
                                    other_chainid2.push_back(other_chainid);
                                }
                            }
                            done_chain.push_back(other_chainid);
                        } // if other_chainid
                    } // for yk
                    if(done_chain.size() > 0) {
                        break;
                    }
                } // for xs
                find2:
                if(other_chainid2.size() == 1) {
                    //fprintf(stderr, "chain2 %d -> %d\n", chainid, other_chainid2);
                    merge_chain[other_chainid2[0]] = chainid;
                }
            }
            else {
                // 縦書き
                double start_cx0 = boxes[line_box_chain[chainid].back()].cx;
                for(int i = int(line_box_chain[chainid].size()) - 1; i >= 0; i--) {
                    int boxid = line_box_chain[chainid][i];
                    start_cx0 = 0.25 * start_cx0 + 0.75 * boxes[boxid].cx;
                }
                
                double end_cx0 = boxes[line_box_chain[chainid].front()].cx;
                for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                    int boxid = line_box_chain[chainid][i];
                    end_cx0 = 0.25 * end_cx0 + 0.75 * boxes[boxid].cx;
                }

                double space = std::transform_reduce(
                    line_box_chain[chainid].begin(), 
                    std::prev(line_box_chain[chainid].end()),
                    std::next(line_box_chain[chainid].begin()),
                    0.0,
                    [&](double a, double b) { return a + b; },
                    [&](int a, int b) {
                        if(boxes[b].cy - boxes[b].h / 2 > boxes[a].cy + boxes[a].h / 2)
                            return (boxes[b].cy - boxes[b].h / 2) - (boxes[a].cy + boxes[a].h / 2);
                        else
                            return 0.0f;
                    }) / (line_box_chain[chainid].size() - 1);

                if(space < 0)
                    space = 0;

                std::vector<int> other_chainid1;
                std::vector<int> done_chain;
                // 上側に検索
                for(int ys = 0; ys < (h + space) * 3.0; ys+=scale) {
                    int y = boxes[boxid1].cy - h / 2 - ys;
                    y /= scale;
                    if(y < 0 || y >= height) continue;
                    for(int xk = -5; xk <= 5; xk++) {
                        int x = start_cx0 + w / 8.0 * xk / 5;
                        x /= scale;
                        if(x < 0 || x >= width) continue;
                        if(lineblocker[y * width + x]) {
                            other_chainid1.clear();
                            goto find3;
                        }
                        int other_chainid = chainid_map[y * width + x];
                        if(other_chainid >= 0 && other_chainid != chainid && 
                                std::find(done_chain.begin(), done_chain.end(), other_chainid) == done_chain.end() &&
                                std::find(other_chainid1.begin(), other_chainid1.end(), other_chainid) == other_chainid1.end()) {

                            if(fabs(boxes[line_box_chain[other_chainid].back()].direction) > M_PI_4 &&
                                boxes[line_box_chain[other_chainid].back()].double_line == boxes[boxid1].double_line) {

                                double w1 = std::transform_reduce(
                                    line_box_chain[other_chainid].begin(), 
                                    line_box_chain[other_chainid].end(), 
                                    0.0, 
                                    [&](double acc, double i) { return std::max(acc, i); },
                                    [&](int x) { return boxes[x].w; });

                                double end_cx1 = boxes[line_box_chain[other_chainid].front()].cx;
                                for(int i = 0; i < line_box_chain[other_chainid].size(); i++) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    end_cx1 = 0.25 * end_cx0 + 0.75 * boxes[boxid].cx;
                                }

                                if(fabs(start_cx0 - end_cx1) < std::max(w, w1)) {
                                    other_chainid1.push_back(other_chainid);
                                }
                            }
                            done_chain.push_back(other_chainid);
                        } // if other_chainid
                    } // for xk
                    if(done_chain.size() > 0) {
                        break;
                    }
                } // for ys
                find3:
                if(other_chainid1.size() == 1) {
                    //fprintf(stderr, "chain1 %d -> %d\n", other_chainid1, chainid);
                    merge_chain[chainid] = other_chainid1[0];
                }
                // 末尾が連結可能か検索
                std::vector<int> other_chainid2;
                // 下側に検索
                for(int ys = 0; ys < (h + space) * 3.0; ys+=scale) {
                    int y = boxes[boxid2].cy + h / 2 + ys;
                    y /= scale;
                    if(y < 0 || y >= height) continue;
                    for(int xk = -5; xk <= 5; xk++) {
                        int x = end_cx0 + w / 8.0 * xk / 5;
                        x /= scale;
                        if(x < 0 || x >= width) continue;
                        if(lineblocker[y * width + x]) {
                            other_chainid2.clear();
                            goto find4;
                        }
                        int other_chainid = chainid_map[y * width + x];
                        if(other_chainid >= 0 && other_chainid != chainid && 
                                std::find(done_chain.begin(), done_chain.end(), other_chainid) == done_chain.end() &&
                                std::find(other_chainid2.begin(), other_chainid2.end(), other_chainid) == other_chainid2.end()) {

                            if(fabs(boxes[line_box_chain[other_chainid].back()].direction) > M_PI_4 &&
                                boxes[line_box_chain[other_chainid].front()].double_line == boxes[boxid2].double_line) {

                                double w1 = std::transform_reduce(
                                    line_box_chain[other_chainid].begin(), 
                                    line_box_chain[other_chainid].end(), 
                                    0.0, 
                                    [&](double acc, double i) { return std::max(acc, i); },
                                    [&](int x) { return boxes[x].w; });

                                double start_cx1 = boxes[line_box_chain[other_chainid].front()].cx;
                                for(int i = int(line_box_chain[other_chainid].size()) - 1; i >= 0; i--) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    start_cx1 = 0.25 * start_cx1 + 0.75 * boxes[boxid].cx;
                                }

                                if(fabs(end_cx0 - start_cx1) < std::max(w, w1)) {
                                    other_chainid2.push_back(other_chainid);
                                }
                            }
                            done_chain.push_back(other_chainid);
                        } // if other_chainid
                    } // for xk
                    if(done_chain.size() > 0) {
                        break;
                    }
                } // for ys
                find4:
                if(other_chainid2.size() == 1) {
                    //fprintf(stderr, "chain2 %d -> %d\n", chainid, other_chainid2);
                    merge_chain[other_chainid2[0]] = chainid;
                }
            }
        }

        // 連結終了
        if(std::all_of(merge_chain.begin(), merge_chain.end(), [](auto x){ return x == -1; })) {
            break;
        }

        process_merge(line_box_chain, merge_chain);
    }
}

int count_unbind(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    int unbind_count = int(boxes.size());
    for(const auto &chain: line_box_chain) {
        unbind_count -= chain.size();
    }
    return unbind_count;
}

bool fix_unbined(
    const std::vector<bool> &lineblocker,
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain)
{
    fprintf(stderr, "fix_unbined\n");
    fix_chain_info(boxes, line_box_chain);
    int unbind_count = count_unbind(boxes, line_box_chain);
    while(true) {
        fix_chain_info(boxes, line_box_chain);

        // 未接続boxのリストアップ
        std::vector<int> unbined_boxes(boxes.size());
        std::iota(unbined_boxes.begin(), unbined_boxes.end(), 0);
        for(const auto &chain: line_box_chain) {
            for(const auto boxid: chain) {
                std::erase(unbined_boxes, boxid);
            }
        }
        for(auto it = unbined_boxes.begin(); it != unbined_boxes.end();) {
            if((boxes[*it].subtype & (2+4)) == 2+4) {
                it = unbined_boxes.erase(it);
            }
            else {
                ++it;
            }
        }
        // std::cerr << "unbind boxes" << std::endl;
        // std::copy(unbined_boxes.begin(), unbined_boxes.end(), std::ostream_iterator<int>(std::cerr, ","));
        // std::cerr << std::endl;

        // 未接続を繋ぐ
        int success_count = 0;
        for (auto it = unbined_boxes.begin(); it != unbined_boxes.end();) {
            std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, lineblocker);

            std::vector<int> x0;
            std::vector<int> y0;
            x0.push_back(boxes[*it].cx / scale);
            y0.push_back(boxes[*it].cy / scale);
            x0.push_back((boxes[*it].cx - boxes[*it].w/2) / scale);
            y0.push_back((boxes[*it].cy - boxes[*it].h/2) / scale);
            x0.push_back((boxes[*it].cx - boxes[*it].w/4) / scale);
            y0.push_back((boxes[*it].cy - boxes[*it].h/4) / scale);
            x0.push_back((boxes[*it].cx + boxes[*it].w/2) / scale);
            y0.push_back((boxes[*it].cy - boxes[*it].h/2) / scale);
            x0.push_back((boxes[*it].cx + boxes[*it].w/4) / scale);
            y0.push_back((boxes[*it].cy - boxes[*it].h/4) / scale);
            x0.push_back((boxes[*it].cx - boxes[*it].w/2) / scale);
            y0.push_back((boxes[*it].cy + boxes[*it].h/2) / scale);
            x0.push_back((boxes[*it].cx - boxes[*it].w/4) / scale);
            y0.push_back((boxes[*it].cy + boxes[*it].h/4) / scale);
            x0.push_back((boxes[*it].cx + boxes[*it].w/2) / scale);
            y0.push_back((boxes[*it].cy + boxes[*it].h/2) / scale);
            x0.push_back((boxes[*it].cx + boxes[*it].w/4) / scale);
            y0.push_back((boxes[*it].cy + boxes[*it].h/4) / scale);
            for(int i = 0; i < x0.size(); i++) {
                if(x0[i] < 0 || x0[i] >= width || y0[i] < 0 || y0[i] >= height) continue;
                int other_chainid0 = chainid_map[y0[i] * width + x0[i]];
                if(other_chainid0 >= 0) {
                    line_box_chain[other_chainid0].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid0].front()].direction;
                    success_count++;
                    it = unbined_boxes.erase(it);
                    goto next_continue;
                }
            }

            // 左右を検索
            {
                int other_chainid1 = -1;
                for(int xi = boxes[*it].cx + boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w * 1.5; xi+=scale) {
                    int x1 = xi / scale;
                    if(x1 < 0 || x1 >= width) break;
                    for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h / 2; yi+=scale) {
                        int y1 = yi / scale;
                        if(y1 < 0 || y1 >= height) continue;
                        if(lineblocker[y1 * width + x1]) goto next1;
                        other_chainid1 = chainid_map[y1 * width + x1];
                        if(other_chainid1 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid1].front()].direction;
                            if(fabs(direction) < M_PI_4) {
                                // 他のchainが見つかった
                                line_box_chain[other_chainid1].push_back(*it);
                                boxes[*it].direction = boxes[line_box_chain[other_chainid1].front()].direction; 
                                success_count++;
                                it = unbined_boxes.erase(it);
                                goto next_continue;
                            }
                        }
                    }
                }
                next1:
                ;
            }
            {
                int other_chainid1 = -1;
                for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi > boxes[*it].cx - boxes[*it].w * 1.5; xi-=scale) {
                    int x1 = xi / scale;
                    if(x1 < 0 || x1 >= width) break;
                    for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h / 2; yi+=scale) {
                        int y1 = yi / scale;
                        if(y1 < 0 || y1 >= height) continue;
                        if(lineblocker[y1 * width + x1]) goto next2;
                        other_chainid1 = chainid_map[y1 * width + x1];
                        if(other_chainid1 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid1].front()].direction;
                            if(fabs(direction) < M_PI_4) {
                                // 他のchainが見つかった
                                line_box_chain[other_chainid1].push_back(*it);
                                boxes[*it].direction = boxes[line_box_chain[other_chainid1].front()].direction; 
                                success_count++;
                                it = unbined_boxes.erase(it);
                                goto next_continue;
                            }
                        }
                    }
                }
                next2:
                ;
            }

            // 上下側を検索
            {
                int other_chainid2 = -1;
                for(int yi = boxes[*it].cy + boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h * 1.5; yi+=scale) {
                    int y2 = yi / scale;
                    if(y2 < 0 || y2 >= height) break;
                    for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w / 2; xi+=scale) {
                        int x2 = xi / scale;
                        if(x2 < 0 || x2 >= width) continue;
                        if(lineblocker[y2 * width + x2]) goto next3;
                        other_chainid2 = chainid_map[y2 * width + x2];
                        if(other_chainid2 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid2].front()].direction;
                            if(fabs(direction) > M_PI_4) {
                                // 他のchainが見つかった
                                line_box_chain[other_chainid2].push_back(*it);
                                boxes[*it].direction = boxes[line_box_chain[other_chainid2].front()].direction; 
                                success_count++;
                                it = unbined_boxes.erase(it);
                                goto next_continue;
                            }
                        }
                    }
                }
                next3:
                ;
            }
            {
                int other_chainid2 = -1;
                for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi > boxes[*it].cy - boxes[*it].h * 1.5; yi-=scale) {
                    int y2 = yi / scale;
                    if(y2 < 0 || y2 >= height) break;
                    for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w / 2; xi+=scale) {
                        int x2 = xi / scale;
                        if(x2 < 0 || x2 >= width) continue;
                        if(lineblocker[y2 * width + x2]) goto next4;
                        other_chainid2 = chainid_map[y2 * width + x2];
                        if(other_chainid2 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid2].front()].direction;
                            if(fabs(direction) > M_PI_4) {
                                // 他のchainが見つかった
                                line_box_chain[other_chainid2].push_back(*it);
                                boxes[*it].direction = boxes[line_box_chain[other_chainid2].front()].direction; 
                                success_count++;
                                it = unbined_boxes.erase(it);
                                goto next_continue;
                            }
                        }
                    }
                }
                next4:
                ;
            }

            ++it;
            next_continue:
            ;
        }

        if(success_count == 0) break;
    }
    return unbind_count != count_unbind(boxes, line_box_chain);
}

void chain_space(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<float> &sepimage,
    const std::vector<int> &idimage)
{
    fprintf(stderr,"chain_space\n");
    fix_chain_info(boxes, line_box_chain);

    // 未接続boxのリストアップ
    std::vector<int> unbined_boxes(boxes.size());
    std::iota(unbined_boxes.begin(), unbined_boxes.end(), 0);
    for(const auto &chain: line_box_chain) {
        for(const auto boxid: chain) {
            std::erase(unbined_boxes, boxid);
        }
    }

    // 単独でスペースを含む場合は、暫定的にchainに加える
    for(auto it = unbined_boxes.begin(); it != unbined_boxes.end();) {
        // ふりがなは飛ばす
        if((boxes[*it].subtype & (2+4)) == (2+4)) {
            it = unbined_boxes.erase(it);
            continue;
        }
        if((boxes[*it].subtype & 8) != 8) {
            ++it;
        }
        else {
            std::vector<int> new_chain;
            new_chain.push_back(*it);
            line_box_chain.push_back(new_chain);
            it = unbined_boxes.erase(it);
        }
    }

    auto chainid_map = create_chainid_map(boxes, line_box_chain, lineblocker);
    std::vector<int> chain_cont(line_box_chain.size(), -1);
    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() == 0) continue;
        int first_id = line_box_chain[chainid].front();
        if((boxes[first_id].subtype & 8) != 8) continue;

        float direction = boxes[line_box_chain[chainid].front()].direction;

        float ave_dist = 0;
        if(line_box_chain[chainid].size() > 1) {
            for(int i = 0; i < line_box_chain[chainid].size() - 1; i++) {
                if(fabs(direction) < M_PI_4) {
                    // 横書き
                    ave_dist += boxes[line_box_chain[chainid][i+1]].cx - boxes[line_box_chain[chainid][i]].cx;
                }
                else {
                    // 縦書き
                    ave_dist += boxes[line_box_chain[chainid][i+1]].cy - boxes[line_box_chain[chainid][i]].cy;
                }
            }
            ave_dist /= line_box_chain[chainid].size() - 1;
        }
        else {
            if(fabs(direction) < M_PI_4) {
                // 横書き
                ave_dist = boxes[first_id].w;
            }
            else {
                // 縦書き
                ave_dist = boxes[first_id].h;
            }            
        }

        if(fabs(direction) < M_PI_4) {
            // 横書き
            int y0 = boxes[first_id].cy;
            int h = boxes[first_id].h;
            for(int x = boxes[first_id].cx - boxes[first_id].w / 2; x > boxes[first_id].cx - boxes[first_id].w / 2 - ave_dist * 3; x-=scale) {
                int ix = x / scale;
                if(ix < 0 || ix >= width) continue;
                for(int y = y0 - h/2; y < y0 + h/2; y+=scale) {
                    int iy = y / scale;
                    if(iy < 0 || iy >= height) continue;
                    if(sepimage[iy * width + ix] > sep_valueth2) goto find1;
                    int other_chain = chainid_map[iy * width + ix];
                    if(other_chain < 0 || other_chain == chainid) {
                        int other_boxid = idimage[iy * width + ix];
                        if(other_boxid < 0) continue;
                        if(std::find(unbined_boxes.begin(), unbined_boxes.end(), other_boxid) != unbined_boxes.end()) {
                            line_box_chain[chainid].push_back(other_boxid);
                            std::erase(unbined_boxes, other_boxid);
                            boxes[other_boxid].direction = direction;
                            goto find1;
                        }
                        continue;
                    }
                    if(chain_cont[chainid] < 0) {
                        chain_cont[chainid] = other_chain;
                        goto find1;
                    }
                }
            }
            find1:
            ;
        }
        else {
            // 縦書き
            int x0 = boxes[first_id].cx;
            int w = boxes[first_id].w;
            for(int y = boxes[first_id].cy - boxes[first_id].h / 2; y > boxes[first_id].cy - boxes[first_id].h / 2 - ave_dist * 2; y-=scale) {
                int iy = y / scale;
                if(iy < 0 || iy >= height) continue;

                for(int x = x0 - w/2; x < x0 + w/2; x+=scale) {
                    int ix = x / scale;
                    if(ix < 0 || ix >= width) continue;
                    if(sepimage[iy * width + ix] > sep_valueth2) goto find2;
                    int other_chain = chainid_map[iy * width + ix];
                    if(other_chain < 0 || other_chain == chainid) {
                        int other_boxid = idimage[iy * width + ix];
                        if(other_boxid < 0) continue;
                        if(std::find(unbined_boxes.begin(), unbined_boxes.end(), other_boxid) != unbined_boxes.end()) {
                            line_box_chain[chainid].push_back(other_boxid);
                            std::erase(unbined_boxes, other_boxid);
                            boxes[other_boxid].direction = direction;
                            goto find2;
                        }
                        continue;
                    }
                    if(chain_cont[chainid] < 0) {
                        chain_cont[chainid] = other_chain;
                        goto find2;
                    }
                }
            }
            find2:
            ;
        }
    }

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(chain_cont[chainid] < 0) continue;

        std::vector<int> chain;
        int root = chain_cont[chainid];
        while(root >= 0 && std::find(chain.begin(), chain.end(), root) == chain.end()) {
            chain.push_back(root);
            root = chain_cont[root];
        }
        chain_cont[chainid] = root;
    }

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(chain_cont[chainid] < 0) continue;
        int root = chain_cont[chainid];

        std::copy(line_box_chain[chainid].begin(), line_box_chain[chainid].end(), back_inserter(line_box_chain[root]));
        line_box_chain[chainid].clear();
    }

    fix_chain_info(boxes, line_box_chain);
}

void search_loop(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage,
    const std::vector<float> &sepimage)
{
    fprintf(stderr, "loop\n");
    do{
        combine_chains(lineblocker, boxes, line_box_chain);
    }while(fix_unbined(lineblocker, boxes, line_box_chain));

    split_doubleline1(boxes, line_box_chain);
    split_doubleline2(boxes, line_box_chain);
    split_doubleline3(boxes, line_box_chain);

    chain_space(boxes, line_box_chain, lineblocker, sepimage, idimage);

    fprintf(stderr, "loop done\n");
}
