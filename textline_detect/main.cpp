#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <map>
#include <random>
#include <iterator>
#include <utility>

#include <fstream>
#include <iostream>

const double ruby_cutoff = 0.5;
const double rubybase_cutoff = 0.4;
const double space_cutoff = 0.5;
const float line_valueth = 0.25;
const float sep_valueth = 0.25;
const float sep_valueth2 = 0.3;
const float sep_clusterth = 50.0;
const int linearea_th = 50;

int run_mode = 0;
int width = 0;
int height = 0;

int org_width = 0;
int org_height = 0;

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

void sort_chain(
    std::vector<int> &chain,
    const std::vector<charbox> &boxes) 
{
    // chain内をソート
    if(fabs(boxes[chain.front()].direction) < M_PI_4) {
        // 横書き
        std::sort(chain.begin(), chain.end(), [boxes](auto a, auto b){
            if((boxes[b].cx - boxes[b].w/2 < boxes[a].cx && boxes[a].cx < boxes[b].cx + boxes[b].w/2)
                || (boxes[a].cx - boxes[a].w/2 < boxes[b].cx && boxes[b].cx < boxes[a].cx + boxes[a].w/2)) {
                    return boxes[a].cy < boxes[b].cy;
            }
            return boxes[a].cx < boxes[b].cx;
        });
    }
    else {
        // 縦書き
        std::sort(chain.begin(), chain.end(), [boxes](auto a, auto b){
            if((boxes[b].cy - boxes[b].h/2 < boxes[a].cy && boxes[a].cy < boxes[b].cy + boxes[b].h/2)
                || (boxes[a].cy - boxes[a].h/2 < boxes[b].cy && boxes[b].cy < boxes[a].cy + boxes[a].h/2)) {
                    return boxes[a].cx < boxes[b].cx;
            }
            return boxes[a].cy < boxes[b].cy;
        });
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
    direction = boxes[chain.front()].direction;
    w = 0;
    h = 0;
    int count = 0;
    if(fabs(direction) < M_PI_4) {
        // 横書き
        start_cy = 0;
        double sum = 0;
        for(int i = 0; i < chain.size(); i++) {
            if((boxes[chain[i]].subtype & (2+4)) == 2+4) continue;
            double weight = boxes[chain[i]].w * boxes[chain[i]].h / (i + 1);
            start_cy += boxes[chain[i]].cy * weight;
            sum += weight;
            w += boxes[chain[i]].w;
            h += boxes[chain[i]].h;
            count++;
        }
        if(count > 0) {
            w /= count;
            h /= count;
        }
        start_cy /= sum;
        start_cx = boxes[chain.front()].cx;

        end_cy = 0;
        sum = 0;
        for(int i = 0; i < chain.size(); i++) {
            if((boxes[chain[i]].subtype & (2+4)) == (2+4)) continue;
            double weight = boxes[chain[i]].w * boxes[chain[i]].h / (chain.size() - i);
            end_cy += boxes[chain[i]].cy * weight;
            sum += weight;
        }
        end_cy /= sum;
        end_cx = boxes[chain.back()].cx;
    }
    else {
        // 縦書き
        start_cx = 0;
        double sum = 0;
        for(int i = 0; i < chain.size(); i++) {
            if((boxes[chain[i]].subtype & (2+4)) == (2+4)) continue;
            double weight = boxes[chain[i]].w * boxes[chain[i]].h / (i + 1);
            start_cx += boxes[chain[i]].cx * weight;
            sum += weight;
            w += boxes[chain[i]].w;
            h += boxes[chain[i]].h;
            count++;
        }
        if(count > 0) {
            w /= count;
            h /= count;
        }
        start_cx /= sum;
        start_cy = boxes[chain.front()].cy;

        end_cx = 0;
        sum = 0;
        for(int i = 0; i < chain.size(); i++) {
            if((boxes[chain[i]].subtype & (2+4)) == (2+4)) continue;
            double weight = boxes[chain[i]].w * boxes[chain[i]].h / (chain.size() - i);
            end_cx += boxes[chain[i]].cx * weight;
            sum += weight;
        }
        end_cx /= sum;
        end_cy = boxes[chain.back()].cy;
    }
}

int count_unbind(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    int unbind_count = boxes.size();
    for(const auto &chain: line_box_chain) {
        unbind_count -= chain.size();
    }
    return unbind_count;
}

void calc_line(
    const std::vector<float> x,
    const std::vector<float> y,
    const std::vector<float> w,
    double &a, double &b)
{
    double sum1 = 0;
    double sumXY = 0;
    double sumX = 0;
    double sumY = 0;
    double sumX2 = 0;
    for(int i = 0; i < x.size(); i++) {
        double weight = w[i];
        sum1 += weight;
        sumXY += x[i] * y[i] * weight;
        sumX += x[i] * weight;
        sumY += y[i] * weight;
        sumX2 += x[i] * x[i] * weight;
    }
    double delta = sum1 * sumX2 - sumX * sumX;
    a = (sum1 * sumXY - sumX * sumY) / delta;
    b = (sumX2 * sumY - sumX * sumXY) / delta;
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
    int chainid,
    int extra_len = 0)
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
            for(int xp = x1 - extra_len * w * 2; xp < x1; xp += 2) {
                int yp = (xp - x1) * a + y1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }

        for(int i = 0; i < xi.size() - 1; i++) {
            double x1 = xi[i];
            double y1 = yi[i];
            double x2 = xi[i+1];
            double y2 = yi[i+1];
            double a = (y2 - y1) / (x2 - x1);
            if(!isfinite(a)) continue;
            for(int xp = x1; xp < x2; xp += 2) {
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
            for(int xp = x2; xp < x2 + extra_len * w * 2; xp += 2) {
                int yp = (xp - x1) * a + y1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }
    }
    else {
        // 縦書き
        int len = end_cy - start_cy;
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
            for(int yp = y1 - extra_len * h * 2; yp < y1; yp += 2) {
                int xp = (yp - y1) * a + x1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }

        for(int i = 0; i < xi.size() - 1; i++) {
            double x1 = xi[i];
            double y1 = yi[i];
            double x2 = xi[i+1];
            double y2 = yi[i+1];
            double a = (x2 - x1) / (y2 - y1);
            if(!isfinite(a)) continue;
            for(int yp = y1; yp < y2; yp += 2) {
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
            for(int yp = y2; yp < y2 + extra_len * h * 2; yp += 2) {
                int xp = (yp - y1) * a + x1;
                x.push_back(xp);
                y.push_back(yp);
            }
        }
    }
}

// 暫定chain id のマップを作る
std::vector<int> create_chainid_map(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    double ratio = 1.0)
{
    //std::cerr << "create_chainid_map" << std::endl;
    std::vector<int> chainid_map(width*height, -1);

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() == 0) continue;

        std::vector<int> x;
        std::vector<int> y;
        float direction;
        double w, h;
        make_track_line(x,y, direction, w, h, boxes, line_box_chain, chainid);

        if(fabs(direction) < M_PI_4) {
            // 横書き
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / 2;
                int yi = y[i] / 2;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int yp = yi - h/4 * ratio; yp < yi + h/4 * ratio; yp++) {
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
                int xi = x[i] / 2;
                int yi = y[i] / 2;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int xp = xi - w/4 * ratio; xp < xi + w/4 * ratio; xp++) {
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
            int left = (box.cx - box.w / 2) / 2;
            int right = (box.cx + box.w / 2) / 2;
            int top = (box.cy - box.h / 2) / 2;
            int bottom = (box.cy + box.h / 2) / 2;
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

    // {
    //     std::ofstream outfile("chainidmap.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << chainid_map[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }

    return chainid_map;
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
    }
}

void process_merge(
    std::vector<std::vector<int>> &line_box_chain,
    std::vector<int> &merge_chain)
{
    // chainの連結の処理
    std::vector<int> root_id(line_box_chain.size(), -1);
    int new_id = line_box_chain.size();
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
                new_id = line_box_chain.size();
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
        std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain);

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
                double start_cy0 = 0;
                double weight = 0;
                for(int i = line_box_chain[chainid].size() - 1; i >= 0; i--) {
                    int boxid = line_box_chain[chainid][i];
                    double weight1 = boxes[boxid].w * boxes[boxid].h / (i + 1);
                    start_cy0 += weight1 * boxes[boxid].cy;
                    weight += weight1;
                }
                start_cy0 /= weight;

                double end_cy0 = 0;
                weight = 0;
                for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                    int boxid = line_box_chain[chainid][i];
                    double weight1 = boxes[boxid].w * boxes[boxid].h / (line_box_chain[chainid].size() - i);
                    end_cy0 += weight1 * boxes[boxid].cy;
                    weight += weight1;
                }
                end_cy0 /= weight;

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
                for(int xs = 0; xs < (w + space) * 5.0; xs+=2) {
                    int x = boxes[boxid1].cx - w / 2 - xs;
                    x /= 2;
                    if(x < 0 || x >= width) continue;
                    for(int yk = -5; yk <= 5; yk++) {
                        int y = start_cy0 + h / 4.0 * yk / 5;
                        y /= 2;
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

                                double end_cy1 = 0;
                                weight = 0;
                                for(int i = 0; i < line_box_chain[other_chainid].size(); i++) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    double weight1 = boxes[boxid].w * boxes[boxid].h / (line_box_chain[other_chainid].size() - i);
                                    end_cy1 += weight1 * boxes[boxid].cy;
                                    weight += weight1;
                                }
                                end_cy1 /= weight;
                                //fprintf(stderr, "chain1 %d -> %d test %f %f %f\n", other_chainid1, chainid, start_cy0, end_cy1, std::max(h, h1));

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
                for(int xs = 0; xs < (w + space) * 5.0; xs+=2) {
                    int x = boxes[boxid2].cx + w / 2 + xs;
                    x /= 2;
                    if(x < 0 || x >= width) continue;
                    for(int yk = -5; yk <= 5; yk++) {
                        int y = end_cy0 + h / 4.0 * yk / 5;
                        y /= 2;
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

                                double start_cy1 = 0;
                                weight = 0;
                                for(int i = line_box_chain[other_chainid].size() - 1; i >= 0; i--) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    double weight1 = boxes[boxid].w * boxes[boxid].h / (i + 1);
                                    start_cy1 += weight1 * boxes[boxid].cy;
                                    weight += weight1;
                                }
                                start_cy1 /= weight;

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
                double start_cx0 = 0;
                double weight = 0;
                for(int i = line_box_chain[chainid].size() - 1; i >= 0; i--) {
                    int boxid = line_box_chain[chainid][i];
                    double weight1 = boxes[boxid].w * boxes[boxid].h / (i + 1);
                    start_cx0 += weight1 * boxes[boxid].cx;
                    weight += weight1;
                }
                start_cx0 /= weight;

                double end_cx0 = 0;
                weight = 0;
                for(int i = 0; i < line_box_chain[chainid].size(); i++) {
                    int boxid = line_box_chain[chainid][i];
                    double weight1 = boxes[boxid].w * boxes[boxid].h / (line_box_chain[chainid].size() - i);
                    end_cx0 += weight1 * boxes[boxid].cx;
                    weight += weight1;
                }
                end_cx0 /= weight;

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
                for(int ys = 0; ys < (h + space) * 5.0; ys+=2) {
                    int y = boxes[boxid1].cy - h / 2 - ys;
                    y /= 2;
                    if(y < 0 || y >= height) continue;
                    for(int xk = -5; xk <= 5; xk++) {
                        int x = start_cx0 + w / 4.0 * xk / 5;
                        x /= 2;
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

                                double end_cx1 = 0;
                                weight = 0;
                                for(int i = 0; i < line_box_chain[other_chainid].size(); i++) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    double weight1 = boxes[boxid].w * boxes[boxid].h / (line_box_chain[other_chainid].size() - i);
                                    end_cx1 += weight1 * boxes[boxid].cx;
                                    weight += weight1;
                                }
                                end_cx1 /= weight;

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
                for(int ys = 0; ys < (h + space) * 5.0; ys+=2) {
                    int y = boxes[boxid2].cy + h / 2 + ys;
                    y /= 2;
                    if(y < 0 || y >= height) continue;
                    for(int xk = -5; xk <= 5; xk++) {
                        int x = end_cx0 + w / 4.0 * xk / 5;
                        x /= 2;
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

                                double start_cx1 = 0;
                                weight = 0;
                                for(int i = line_box_chain[other_chainid].size() - 1; i >= 0; i--) {
                                    int boxid = line_box_chain[other_chainid][i];
                                    double weight1 = boxes[boxid].w * boxes[boxid].h / (i + 1);
                                    start_cx1 += weight1 * boxes[boxid].cy;
                                    weight += weight1;
                                }
                                start_cx1 /= weight;

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

        std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, 2.0);

        // 未接続boxのリストアップ
        std::vector<int> unbined_boxes(boxes.size());
        std::iota(unbined_boxes.begin(), unbined_boxes.end(), 0);
        for(const auto &chain: line_box_chain) {
            for(const auto boxid: chain) {
                std::erase(unbined_boxes, boxid);
            }
        }
        for(auto it = unbined_boxes.begin(); it != unbined_boxes.end();) {
            if(boxes[*it].code1 > ruby_cutoff) {
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
            bool bind_success = false;

            std::vector<int> x0;
            std::vector<int> y0;
            x0.push_back(boxes[*it].cx / 2);
            y0.push_back(boxes[*it].cy / 2);
            x0.push_back((boxes[*it].cx - boxes[*it].w/2) / 2);
            y0.push_back((boxes[*it].cy - boxes[*it].h/2) / 2);
            x0.push_back((boxes[*it].cx - boxes[*it].w/4) / 2);
            y0.push_back((boxes[*it].cy - boxes[*it].h/4) / 2);
            x0.push_back((boxes[*it].cx + boxes[*it].w/2) / 2);
            y0.push_back((boxes[*it].cy - boxes[*it].h/2) / 2);
            x0.push_back((boxes[*it].cx + boxes[*it].w/4) / 2);
            y0.push_back((boxes[*it].cy - boxes[*it].h/4) / 2);
            x0.push_back((boxes[*it].cx - boxes[*it].w/2) / 2);
            y0.push_back((boxes[*it].cy + boxes[*it].h/2) / 2);
            x0.push_back((boxes[*it].cx - boxes[*it].w/4) / 2);
            y0.push_back((boxes[*it].cy + boxes[*it].h/4) / 2);
            x0.push_back((boxes[*it].cx + boxes[*it].w/2) / 2);
            y0.push_back((boxes[*it].cy + boxes[*it].h/2) / 2);
            x0.push_back((boxes[*it].cx + boxes[*it].w/4) / 2);
            y0.push_back((boxes[*it].cy + boxes[*it].h/4) / 2);
            for(int i = 0; i < x0.size() && !bind_success; i++) {
                if(x0[i] < 0 || x0[i] >= width || y0[i] < 0 || y0[i] >= height) continue;
                int other_chainid0 = chainid_map[y0[i] * width + x0[i]];
                if(other_chainid0 >= 0) {
                    line_box_chain[other_chainid0].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid0].front()].direction;
                    bind_success = true;
                }
            }

            // 左右を検索
            if(!bind_success) {
                int other_chainid1 = -1;
                for(int xi = boxes[*it].cx + boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w * 1.5; xi+=2) {
                    int x1 = xi / 2;
                    if(x1 < 0 || x1 >= width) break;
                    for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h / 2; yi+=2) {
                        int y1 = yi / 2;
                        if(y1 < 0 || y1 >= height) continue;
                        if(lineblocker[y1 * width + x1]) goto next1;
                        other_chainid1 = chainid_map[y1 * width + x1];
                        if(other_chainid1 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid1].front()].direction;
                            if(fabs(direction) < M_PI_4) {
                                bind_success = true;
                                break;
                            }
                        }
                    }
                    if(bind_success) break;
                }
                next1:
                if(bind_success) {
                    // 他のchainが見つかった
                    line_box_chain[other_chainid1].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid1].front()].direction; 
                }
            }
            if(!bind_success) {
                int other_chainid1 = -1;
                for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi > boxes[*it].cx - boxes[*it].w * 1.5; xi-=2) {
                    int x1 = xi / 2;
                    if(x1 < 0 || x1 >= width) break;
                    for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h / 2; yi+=2) {
                        int y1 = yi / 2;
                        if(y1 < 0 || y1 >= height) continue;
                        if(lineblocker[y1 * width + x1]) goto next2;
                        other_chainid1 = chainid_map[y1 * width + x1];
                        if(other_chainid1 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid1].front()].direction;
                            if(fabs(direction) < M_PI_4) {
                                bind_success = true;
                                break;
                            }
                        }
                    }
                    if(bind_success) break;
                }
                next2:
                if(bind_success) {
                    // 他のchainが見つかった
                    line_box_chain[other_chainid1].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid1].front()].direction; 
                }
            }

            // 上下側を検索
            if(!bind_success) {
                int other_chainid2 = -1;
                for(int yi = boxes[*it].cy + boxes[*it].h / 2; yi < boxes[*it].cy + boxes[*it].h * 1.5; yi+=2) {
                    int y2 = yi / 2;
                    if(y2 < 0 || y2 >= height) break;
                    for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w / 2; xi+=2) {
                        int x2 = xi / 2;
                        if(x2 < 0 || x2 >= width) continue;
                        if(lineblocker[y2 * width + x2]) goto next3;
                        other_chainid2 = chainid_map[y2 * width + x2];
                        if(other_chainid2 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid2].front()].direction;
                            if(fabs(direction) > M_PI_4) {
                                bind_success = true;
                                break;
                            }
                        }
                    }
                    if(bind_success) break;
                }
                next3:
                if(bind_success) {
                    // 他のchainが見つかった
                    line_box_chain[other_chainid2].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid2].front()].direction; 
                }
            }
            if(!bind_success) {
                int other_chainid2 = -1;
                for(int yi = boxes[*it].cy - boxes[*it].h / 2; yi > boxes[*it].cy - boxes[*it].h * 1.5; yi-=2) {
                    int y2 = yi / 2;
                    if(y2 < 0 || y2 >= height) break;
                    for(int xi = boxes[*it].cx - boxes[*it].w / 2; xi < boxes[*it].cx + boxes[*it].w / 2; xi+=2) {
                        int x2 = xi / 2;
                        if(x2 < 0 || x2 >= width) continue;
                        if(lineblocker[y2 * width + x2]) goto next4;
                        other_chainid2 = chainid_map[y2 * width + x2];
                        if(other_chainid2 >= 0) {
                            float direction = boxes[line_box_chain[other_chainid2].front()].direction;
                            if(fabs(direction) > M_PI_4) {
                                bind_success = true;
                                break;
                            }
                        }
                    }
                    if(bind_success) break;
                }
                next4:
                if(bind_success) {
                    // 他のchainが見つかった
                    line_box_chain[other_chainid2].push_back(*it);
                    boxes[*it].direction = boxes[line_box_chain[other_chainid2].front()].direction; 
                }
            }

            if (bind_success) {
                it = unbined_boxes.erase(it);
                success_count++;
                chainid_map = create_chainid_map(boxes, line_box_chain);
            }
            else {
                ++it;
            }
        }

        if(success_count == 0) break;
    }
    return unbind_count != count_unbind(boxes, line_box_chain);
}

// 方向の違う行が混ざっている場合分離する
void split_doubleline1(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain) 
{
    std::cerr << "split_doubleline1" << std::endl;
    fix_chain_info(boxes, line_box_chain);
    for(auto chain_it = line_box_chain.begin(); chain_it != line_box_chain.end(); ++chain_it) {
        if(chain_it->size() < 3) continue;

        std::vector<int> splited;

        int front_id = chain_it->front();
        int back_id = chain_it->back();
        float max_h = 0;
        float max_w = 0;
        for(const auto boxid: *chain_it) {
            max_w = std::max(max_w, boxes[boxid].w);
            max_h = std::max(max_h, boxes[boxid].h);
        }

        float direction = boxes[front_id].direction;
        // 方向の違う行が混ざっている場合分離する
        for(const auto boxid: *chain_it) {
            if(abs(direction) < M_PI_4) {
                //横書き
                if(abs(boxes[boxid].direction) < M_PI_4) {
                }
                else {
                    splited.push_back(boxid);
                }
            }
            else {
                //縦書き
                if(abs(boxes[boxid].direction) < M_PI_4) {
                    splited.push_back(boxid);
                }
                else {
                }
            }
        }

        if(splited.size() == chain_it->size()) {
            continue;
        }

        if(splited.size() > 0) {
            for(auto it = chain_it->begin(); it != chain_it->end();) {
                if(std::find(splited.begin(), splited.end(), *it) != splited.end()) {
                    it = chain_it->erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        if(splited.size() >= 2) {
            sort_chain(splited, boxes);
            chain_it = line_box_chain.insert(chain_it, splited);
        }
    }
}

// 距離の離れすぎている文字は分離する
void split_doubleline2(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain) 
{
    std::cerr << "split_doubleline2" << std::endl;
    fix_chain_info(boxes, line_box_chain);
    for(auto chain_it = line_box_chain.begin(); chain_it != line_box_chain.end(); ++chain_it) {
        if(chain_it->size() < 3) continue;

        std::vector<int> splited;

        int front_id = chain_it->front();
        int back_id = chain_it->back();
        float max_h = 0;
        float max_w = 0;
        for(const auto boxid: *chain_it) {
            max_w = std::max(max_w, boxes[boxid].w);
            max_h = std::max(max_h, boxes[boxid].h);
        }

        // 距離の離れすぎている文字は分離する
        float direction = boxes[front_id].direction;
        bool split_flag = false;
        if(abs(direction) < M_PI_4) {
            //横書き
            float x1 = boxes[front_id].cx;
            float y1 = boxes[front_id].cy;
            for(const auto boxid: *chain_it) {
                float x = boxes[boxid].cx;
                float y = boxes[boxid].cy;

                if(fabs(x - x1) > 10 * std::max(max_w, max_h)) {
                    split_flag = true;    
                }
                if(split_flag) {
                    splited.push_back(boxid);
                }
                x1 = x;
                y1 = y;
            }
        }
        else {
            //縦書き
            float x1 = boxes[front_id].cx;
            float y1 = boxes[front_id].cy;

            for(const auto boxid: *chain_it) {
                float x = boxes[boxid].cx;
                float y = boxes[boxid].cy;

                if(fabs(y - y1) > 2 * std::max(max_w, max_h)) {
                    split_flag = true;    
                }
                if(split_flag) {
                    splited.push_back(boxid);
                }
                x1 = x;
                y1 = y;
            }
        }

        if(splited.size() == chain_it->size()) {
            continue;
        }

        if(splited.size() > 0) {
            for(auto it = chain_it->begin(); it != chain_it->end();) {
                if(std::find(splited.begin(), splited.end(), *it) != splited.end()) {
                    it = chain_it->erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        if(splited.size() >= 2) {
            sort_chain(splited, boxes);
            chain_it = line_box_chain.insert(chain_it, splited);
        }
    }
}

// 途中で2行に分かれている場合は分離する
void split_doubleline3(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain) 
{
    std::cerr << "split_doubleline3" << std::endl;
    fix_chain_info(boxes, line_box_chain);
    for(auto chain_it = line_box_chain.begin(); chain_it != line_box_chain.end(); ++chain_it) {
        if(chain_it->size() < 3) continue;
        if(std::count_if(chain_it->begin(), chain_it->end(), [boxes](int i){ return boxes[i].double_line > 0; }) > 0) continue;

        int front_id = chain_it->front();
        int back_id = chain_it->back();
        float max_h = 0;
        float max_w = 0;
        for(const auto boxid: *chain_it) {
            max_w = std::max(max_w, boxes[boxid].w);
            max_h = std::max(max_h, boxes[boxid].h);
        }

        // 途中で2行に分かれている場合は分離する
        float direction = boxes[front_id].direction;
        if(abs(direction) < M_PI_4) {
            //横書き
            float last_sx = -1;
            float last_ex = -1;
            int last_idx = -1;
            for(const auto boxid: *chain_it) {
                if((boxes[boxid].subtype & (2+4)) == 2+4) continue;
                float cx = boxes[boxid].cx;
                float w = boxes[boxid].w;
                float minx = std::max(last_sx, cx - w/2);
                float maxx = std::min(last_ex, cx + w/2);
                if(last_idx >= 0) {
                    if(minx < maxx && (maxx - minx) > w * 0.2) {
                        if(boxes[last_idx].cy < boxes[boxid].cy) {
                            if(boxes[last_idx].cy + boxes[last_idx].h / 2 < boxes[boxid].cy - boxes[boxid].h / 2) {
                                boxes[last_idx].double_line = 1;
                                boxes[boxid].double_line = 2;
                            }
                        }
                        else {
                            if(boxes[boxid].cy + boxes[boxid].h / 2 < boxes[last_idx].cy - boxes[last_idx].h / 2) {
                                boxes[last_idx].double_line = 2;
                                boxes[boxid].double_line = 1;
                            }
                        }
                    }
                }
                last_sx = cx - w/2;
                last_ex = cx + w/2;
                last_idx = boxid;
            }
            if(std::count_if(chain_it->begin(), chain_it->end(), [boxes](int i){ return boxes[i].double_line > 0; }) > 0) {
                std::vector<float> h1;
                std::vector<float> s1;
                std::vector<float> cy1;
                std::vector<float> cy2;
                for(const auto boxid: *chain_it) {
                    if(boxes[boxid].double_line == 1) {
                        h1.push_back(boxes[boxid].h);
                        s1.push_back(std::max(boxes[boxid].h, boxes[boxid].w));
                        cy1.push_back(boxes[boxid].cy);
                    }
                    else if(boxes[boxid].double_line == 2) {
                        h1.push_back(boxes[boxid].h);
                        s1.push_back(std::max(boxes[boxid].h, boxes[boxid].w));
                        cy2.push_back(boxes[boxid].cy);
                    }
                }
                float h_s = std::accumulate(h1.begin(), h1.end(), 0.0) / h1.size();
                float s_s = std::accumulate(s1.begin(), s1.end(), 0.0) / s1.size();
                float cy1_s = (cy1.size() > 0) ? std::accumulate(cy1.begin(), cy1.end(), 0.0) / cy1.size() : -1;
                float cy2_s = (cy2.size() > 0) ? std::accumulate(cy2.begin(), cy2.end(), 0.0) / cy2.size() : -1;
                int splitcount = 0;
                for(const auto boxid: *chain_it) {
                    if(boxes[boxid].double_line > 0) {
                        splitcount++;
                    }
                    if(splitcount > 0 && boxes[boxid].double_line == 0) {
                        if(fabs(boxes[boxid].cy - cy1_s) < h_s / 4) {
                            boxes[boxid].double_line = 1;
                        }
                        else if(fabs(boxes[boxid].cy - cy2_s) < h_s / 4) {
                            boxes[boxid].double_line = 2;
                        }
                        else {
                            splitcount = 0;
                        }
                    }
                }
            }
        }
        else {
            //縦書き
            float last_sy = -1;
            float last_ey = -1;
            int last_idx = -1;
            for(const auto boxid: *chain_it) {
                if((boxes[boxid].subtype & (2+4)) == 2+4) continue;
                float cy = boxes[boxid].cy;
                float h = boxes[boxid].h;
                float miny = std::max(last_sy, cy - h/2);
                float maxy = std::min(last_ey, cy + h/2);
                if(last_idx >= 0) {
                    if(miny < maxy && (maxy - miny) > h * 0.2) {
                        if(boxes[last_idx].cx > boxes[boxid].cx) {
                            if(boxes[boxid].cx + boxes[boxid].w / 2 < boxes[last_idx].cx - boxes[last_idx].w / 2) {
                                boxes[last_idx].double_line = 1;
                                boxes[boxid].double_line = 2;
                            }
                        }
                        else {
                            if(boxes[last_idx].cx + boxes[last_idx].w / 2 < boxes[boxid].cx - boxes[boxid].w / 2) {
                                boxes[last_idx].double_line = 2;
                                boxes[boxid].double_line = 1;
                            }
                        }
                    }
                }
                last_sy = cy - h/2;
                last_ey = cy + h/2;
                last_idx = boxid;
            }

            if(std::count_if(chain_it->begin(), chain_it->end(), [boxes](int i){ return boxes[i].double_line > 0; }) > 0) {
                std::vector<float> w1;
                std::vector<float> s1;
                std::vector<float> cx1;
                std::vector<float> cx2;
                for(const auto boxid: *chain_it) {
                    if(boxes[boxid].double_line == 1) {
                        w1.push_back(boxes[boxid].w);
                        s1.push_back(std::max(boxes[boxid].h, boxes[boxid].w));
                        cx1.push_back(boxes[boxid].cx);
                    }
                    else if(boxes[boxid].double_line == 2) {
                        w1.push_back(boxes[boxid].w);
                        s1.push_back(std::max(boxes[boxid].h, boxes[boxid].w));
                        cx2.push_back(boxes[boxid].cx);
                    }
                }
                float w_s = std::accumulate(w1.begin(), w1.end(), 0.0) / w1.size();
                float s_s = std::accumulate(s1.begin(), s1.end(), 0.0) / s1.size();
                float cx1_s = (cx1.size() > 0) ? std::accumulate(cx1.begin(), cx1.end(), 0.0) / cx1.size() : -1;
                float cx2_s = (cx2.size() > 0) ? std::accumulate(cx2.begin(), cx2.end(), 0.0) / cx2.size() : -1;
                int splitcount = 0;
                for(const auto boxid: *chain_it) {
                    if(boxes[boxid].double_line > 0) {
                        splitcount++;
                    }
                    if(splitcount > 0 && boxes[boxid].double_line == 0) {
                        if(fabs(boxes[boxid].cx - cx1_s) < w_s / 4) {
                            boxes[boxid].double_line = 1;
                        }
                        else if(fabs(boxes[boxid].cx - cx2_s) < w_s / 4) {
                            boxes[boxid].double_line = 2;
                        }
                        else {
                            splitcount = 0;
                        }
                    }
                }
            }
        }

        while(std::count_if(chain_it->begin(), chain_it->end(), [boxes](int i){ return boxes[i].double_line > 0; }) > 0) {
            std::vector<int> splited1;
            std::vector<int> splited2;
            std::vector<int> remain;

            for(auto it = chain_it->begin(); it != chain_it->end();++it) {
                if(boxes[*it].double_line == 1) {
                    splited1.push_back(*it);
                }
                else if (boxes[*it].double_line == 2) {
                    splited2.push_back(*it);
                }
                else {
                    if (splited1.size() > 1 && splited2.size() > 1) {
                        std::copy(it, chain_it->end(), back_inserter(remain));
                        break;
                    }
                    else {
                        splited1.clear();
                        splited2.clear();
                    }
                }
            }
            if(splited1.size() > 0) {
                for(auto it = chain_it->begin(); it != chain_it->end();) {
                    if(std::find(splited1.begin(), splited1.end(), *it) != splited1.end()) {
                        it = chain_it->erase(it);
                    }
                    else {
                        ++it;
                    }
                }
            }
            if(splited2.size() > 0) {
                for(auto it = chain_it->begin(); it != chain_it->end();) {
                    if(std::find(splited2.begin(), splited2.end(), *it) != splited2.end()) {
                        it = chain_it->erase(it);
                    }
                    else {
                        ++it;
                    }
                }
            }
            if(remain.size() > 0) {
                for(auto it = chain_it->begin(); it != chain_it->end();) {
                    if(std::find(remain.begin(), remain.end(), *it) != remain.end()) {
                        it = chain_it->erase(it);
                    }
                    else {
                        ++it;
                    }
                }
            }
            if(splited1.size() >= 2) {
                sort_chain(splited1, boxes);
                chain_it = line_box_chain.insert(chain_it, splited1);
            }
            if(splited2.size() >= 2) {
                sort_chain(splited2, boxes);
                chain_it = line_box_chain.insert(chain_it, splited2);
            }
            if(remain.size() >= 2) {
                sort_chain(remain, boxes);
                chain_it = line_box_chain.insert(chain_it, remain);
            }

            if(remain.size() == 0) {
                break;
            }
        }
    }
}

int number_unbind(
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage,
    int next_id)
{
    const double allow_maindiff = 1;
    const double allow_subdiff = 10;
    std::vector<int> unbind_boxes;
    std::vector<float> cx;
    std::vector<float> cy;
    for(const auto &box: boxes) {
        if(box.idx < 0) {
            unbind_boxes.push_back(box.id);
            cx.push_back(box.cx);
            cy.push_back(box.cy);
        }
    }
    if(unbind_boxes.size() == 0) {
        return next_id;
    }
    std::cerr << "number_unbind " << unbind_boxes.size() << std::endl;

    std::vector<std::vector<int>> hori_line_boxid;
    std::vector<std::vector<int>> vert_line_boxid;

    std::vector<int> x_sortidx(cx.size());
    std::iota(x_sortidx.begin(), x_sortidx.end(), 0);
    std::sort(x_sortidx.begin(), x_sortidx.end(), [&](int a, int b) {
        return cx[a] < cx[b];
    });
    std::vector<int> y_sortidx(cy.size());
    std::iota(y_sortidx.begin(), y_sortidx.end(), 0);
    std::sort(y_sortidx.begin(), y_sortidx.end(), [&](int a, int b) {
        return cy[a] < cy[b];
    });

    {
        // 水平方向の文字列があるかも
        // cyがどこかに固まっている

        std::vector<float> sorted_cy;
        for(const auto i: y_sortidx) {
            sorted_cy.push_back(cy[i]);
        }
        std::vector<float> diff_cy(sorted_cy.size());
        std::adjacent_difference(sorted_cy.begin(), sorted_cy.end(), diff_cy.begin());

        std::vector<std::vector<int>> agg_idx;
        for(int i = 1; i < diff_cy.size(); i++) {
            int boxid1 = unbind_boxes[y_sortidx[i-1]];
            int boxid2 = unbind_boxes[y_sortidx[i]];
            float s1 = std::max(boxes[boxid1].w, boxes[boxid1].h);
            float s2 = std::max(boxes[boxid2].w, boxes[boxid2].h);
            float s = std::max(s1, s2);
            if(diff_cy[i] < s * allow_maindiff) {
                int chain_idx = -1;
                for(int j = 0; j < agg_idx.size(); j++) {
                    if(std::find(agg_idx[j].begin(), agg_idx[j].end(), boxid1) != agg_idx[j].end()) {
                        chain_idx = j;
                        break;
                    }
                }
                if(chain_idx < 0) {
                    std::vector<int> tmp_chain;
                    tmp_chain.push_back(boxid1);
                    tmp_chain.push_back(boxid2);
                    agg_idx.push_back(tmp_chain);
                }
                else {
                    agg_idx[chain_idx].push_back(boxid2);
                }
            }
        }

        std::vector<std::vector<int>> line_boxid;
        for(const auto &chain: agg_idx) {
            std::vector<float> cx2;
            std::vector<float> cy2;

            for(const auto boxidx: chain) {
                cx2.push_back(boxes[boxidx].cx);
                cy2.push_back(boxes[boxidx].cy);
            }

            std::vector<int> x_sortidx2(cx2.size());
            std::iota(x_sortidx2.begin(), x_sortidx2.end(), 0);
            std::sort(x_sortidx2.begin(), x_sortidx2.end(), [&](int a, int b) {
                return cx2[a] < cx2[b];
            });

            std::vector<float> sorted_cx2;
            for(const auto i: x_sortidx2) {
                sorted_cx2.push_back(cx2[i]);
            }
            std::vector<float> diff_cx2(sorted_cx2.size());
            std::adjacent_difference(sorted_cx2.begin(), sorted_cx2.end(), diff_cx2.begin());

            std::vector<float> sorted_cy2;
            for(const auto i: x_sortidx2) {
                sorted_cy2.push_back(cy2[i]);
            }
            std::vector<float> diff_cy2(sorted_cy2.size());
            std::adjacent_difference(sorted_cy2.begin(), sorted_cy2.end(), diff_cy2.begin());

            std::vector<int> valid_diffy(sorted_cy2.size());
            for(int i = 0; i < x_sortidx2.size(); i++) {
                int boxid = chain[x_sortidx2[i]];
                float s = std::max(boxes[boxid].w, boxes[boxid].h);
                if(i > 0 && fabs(diff_cy2[i]) < s * allow_maindiff) {
                    // ひとつ前との差分
                    valid_diffy[x_sortidx2[i-1]] |= 1;
                    valid_diffy[x_sortidx2[i]] |= 2;
                }
                if(i + 1 < sorted_cy2.size() && fabs(diff_cy2[i+1]) < s * allow_maindiff) {
                    // ひとつ後ろとの差分
                    valid_diffy[x_sortidx2[i+1]] |= 2;
                    valid_diffy[x_sortidx2[i]] |= 1;
                }
            }

            std::vector<int> valid_diffx(sorted_cx2.size());
            for(int i = 0; i < x_sortidx2.size(); i++) {
                int boxid = chain[x_sortidx2[i]];
                float s = std::max(boxes[boxid].w, boxes[boxid].h);
                if(i > 0 && diff_cx2[i] < s * allow_subdiff) {
                    // ひとつ前との差分
                    valid_diffx[x_sortidx2[i-1]] |= 1;
                    valid_diffx[x_sortidx2[i]] |= 2;
                }
                if(i + 1 < sorted_cx2.size() && diff_cx2[i+1] < s * allow_subdiff) {
                    // ひとつ後ろとの差分
                    valid_diffx[x_sortidx2[i+1]] |= 2;
                    valid_diffx[x_sortidx2[i]] |= 1;
                }
            }

            int linecount = 0;
            for(int i = 0; i < x_sortidx2.size(); i++) {
                int idx = x_sortidx2[i];
                if(linecount == 0) {
                    if(valid_diffx[idx] == 1 && valid_diffy[idx] > 0) {
                        line_boxid.push_back(std::vector<int>());
                        line_boxid.back().push_back(chain[idx]);
                        linecount++;
                    }
                }
                else {
                    if(valid_diffx[idx] == 3 && valid_diffy[idx] == 3) {
                        line_boxid.back().push_back(chain[idx]);
                        linecount++;
                    }
                    else if(valid_diffx[idx] > 1 && valid_diffy[idx] > 1) {
                        line_boxid.back().push_back(chain[idx]);
                        linecount = 0;
                    }
                    else {
                        linecount = 0;
                    }
                }
            }

        }

        for(const auto &chain: line_boxid) {
            std::vector<int> tmpline;
            if(chain.size() < 2) continue;
            tmpline.push_back(chain[0]);
            for(int i = 0; i < chain.size() - 1; i++) {
                int x1 = boxes[chain[i]].cx + boxes[chain[i]].w / 2;
                int y1 = boxes[chain[i]].cy;
                int x2 = boxes[chain[i+1]].cx - boxes[chain[i+1]].w / 2;
                int y2 = boxes[chain[i+1]].cy;

                x1 /= 2;
                x2 /= 2;
                y1 /= 2;
                y2 /= 2;
                bool success = true;
                if(fabs(x2 - x1) > 0) {
                    double a = (y2 - y1) / (x2 - x1);
                    for(int x = x1; x < x2; x++) {
                        int y = a * (x - x1) + y1;
                        if(lineblocker[y * width + x]) {
                            success = false;
                            break;
                        }
                    }
                }
                if(success) {
                    tmpline.push_back(chain[i+1]);
                }
            }
            if(tmpline.size() > 1) {
                hori_line_boxid.push_back(tmpline);
            }
        }        
    }

    {
        // 垂直方向の文字列があるかも
        // cxがどこかに固まっている

        std::vector<float> sorted_cx;
        for(const auto i: y_sortidx) {
            sorted_cx.push_back(cx[i]);
        }
        std::vector<float> diff_cx(sorted_cx.size());
        std::adjacent_difference(sorted_cx.begin(), sorted_cx.end(), diff_cx.begin());

        std::vector<float> sorted_cy;
        for(const auto i: y_sortidx) {
            sorted_cy.push_back(cy[i]);
        }
        std::vector<float> diff_cy(sorted_cy.size());
        std::adjacent_difference(sorted_cy.begin(), sorted_cy.end(), diff_cy.begin());


        std::vector<int> valid_diffx(sorted_cx.size());
        for(int i = 0; i < y_sortidx.size(); i++) {
            int boxid = unbind_boxes[y_sortidx[i]];
            float s = std::max(boxes[boxid].w, boxes[boxid].h);
            if(i > 0 && fabs(diff_cx[i]) < s * allow_maindiff) {
                // ひとつ前との差分
                valid_diffx[y_sortidx[i-1]] |= 1;
                valid_diffx[y_sortidx[i]] |= 2;
            }
            if(i + 1 < sorted_cx.size() && fabs(diff_cx[i+1]) < s * allow_maindiff) {
                // ひとつ後ろとの差分
                valid_diffx[y_sortidx[i+1]] |= 2;
                valid_diffx[y_sortidx[i]] |= 1;
            }
        }

        std::vector<int> valid_diffy(sorted_cy.size());
        for(int i = 0; i < y_sortidx.size(); i++) {
            int boxid = unbind_boxes[y_sortidx[i]];
            float s = std::max(boxes[boxid].w, boxes[boxid].h);
            if(i > 0 && diff_cy[i] < s * allow_subdiff) {
                // ひとつ前との差分
                valid_diffy[y_sortidx[i-1]] |= 1;
                valid_diffy[y_sortidx[i]] |= 2;
            }
            if(i + 1 < sorted_cy.size() && diff_cy[i+1] < s * allow_subdiff) {
                // ひとつ後ろとの差分
                valid_diffy[y_sortidx[i+1]] |= 2;
                valid_diffy[y_sortidx[i]] |= 1;
            }
        }
        int linecount = 0;
        std::vector<std::vector<int>> line_boxid;
        for(int i = 0; i < y_sortidx.size(); i++) {
            int idx = y_sortidx[i];
            if(linecount == 0) {
                if(valid_diffy[idx] == 1 && valid_diffx[idx] > 0) {
                    line_boxid.push_back(std::vector<int>());
                    line_boxid.back().push_back(unbind_boxes[idx]);
                    linecount++;
                }
            }
            else {
                if(valid_diffy[idx] == 3 && valid_diffx[idx] == 3) {
                    line_boxid.back().push_back(unbind_boxes[idx]);
                    linecount++;
                }
                else if(valid_diffy[idx] > 1 && valid_diffx[idx] > 1) {
                    line_boxid.back().push_back(unbind_boxes[idx]);
                    linecount = 0;
                }
                else {
                    linecount = 0;
                }
            }
        }

        for(const auto &chain: line_boxid) {
            std::vector<int> tmpline;
            if(chain.size() < 2) continue;
            tmpline.push_back(chain[0]);
            for(int i = 0; i < chain.size() - 1; i++) {
                int x1 = boxes[chain[i]].cx;
                int y1 = boxes[chain[i]].cy + boxes[chain[i]].h / 2;
                int x2 = boxes[chain[i+1]].cx;
                int y2 = boxes[chain[i+1]].cy - boxes[chain[i+1]].h / 2;

                x1 /= 2;
                x2 /= 2;
                y1 /= 2;
                y2 /= 2;
                bool success = true;
                if(fabs(y2 - y1) > 0) {
                    double a = (x2 - x1) / (y2 - y1);
                    for(int y = y1; y < y2; y++) {
                        int x = a * (y - y1) + x1;
                        if(lineblocker[y * width + x]) {
                            success = false;
                            break;
                        }
                    }
                }
                if(success) {
                    tmpline.push_back(chain[i+1]);
                }
            }
            if(tmpline.size() > 1) {
                vert_line_boxid.push_back(tmpline);
            }
        }        
    }

    std::vector<int> line_priority1(unbind_boxes.size());
    std::vector<int> line_priority2(unbind_boxes.size());
    std::vector<int> line_id1(unbind_boxes.size());
    std::vector<int> line_id2(unbind_boxes.size());
    std::vector<int> invalid_line;
    int line_idcount = 0;
    for(const auto &chain: hori_line_boxid) {
        for(const auto boxid: chain) {
            int idx = std::distance(unbind_boxes.begin(), std::find(unbind_boxes.begin(), unbind_boxes.end(), boxid));
            line_priority1[idx] = chain.size();
            line_id1[idx] = line_idcount;
        }
        line_idcount++;
    }
    for(const auto &chain: vert_line_boxid) {
        for(const auto boxid: chain) {
            int idx = std::distance(unbind_boxes.begin(), std::find(unbind_boxes.begin(), unbind_boxes.end(), boxid));
            line_priority2[idx] = chain.size();
            line_id2[idx] = line_idcount;
        }
        line_idcount++;
    }

    for(int i = 0; i < unbind_boxes.size(); i++) {
        if(line_priority1[i] > 0 && line_priority2[i] > 0) {
            // 縦と横の両方に参加しているものがいる
            if(line_priority1[i] >= line_priority2[i]) {
                // 水平を残す
                invalid_line.push_back(line_id2[i]);
            }
            else {
                invalid_line.push_back(line_id1[i]);
            }
        }
    }

    line_idcount = -1;
    for(const auto &chain: hori_line_boxid) {
        line_idcount++;
        if(std::find(invalid_line.begin(), invalid_line.end(), line_idcount) != invalid_line.end()) continue;
        int idx = next_id++;
        int subidx = 0;
        for(const auto boxid: chain) {
            boxes[boxid].idx = idx;
            boxes[boxid].subidx = subidx++;
            boxes[boxid].subtype &= ~1;
            boxes[boxid].direction = 0;
        }
    }
    for(const auto &chain: vert_line_boxid) {
        line_idcount++;
        if(std::find(invalid_line.begin(), invalid_line.end(), line_idcount) != invalid_line.end()) continue;
        int idx = next_id++;
        int subidx = 0;
        for(const auto boxid: chain) {
            boxes[boxid].idx = idx;
            boxes[boxid].subidx = subidx++;
            boxes[boxid].subtype |= 1;
            boxes[boxid].direction = M_PI_2;
        }
    }
    int direction = 0;
    for(const auto box: boxes) {
        if(box.idx < 0) continue;
        if((box.subtype & 1) == 0) {
            direction++;
        }
        else {
            direction--;
        }
    }

    for(const auto boxid: unbind_boxes) {
        if(boxes[boxid].idx < 0) {
            boxes[boxid].idx = next_id++;
            boxes[boxid].subidx = 0;
            if (direction >= 0) {
                boxes[boxid].subtype &= ~1;
                boxes[boxid].direction = 0;
            }
            else {
                boxes[boxid].subtype |= 1;
                boxes[boxid].direction = M_PI_2;
            }
        }
    }
    return next_id;
}

void search_ruby(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage)
{
    std::cerr << "search_ruby" << std::endl;

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() < 2) continue;

        // chain内をソート
        sort_chain(line_box_chain[chainid], boxes);

        std::vector<int> x;
        std::vector<int> y;
        float direction;
        double w, h;
        make_track_line(x,y, direction, w, h, boxes, line_box_chain, chainid);

        for(const auto &boxid: line_box_chain[chainid]) {
            boxes[boxid].direction = direction;
        }

        if(fabs(direction) < M_PI_4) {
            // 横書き

            // ルビの所属chainを検索
            std::vector<int> ruby_boxid;
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / 2;
                int yi = y[i] / 2;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int k = 1; k <= 10; k++) {
                    int yp = yi - h / 2 * k / 10;
                    if(yp < 0 || yp >= height) continue;
                    if(lineblocker[yp * width + xi]) continue;
                    int other_id = idimage[yp * width + xi];
                    if(other_id < 0) continue;
                    if((boxes[other_id].subtype & (2+4)) != 2+4) continue;
                    if(std::find(ruby_boxid.begin(), ruby_boxid.end(), other_id) != ruby_boxid.end()) continue;
                    ruby_boxid.push_back(other_id);
                }
            }

            // 登録済みの文字は外す
            for(const auto id: ruby_boxid) {
                if(boxes[id].idx >= 0) {
                    std::erase(line_box_chain[boxes[id].idx], id);
                }
                else {
                    boxes[id].idx = chainid;
                    boxes[id].direction = direction;
                }
                line_box_chain[chainid].push_back(id);
            }

            sort_chain(line_box_chain[chainid], boxes);
        }
        else {
            // 縦書き

            // ルビの所属chainを検索
            std::vector<int> ruby_boxid;
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / 2;
                int yi = y[i] / 2;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int k = 1; k <= 10; k++) {
                    int xp = xi + w / 2 * k / 10;
                    if(xp < 0 || xp >= width) continue;
                    if(lineblocker[yi * width + xp]) continue;
                    int other_id = idimage[yi * width + xp];
                    if(other_id < 0) continue;
                    if((boxes[other_id].subtype & (2+4)) != 2+4) continue;
                    if(std::find(ruby_boxid.begin(), ruby_boxid.end(), other_id) != ruby_boxid.end()) continue;
                    ruby_boxid.push_back(other_id);
                }
            }

            // 登録済みの文字は外す
            for(const auto id: ruby_boxid) {
                if(boxes[id].idx >= 0) {
                    std::erase(line_box_chain[boxes[id].idx], id);
                }
                else {
                    boxes[id].idx = chainid;
                    boxes[id].direction = direction;
                }
                line_box_chain[chainid].push_back(id);
            }

            sort_chain(line_box_chain[chainid], boxes);
        }
    }

    for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        if (it->size() < 2) {
            it = line_box_chain.erase(it);
        }
        else {
            ++it;
        }
    }

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(fabs(boxes[line_box_chain[chainid].front()].direction) < M_PI_4) {
            // 横書き
            sort_chain(line_box_chain[chainid], boxes);

            double w1 = std::transform_reduce(
                line_box_chain[chainid].begin(), 
                line_box_chain[chainid].end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].w; });

            // ルビとそれ以外で分ける
            std::vector<int> baseid;
            std::vector<int> rubyid;
            for(auto boxid: line_box_chain[chainid]) {
                if((boxes[boxid].subtype & (2+4)) == 2+4){
                    rubyid.push_back(boxid);
                }
                else {
                    baseid.push_back(boxid);
                }
            }

            // rubybaseの直後にrubyを挿入する
            std::vector<int> fixlist;
            int lastbase = -1;
            float startx = -1;
            float endx = -1;
            for(int i = 0; i < baseid.size(); i++) {
                int boxid = baseid[i];
                if((boxes[boxid].subtype & (2+4)) == 2) {
                    fixlist.push_back(boxid);
                    if(lastbase < 0) {
                        startx = boxes[boxid].cx - w1;
                    }
                    endx = boxes[boxid].cx + w1;
                    lastbase = i;
                }
                else {
                    if(lastbase >= 0) {
                        for(auto bidx: rubyid) {
                            if(boxes[bidx].cx > startx && boxes[bidx].cx < endx) {
                                fixlist.push_back(bidx);
                            }
                        }
                        auto result = std::remove_if(rubyid.begin(), rubyid.end(), [fixlist](int x) { return std::find(fixlist.begin(), fixlist.end(), x) != fixlist.end(); });
                        rubyid.erase(result, rubyid.end());
                    }
                    fixlist.push_back(boxid);
                    lastbase = -1;
                }
            }
            if(lastbase >= 0) {
                for(auto bidx: rubyid) {
                    if(boxes[bidx].cx > startx && boxes[bidx].cx < endx) {
                        fixlist.push_back(bidx);
                    }
                }
                auto result = std::remove_if(rubyid.begin(), rubyid.end(), [fixlist](int x) { return std::find(fixlist.begin(), fixlist.end(), x) != fixlist.end(); });
                rubyid.erase(result, rubyid.end());
            }

            // 親を見つけられなかったルビ
            for(auto bidx: rubyid) {
                float x = boxes[bidx].cx;
                boxes[bidx].subtype &= ~(2+4);
                for(auto it = fixlist.begin(); it != fixlist.end(); ++it) {
                    if((boxes[*it].subtype & (2+4)) == 2+4) continue;
                    float before = boxes[*it].cx - boxes[*it].w / 2;
                    float after = boxes[fixlist.front()].cx - boxes[fixlist.front()].w / 2;
                    for(auto it2 = it; it != fixlist.begin(); --it2) {
                        if((boxes[*it2].subtype & (2+4)) == 2+4) continue;
                        after = boxes[*it2].cx + boxes[*it2].w / 2;
                        break;
                    }
                    if(after < x && x < before) {
                        ++it;
                        it = fixlist.insert(it, bidx);
                        goto find1;
                    }
                }
                fixlist.push_back(bidx);
                find1:
                ;
            }

            int subidx = 0;
            for(auto boxid: fixlist) {
                boxes[boxid].subtype &= ~1;
                boxes[boxid].idx = chainid;
                boxes[boxid].subidx = subidx++;
            }
        }
        else {
            // 縦書き
            sort_chain(line_box_chain[chainid], boxes);

            double h1 = std::transform_reduce(
                line_box_chain[chainid].begin(), 
                line_box_chain[chainid].end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].h; });

            // ルビとそれ以外で分ける
            std::vector<int> baseid;
            std::vector<int> rubyid;
            for(auto boxid: line_box_chain[chainid]) {
                if((boxes[boxid].subtype & (2+4)) == 2+4){
                    rubyid.push_back(boxid);
                }
                else {
                    baseid.push_back(boxid);
                }
            }

            // rubybaseの直後にrubyを挿入する
            std::vector<int> fixlist;
            int lastbase = -1;
            float starty = -1;
            float endy = -1;
            for(int i = 0; i < baseid.size(); i++) {
                int boxid = baseid[i];
                if((boxes[boxid].subtype & (2+4)) == 2) {
                    fixlist.push_back(boxid);
                    if(lastbase < 0) {
                        starty = boxes[boxid].cy - h1;
                    }
                    endy = boxes[boxid].cy + h1;
                    lastbase = i;
                }
                else {
                    if(lastbase >= 0) {
                        for(auto bidx: rubyid) {
                            if(boxes[bidx].cy > starty && boxes[bidx].cy < endy) {
                                fixlist.push_back(bidx);
                            }
                        }
                        auto result = std::remove_if(rubyid.begin(), rubyid.end(), [fixlist](int x) { return std::find(fixlist.begin(), fixlist.end(), x) != fixlist.end(); });
                        rubyid.erase(result, rubyid.end());
                    }
                    fixlist.push_back(boxid);
                    lastbase = -1;
                }
            }
            if(lastbase >= 0) {
                for(auto bidx: rubyid) {
                    if(boxes[bidx].cy > starty && boxes[bidx].cy < endy) {
                        fixlist.push_back(bidx);
                    }
                }
                auto result = std::remove_if(rubyid.begin(), rubyid.end(), [fixlist](int x) { return std::find(fixlist.begin(), fixlist.end(), x) != fixlist.end(); });
                rubyid.erase(result, rubyid.end());
            }

            // 親を見つけられなかったルビ
            for(auto bidx: rubyid) {
                float y = boxes[bidx].cy;
                boxes[bidx].subtype &= ~(2+4);
                for(auto it = fixlist.begin(); it != fixlist.end(); ++it) {
                    float before = boxes[*it].cy - boxes[*it].h / 2;
                    float after = boxes[fixlist.front()].cy - boxes[fixlist.front()].h / 2;
                    for(auto it2 = it; it != fixlist.begin(); --it2) {
                        if((boxes[*it2].subtype & (2+4)) == 2+4) continue;
                        after = boxes[*it2].cy + boxes[*it2].h / 2;
                        break;
                    }
                    if(after < y && y < before) {
                        ++it;
                        it = fixlist.insert(it, bidx);
                        goto find2;
                    }
                }
                fixlist.push_back(bidx);
                find2:
                ;
            }

            int subidx = 0;
            for(auto boxid: fixlist) {
                boxes[boxid].subtype |= 1;
                boxes[boxid].idx = chainid;
                boxes[boxid].subidx = subidx++;
            }
        }
    }
}

void chain_space(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
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

    auto chainid_map = create_chainid_map(boxes, line_box_chain);
    std::vector<int> chain_cont(line_box_chain.size(), -1);
    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() == 0) continue;
        int last_id = line_box_chain[chainid].back();
        if((boxes[last_id].subtype & 8) != 8) continue;

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
                ave_dist = boxes[last_id].w;
            }
            else {
                // 縦書き
                ave_dist = boxes[last_id].h;
            }            
        }

        if(fabs(direction) < M_PI_4) {
            // 横書き
            int y0 = boxes[last_id].cy;
            int h = boxes[last_id].h;
            for(int x = boxes[last_id].cx + boxes[last_id].w / 2; x < boxes[last_id].cx + boxes[last_id].w / 2 + ave_dist * 3; x+=2) {
                int ix = x / 2;
                if(ix < 0 || ix >= width) continue;
                for(int y = y0 - h/2; y < y0 + h/2; y+=2) {
                    int iy = y / 2;
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
                    }
                    if(chain_cont[chainid] < 0) {
                        chain_cont[chainid] = other_chain;
                    }
                    else if(chain_cont[chainid] != other_chain) {
                        chain_cont[chainid] = -1;
                        goto find1;
                    }
                }
            }
            find1:
            ;
        }
        else {
            // 縦書き
            int x0 = boxes[last_id].cx;
            int w = boxes[last_id].w;
            for(int y = boxes[last_id].cy + boxes[last_id].h / 2; y < boxes[last_id].cy + boxes[last_id].h / 2 + ave_dist * 3; y+=2) {
                int iy = y / 2;
                if(iy < 0 || iy >= height) continue;

                for(int x = x0 - w/2; x < x0 + w/2; x+=2) {
                    int ix = x / 2;
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
                    }
                    if(chain_cont[chainid] < 0) {
                        chain_cont[chainid] = other_chain;
                    }
                    else if(chain_cont[chainid] != other_chain) {
                        chain_cont[chainid] = -1;
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
        while(chain_cont[root] >= 0 && std::find(chain.begin(), chain.end(), root) == chain.end()) {
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

    for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        if (it->size() < 2) {
            it = line_box_chain.erase(it);
        }
        else {
            ++it;
        }
    }
    fix_chain_info(boxes, line_box_chain);
}

void print_chaininfo(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain)
{
    fprintf(stderr, "print_chaininfo\n");
    fprintf(stderr, "****************\n");
    for(int i = 0; i < line_box_chain.size(); i++) {
        fprintf(stderr, " chain %d len %lu\n", i, line_box_chain[i].size());
        int boxid1 = line_box_chain[i].front();
        int boxid2 = line_box_chain[i].back();

        fprintf(stderr, "  %f %d x %f y %f w %f h %f - %d x %f y %f w %f h %f\n",
            boxes[boxid1].direction / M_PI * 180, 
            boxid1, boxes[boxid1].cx, boxes[boxid1].cy, boxes[boxid1].w, boxes[boxid1].h,
            boxid2, boxes[boxid2].cx, boxes[boxid2].cy, boxes[boxid2].w, boxes[boxid2].h);

        std::copy(line_box_chain[i].begin(), line_box_chain[i].end(), std::ostream_iterator<int>(std::cerr, ","));
        std::cerr << std::endl;
        fprintf(stderr, "=================\n");
        for(int j = 0; j < line_box_chain[i].size(); j++) {
            int boxid = line_box_chain[i][j];
            fprintf(stderr, "    %d %d %d, %d %d %d, %f x %f y %f w %f h %f\n",
                i, j, boxid,
                boxes[boxid].idx, boxes[boxid].subidx, boxes[boxid].subtype, 
                boxes[boxid].direction / M_PI * 180, 
                boxes[boxid].cx, boxes[boxid].cy, boxes[boxid].w, boxes[boxid].h);
        }
    }
    fprintf(stderr, "****************\n");
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

    chain_space(boxes, line_box_chain, sepimage, idimage);

    fprintf(stderr, "loop done\n");

    //print_chaininfo(boxes, line_box_chain);

    for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        if (it->size() < 2) {
            it = line_box_chain.erase(it);
        }
        else {
            ++it;
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

    // 短いチェーンは方向を修正しておく
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

    // chain id を登録する
    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        for(auto boxid: line_box_chain[chainid]) {
            boxes[boxid].idx = chainid;
            boxes[boxid].subtype |= (fabs(boxes[boxid].direction) < M_PI_4) ? 0: 1;
        }
    }

    // ルビの検索
    search_ruby(boxes, line_box_chain, lineblocker, idimage);

    // 飛んでる番号があるので振り直す
    int id_max = 0;
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

        id_max = number_unbind(boxes, lineblocker, idimage, chain_remap.size());
        std::cerr << "id max " << id_max << std::endl;
    }

    // for(const auto &box: boxes) {
    //     fprintf(stderr, "box %d cx %f cy %f w %f h %f c1 %f c2 %f c4 %f c8 %f\n", box.id, box.cx, box.cy, box.w, box.h, box.code1, box.code2, box.code4, box.code8);
    //     fprintf(stderr, " block %d idx %d subidx %d subtype %d dir %f\n", box.block, box.idx, box.subidx, box.subtype, box.direction);
    // }

    struct lineparam {
        int d;
        float cx;
        float cy;
        float allow_space;
        float size;
        int doubleline;
        int count;
    };

    // 行の順番に番号を振り直す
    {
        std::cerr << "renumber id" << std::endl;
        std::vector<int> chain_remap(id_max);
        std::vector<lineparam> lineparams(id_max);
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            if((box.subtype & (2+4)) == 2+4) continue;
            if((box.subtype & 1) == 0) {
                // 横書き
                lineparams[box.idx].d = 0;
                if(lineparams[box.idx].count == 0 || box.cx < lineparams[box.idx].cx) {
                    lineparams[box.idx].cx = box.cx;
                    lineparams[box.idx].cy = box.cy;
                }
                lineparams[box.idx].allow_space = std::max(lineparams[box.idx].allow_space, box.h);
            }
            else {
                // 縦書き
                lineparams[box.idx].d = 1;
                if(lineparams[box.idx].count == 0 || box.cy < lineparams[box.idx].cy) {
                    lineparams[box.idx].cx = box.cx;
                    lineparams[box.idx].cy = box.cy;
                }
                lineparams[box.idx].allow_space = std::max(lineparams[box.idx].allow_space, box.w);
            }
            lineparams[box.idx].count++;
        }

        std::iota(chain_remap.begin(), chain_remap.end(), 0);
        std::sort(chain_remap.begin(), chain_remap.end(), [&](const auto a, const auto b){
            if(lineparams[a].d != lineparams[b].d)
                return lineparams[a].d < lineparams[b].d;
            if(lineparams[a].d == 0) {
                // 横書き
                if(fabs(lineparams[a].cy - lineparams[b].cy) < std::max(lineparams[a].allow_space, lineparams[b].allow_space)) {
                    return lineparams[a].cx < lineparams[b].cx;
                }
                return lineparams[a].cy < lineparams[b].cy;
            }
            else {
                // 縦書き
                if(fabs(lineparams[a].cx - lineparams[b].cx) < std::max(lineparams[a].allow_space, lineparams[b].allow_space)) {
                    return lineparams[a].cy < lineparams[b].cy;
                }
                return lineparams[a].cx > lineparams[b].cx;
            }
        });
        for(auto &box: boxes) {
            if(box.idx < 0) continue;
            int id = std::distance(chain_remap.begin(), std::find(chain_remap.begin(), chain_remap.end(), box.idx));
            box.idx = id;
        }
    }

    // ブロックの形成
    std::vector<std::vector<int>> chain_next(id_max, std::vector<int>());
    std::vector<std::vector<int>> chain_prev(id_max, std::vector<int>());
    {
        std::cerr << "make block" << std::endl;
        line_box_chain.clear();
        line_box_chain.resize(id_max);
        std::vector<lineparam> lineparams(id_max);
        for(const auto &box: boxes) {
            if(box.idx < 0) continue;
            if((box.subtype & (2+4)) == 2+4) continue;
            line_box_chain[box.idx].push_back(box.id);
            lineparams[box.idx].size = std::max(lineparams[box.idx].size, std::max(box.w, box.h));
            if((box.subtype & 1) == 0) {
                // 横書き
                if(line_box_chain[box.idx].size() > 2) {
                    lineparams[box.idx].d = 2;
                }
                else {
                    lineparams[box.idx].d = 0;
                }
            }
            else {
                // 縦書き
                lineparams[box.idx].d = 1;
            }
        }
        for(auto &chain: line_box_chain) {
            int count0 = 0;
            int count1 = 0;
            int count2 = 0;
            int chainid = -1;
            if (chain.size() == 0) continue;
            for(auto boxid: chain) {
                chainid = boxes[boxid].idx;
                if(boxes[boxid].double_line == 0) {
                    count0++;
                }
                if(boxes[boxid].double_line == 1) {
                    count1++;
                }
                if(boxes[boxid].double_line == 2) {
                    count2++;
                }
            }
            if(count1 > count0 && count1 > count2) {
                lineparams[chainid].doubleline = 1;
            }
            if(count2 > count0 && count2 > count1) {
                lineparams[chainid].doubleline = 2;
            }
            sort_chain(chain, boxes);
        }

        std::vector<int> chainid_map = create_chainid_map(boxes, line_box_chain, 2.0);

        // {
        //     std::ofstream outfile("chainidmap.txt");
        //     for(int y = 0; y < height; y++){
        //         for(int x = 0; x < width; x++){
        //             outfile << chainid_map[y * width + x] << " ";
        //         }
        //         outfile << std::endl;
        //     }
        // }

        for(int chainid = 0; chainid < id_max; chainid++) {
            if(line_box_chain[chainid].size() < 3) continue;        
            if(lineparams[chainid].doubleline == 0) continue;
            //std::cerr << "chain " << chainid << std::endl;

            std::vector<int> x;
            std::vector<int> y;
            float direction;
            double w, h;
            make_track_line(x,y, direction, w, h, boxes, line_box_chain, chainid);

            if(x.size() == 0 || y.size() == 0) continue;

            if(lineparams[chainid].doubleline == 1) {
                if(lineparams[chainid].d == 2) {
                    // 横書き
                    for(int i = 0; i < x.size(); i++) {
                        int xi = x[i] / 2;
                        int yi = y[i] / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) continue;
                        for(int yp = yi + h / 4; yp < yi + h / 4 + h / 2 * 2; yp++) {
                            if(yp < 0 || yp >= height) continue;
                            if(lineblocker[yp * width + xi]) continue;
                            //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * 2, yp * 2);
                            int other_chain = chainid_map[yp * width + xi];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 1) continue;
                            if(lineparams[other_chain].doubleline != 2) continue;
                            if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;

                            if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                                //fprintf(stderr, "hori %d -> %d\n", chainid, other_chain);
        
                                chain_next[chainid].push_back(other_chain);
                                chain_prev[other_chain].push_back(chainid);
                            }
                        }
                    }

                    for(int xi = (x.front() - lineparams[chainid].size/2) / 2; xi > (x.front() - lineparams[chainid].size * 3) / 2; xi--) {
                        int yi = y.front() / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) break;
                        for(int yp = yi - h / 2; yp < yi + h / 2; yp++) {
                            if(yp < 0 || yp >= height) continue;
                            if(lineblocker[yp * width + xi]) continue;
                            //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * 2, yp * 2);
                            int other_chain = chainid_map[yp * width + xi];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 1) continue;
                            if(lineparams[other_chain].doubleline > 0) continue;

                            if(std::find(chain_prev[chainid].begin(), chain_prev[chainid].end(), other_chain) == chain_prev[chainid].end()) {
                                //fprintf(stderr, "hori %d <- %d\n", chainid, other_chain);
        
                                chain_prev[chainid].push_back(other_chain);
                                chain_next[other_chain].push_back(chainid);
                            }
                        }
                    }
                }
                else if (lineparams[chainid].d == 1){
                    // 縦書き
                    for(int i = 0; i < x.size(); i++) {
                        int xi = x[i] / 2;
                        int yi = y[i] / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) continue;
                        for(int xp = xi - w / 4; xp > xi - w / 4 - w / 2 * 2; xp--) {
                            if(xp < 0 || xp >= width) continue;
                            if(lineblocker[yi * width + xp]) continue;
                            //fprintf(stderr, "vert %d x %d y %d\n", chainid, xp * 2, yi * 2);
                            int other_chain = chainid_map[yi * width + xp];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 2) continue;
                            if(lineparams[other_chain].doubleline != 2) continue;
                            if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;

                            if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                                //fprintf(stderr, "vert %d -> %d\n", chainid, other_chain);

                                chain_next[chainid].push_back(other_chain);
                                chain_prev[other_chain].push_back(chainid);
                            }
                        }
                    }

                    for(int yi = (y.front() - lineparams[chainid].size/2) / 2; yi > (y.front() - lineparams[chainid].size * 3) / 2; yi--) {
                        int xi = x.front() / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) break;
                        for(int xp = xi - h / 2; xp < xi + h / 2; xp++) {
                            if(xp < 0 || xp >= width) continue;
                            if(lineblocker[yi * width + xp]) continue;
                            int other_chain = chainid_map[yi * width + xp];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 2) continue;
                            if(lineparams[other_chain].doubleline > 0) continue;

                            if(std::find(chain_prev[chainid].begin(), chain_prev[chainid].end(), other_chain) == chain_prev[chainid].end()) {
                                //fprintf(stderr, "vert %d <- %d\n", chainid, other_chain);
        
                                chain_prev[chainid].push_back(other_chain);
                                chain_next[other_chain].push_back(chainid);
                            }
                        }
                    }
                }
            }
            else if(lineparams[chainid].doubleline == 2) {
                if(lineparams[chainid].d == 2) {
                    // 横書き
                    for(int i = 0; i < x.size(); i++) {
                        int xi = x[i] / 2;
                        int yi = y[i] / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) continue;
                        for(int yp = yi - h / 4; yp > yi - h / 4 - h / 2 * 2; yp--) {
                            if(yp < 0 || yp >= height) continue;
                            if(lineblocker[yp * width + xi]) continue;
                            //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * 2, yp * 2);
                            int other_chain = chainid_map[yp * width + xi];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 1) continue;
                            if(lineparams[other_chain].doubleline != 1) continue;
                            if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;

                            if(std::find(chain_prev[chainid].begin(), chain_prev[chainid].end(), other_chain) == chain_prev[chainid].end()) {
                                //fprintf(stderr, "hori %d <- %d\n", chainid, other_chain);
        
                                chain_prev[chainid].push_back(other_chain);
                                chain_next[other_chain].push_back(chainid);
                            }
                        }
                    }

                    for(int xi = (x.back() + lineparams[chainid].size/2) / 2; xi < (x.back() + lineparams[chainid].size * 3) / 2; xi++) {
                        int yi = y.back() / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) break;
                        for(int yp = yi - h / 2; yp < yi + h / 2; yp++) {
                            if(yp < 0 || yp >= height) continue;
                            if(lineblocker[yp * width + xi]) continue;
                            //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * 2, yp * 2);
                            int other_chain = chainid_map[yp * width + xi];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 1) continue;
                            if(lineparams[other_chain].doubleline > 0) continue;

                            if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                                //fprintf(stderr, "hori %d -> %d\n", chainid, other_chain);
        
                                chain_next[chainid].push_back(other_chain);
                                chain_prev[other_chain].push_back(chainid);
                            }
                        }
                    }
                }
                else if (lineparams[chainid].d == 1){
                    // 縦書き
                    for(int i = 0; i < x.size(); i++) {
                        int xi = x[i] / 2;
                        int yi = y[i] / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) continue;
                        for(int xp = xi + w / 4; xp < xi + w / 4 + w / 2 * 2; xp++) {
                            if(xp < 0 || xp >= width) continue;
                            if(lineblocker[yi * width + xp]) continue;
                            //fprintf(stderr, "vert %d x %d y %d\n", chainid, xp * 2, yi * 2);
                            int other_chain = chainid_map[yi * width + xp];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 2) continue;
                            if(lineparams[other_chain].doubleline != 1) continue;
                            if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;

                            if(std::find(chain_prev[chainid].begin(), chain_prev[chainid].end(), other_chain) == chain_prev[chainid].end()) {
                                //fprintf(stderr, "vert %d <- %d\n", chainid, other_chain);

                                chain_prev[chainid].push_back(other_chain);
                                chain_next[other_chain].push_back(chainid);
                            }
                        }
                    }

                    for(int yi = (y.back() + lineparams[chainid].size/2) / 2; yi < (y.back() + lineparams[chainid].size * 3) / 2; yi++) {
                        int xi = x.back() / 2;
                        if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                        if(lineblocker[yi * width + xi]) break;
                        for(int xp = xi - h / 2; xp < xi + h / 2; xp++) {
                            if(xp < 0 || xp >= width) continue;
                            if(lineblocker[yi * width + xp]) continue;
                            int other_chain = chainid_map[yi * width + xp];
                            if(other_chain < 0) continue;
                            if(other_chain == chainid) continue;
                            if(lineparams[other_chain].d == 2) continue;
                            if(lineparams[other_chain].doubleline > 0) continue;

                            if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                                //fprintf(stderr, "vert %d -> %d\n", chainid, other_chain);
        
                                chain_next[chainid].push_back(other_chain);
                                chain_prev[other_chain].push_back(chainid);
                            }
                        }
                    }
                }
            }
        }

        for(int chainid = 0; chainid < id_max; chainid++) {
            //std::cerr << "chain " << chainid << std::endl;
            if(line_box_chain[chainid].size() < 2) continue;

            if(lineparams[chainid].doubleline > 0) continue;
            //if(chain_next[chainid].size() > 0) continue;

            std::vector<int> x;
            std::vector<int> y;
            float direction;
            double w, h;
            make_track_line(x,y, direction, w, h, boxes, line_box_chain, chainid);

            if(lineparams[chainid].d == 2) {
                // 横書き
                for(int i = 0; i < x.size(); i++) {
                    int xi = x[i] / 2;
                    int yi = y[i] / 2;
                    if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                    if(lineblocker[yi * width + xi]) continue;
                    for(int yp = yi + h / 4; yp < yi + h / 4 + h / 2 * 2; yp++) {
                        if(yp < 0 || yp >= height) continue;
                        //fprintf(stderr, "hori %d x %d y %d\n", chainid, xi * 2, yp * 2);
                        if(lineblocker[yp * width + xi]) break;
                        int other_chain = chainid_map[yp * width + xi];
                        if(other_chain < 0) continue;
                        if(other_chain == chainid) continue;
                        if(lineparams[other_chain].d == 1) continue;
                        if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;

                        if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                            //fprintf(stderr, "hori %d -> %d\n", chainid, other_chain);
    
                            chain_next[chainid].push_back(other_chain);
                            chain_prev[other_chain].push_back(chainid);
                        }
                    }
                }
            }
            else if (lineparams[chainid].d == 1){
                // 縦書き
                for(int i = 0; i < x.size(); i++) {
                    int xi = x[i] / 2;
                    int yi = y[i] / 2;
                    if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                    if(lineblocker[yi * width + xi]) continue;
                    for(int xp = xi - w / 4; xp > xi - w / 4 - w / 2 * 2; xp--) {
                        if(xp < 0 || xp >= width) continue;
                        //fprintf(stderr, "vert %d x %d y %d\n", chainid, xp * 2, yi * 2);
                        if(lineblocker[yi * width + xp]) break;
                        int other_chain = chainid_map[yi * width + xp];
                        if(other_chain < 0) continue;
                        if(other_chain == chainid) continue;
                        if(lineparams[other_chain].d == 2) continue;
                        if(fabs(lineparams[other_chain].size - lineparams[chainid].size) / std::max(lineparams[chainid].size, lineparams[other_chain].size) > 0.5) continue;
                        
                        if(std::find(chain_next[chainid].begin(), chain_next[chainid].end(), other_chain) == chain_next[chainid].end()) {
                            //fprintf(stderr, "vert %d -> %d\n", chainid, other_chain);

                            chain_next[chainid].push_back(other_chain);
                            chain_prev[other_chain].push_back(chainid);
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
    std::vector<std::vector<int>> block_chain;
    {
        std::cerr << "block chain check" << std::endl;
        std::vector<int> chain_root;
        for(int cur_id = 0; cur_id < id_max; cur_id++){
            if(chain_prev[cur_id].size() == 0) {
                chain_root.push_back(cur_id);
            }
            else {
                std::vector<int> stack;
                for(const auto id: chain_prev[cur_id]) {
                    stack.push_back(id);
                }
                std::vector<int> tmp_ids;
                while(stack.size() > 0) {
                    auto j = stack.back();
                    stack.pop_back();

                    if(chain_prev[j].size() == 0) {
                         goto next_loop;
                    }
                    if(std::find(chain_root.begin(), chain_root.end(), j) != chain_root.end()) {
                        goto next_loop;
                    }

                    if(std::find(tmp_ids.begin(), tmp_ids.end(), j) != tmp_ids.end()) continue;
                    tmp_ids.push_back(j);
                    for(const auto id: chain_prev[j]) {
                        stack.push_back(id);
                    }
                }
                chain_root.push_back(cur_id);
            }
        next_loop:
            ;
        }


        for(auto cur_id: chain_root) {
            std::vector<int> stack;
            stack.push_back(cur_id);
            std::vector<int> tmp_block;
            while(stack.size() > 0) {
                auto j = stack.back();
                stack.pop_back();

                if(std::find(tmp_block.begin(), tmp_block.end(), j) != tmp_block.end()) continue;

                tmp_block.push_back(j);
                for(const auto chainid: chain_next[j]) {
                    stack.push_back(chainid);
                }
            }
            std::sort(tmp_block.begin(), tmp_block.end());
            tmp_block.erase(std::unique(tmp_block.begin(), tmp_block.end()), tmp_block.end());
            block_chain.push_back(tmp_block);
        }
    }

    std::vector<int> block_idx(block_chain.size());
    {
        struct blockparam {
            int d;
            float x_min;
            float x_max;
            float y_min;
            float y_max;
        };
        std::vector<blockparam> blockparams(block_chain.size());
        for(auto &p: blockparams) {
            p.x_min = width * 2;
            p.y_min = height * 2;
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
            int block = blockid_of_chain[box.idx];
            blockparams[block].d = ((box.subtype & 1) == 0) ? 0: 1;
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

        std::iota(block_idx.begin(), block_idx.end(), 0);
        std::sort(block_idx.begin(), block_idx.end(), [blockparams](const int a, const int b){
            if(blockparams[a].d == blockparams[b].d) {
                if(blockparams[a].d != 1) {
                    if(blockparams[a].x_max < blockparams[b].x_min) return true;
                    if(blockparams[b].x_max < blockparams[a].x_min) return false;

                    if(blockparams[a].y_max < blockparams[b].y_min) return true;
                    if(blockparams[b].y_max < blockparams[a].y_min) return false;

                    if(blockparams[a].y_min == blockparams[b].y_min) return a < b;

                    return blockparams[a].y_min < blockparams[b].y_min;
                }
                else {
                    if(blockparams[a].x_min > blockparams[b].x_max) return true;
                    if(blockparams[b].x_min > blockparams[a].x_max) return false;

                    if(blockparams[a].y_max < blockparams[b].y_min) return true;
                    if(blockparams[b].y_max < blockparams[a].y_min) return false;

                    if(blockparams[a].x_max == blockparams[b].x_max) return a < b;

                    return blockparams[a].x_max > blockparams[b].x_max;
                }
            }
            return blockparams[a].d < blockparams[b].d;
        });
    }

    // idを振る
    {
        std::cerr << "id renumber " << block_idx.size() << std::endl;
        std::vector<int> chain_remap(id_max);
        std::fill(chain_remap.begin(), chain_remap.end(), -1);
        for(int i = 0; i < block_idx.size(); i++) {
            for(const auto chainid: block_chain[block_idx[i]]) {
                chain_remap[chainid] = i;
            }
        }
        for(auto &box: boxes) {
            box.block = chain_remap[box.idx];
        }
    }

    // idxを修正
    {
        std::cerr << "fix idx" << std::endl;
        std::vector<std::vector<int>> idx_in_block(block_idx.size());
        for(const auto &box: boxes) {
            idx_in_block[box.block].push_back(box.idx);
        }
        for(auto &list: idx_in_block) {
            if(list.size() < 2) continue;
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        for(auto &box: boxes) {
            auto it = std::find(idx_in_block[box.block].begin(), idx_in_block[box.block].end(), box.idx);
            box.idx = std::distance(idx_in_block[box.block].begin(), it);
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

    // for(auto &box: boxes) {
    //     fprintf(stderr, "%d %d %d %d\n", box.id, box.block, box.idx, box.subidx);
    // }

    fprintf(stderr, "after_search done\n");
}

void prepare_id_image(
    std::vector<int> &idimage,
    std::vector<int> &idimage_main,
    std::vector<charbox> &boxes)
{
    idimage.resize(width*height, -1);
    idimage_main.resize(width*height, -1);
    for(const auto &box: boxes) {
        //fprintf(stderr, "box %d cx %f cy %f w %f h %f c1 %f c2 %f c4 %f c8 %f\n", box.id, box.cx, box.cy, box.w, box.h, box.code1, box.code2, box.code4, box.code8);
        int left = (box.cx - box.w / 2) / 2;
        int right = (box.cx + box.w / 2) / 2;
        int top = (box.cy - box.h / 2) / 2;
        int bottom = (box.cy + box.h / 2) / 2;
        if(left < 0 || right >= width) continue;
        if(top < 0 || bottom >= height) continue;
        if((box.subtype & (2+4)) != 2+4) {
            for(int y = top; y < bottom; y++) {
                for(int x = left; x < right; x++) {
                    idimage_main[y * width + x] = box.id;
                }
            }
        }
        for(int y = top; y < bottom; y++) {
            for(int x = left; x < right; x++) {
                idimage[y * width + x] = box.id;
            }
        }
    }

    // {
    //     std::ofstream outfile("idimage.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << idimage[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }
}

int search_connection(const std::vector<bool> &immap, std::vector<int> &idmap)
{
    idmap.resize(width*height, -1);
    std::vector<bool> visitmap(width*height, false);
    int remain_count = width*height;

    int cluster_idx = 0;
    while(remain_count > 0) {
        int xi = -1;
        int yi = -1;
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(visitmap[y*width + x]) continue;

                if(!immap[y*width + x]) {
                    visitmap[y*width + x] = true;
                    remain_count--;
                    continue;
                }

                xi = x;
                yi = y;
                goto find_loop1;
            }
        }
        find_loop1:
        if(xi < 0 || yi < 0) break;

        std::vector<std::pair<int,int>> stack;
        stack.emplace_back(xi,yi);

        while(!stack.empty()) {
            xi = stack.back().first;
            yi = stack.back().second;
            stack.pop_back();

            if(visitmap[yi*width + xi]) continue;

            visitmap[yi*width + xi] = true;
            remain_count--;

            if(immap[yi*width + xi]) {
                idmap[yi*width + xi] = cluster_idx;
                if(xi - 1 >= 0) {
                    stack.emplace_back(xi-1,yi);
                }
                if(yi - 1 >= 0) {
                    stack.emplace_back(xi,yi-1);
                }
                if(xi + 1 < width) {
                    stack.emplace_back(xi+1,yi);
                }
                if(yi + 1 < height) {
                    stack.emplace_back(xi,yi+1);
                }
            }
        }
        cluster_idx++;
    }
    return cluster_idx;
}

void make_lineblocker(
    std::vector<bool> &lineblocker,
    const std::vector<float> &sepimage)
{
    fprintf(stderr, "scan image2\n");
    lineblocker.resize(width*height, false);

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float value = sepimage[width * y + x];
            if (value > sep_valueth) {
                lineblocker[width * y + x] = true;
            }
        }
    }
    std::vector<int> blocker_cluster;
    int cluster_count = search_connection(lineblocker, blocker_cluster);
    std::vector<double> cluster_weight(cluster_count, 0);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int id = blocker_cluster[width * y + x];
            if(id < 0) continue;
            float value = sepimage[width * y + x];
            cluster_weight[id] += value;
        }
    }
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int id = blocker_cluster[width * y + x];
            if(id < 0) continue;
            if(cluster_weight[id] < sep_clusterth) {
                lineblocker[width * y + x] = false;
            }
        }
    }
    // for(auto v: cluster_weight) {
    //     std::cerr << v << std::endl;
    // }
    // {
    //     std::ofstream outfile("sepsmap.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             if(lineblocker[y * width + x]) {
    //                 outfile << "1 ";
    //             }
    //             else {
    //                 outfile << "0 ";
    //             }
    //         }
    //         outfile << std::endl;
    //     }
    // }
}

void hough_linefind(
    const std::vector<float> &lineimage, 
    const std::vector<float> &sepimage,
    std::vector<charbox> &boxes)
{
    std::vector<int> idimage;
    std::vector<int> idimage_main;
    prepare_id_image(idimage, idimage_main, boxes);

    std::vector<bool> lineblocker;
    make_lineblocker(lineblocker, sepimage);

    int num_rho = std::max(width, height) / 1;
    int num_theta = 180 / 1;

    std::vector<double> accumulator1(num_rho * num_theta);
    std::vector<double> cos_theta(num_theta);
    std::vector<double> sin_theta(num_theta);
    for (int theta_i = 0; theta_i < num_theta; theta_i++) {
        double theta = M_PI * theta_i / num_theta;
        cos_theta[theta_i] = cos(theta);
        sin_theta[theta_i] = sin(theta);
    }

    fprintf(stderr, "scan image\n");
    double d = sqrt(width * width + height * height);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float value1 = lineimage[width * y + x];
            if (value1 > line_valueth) {
                double px = x - width / 2.0;
                double py = y - height / 2.0;
                for (int theta_i = 0; theta_i < num_theta; theta_i++) {
                    double rho = px * cos_theta[theta_i] + py * sin_theta[theta_i];
                    int rho_i = (rho + d) / (2 * d) * num_rho;
                    if (rho_i < 0){
                        rho_i = 0;
                    }
                    if (rho_i >= num_rho) {
                        rho_i = num_rho - 1;
                    }
                    accumulator1[num_theta * rho_i + theta_i] += value1;
                }
            }
        }
    }
    // {
    //     std::ofstream outfile("acc.txt");
    //     for(int y = 0; y < num_rho; y++){
    //         for(int x = 0; x < num_theta; x++){
    //             outfile << accumulator1[y * num_theta + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }

    std::vector<double> sorted_acc;
    std::copy_if(accumulator1.begin(), accumulator1.end(), std::back_inserter(sorted_acc),[](double x){ return x > 0; });    
    std::sort(sorted_acc.begin(), sorted_acc.end(), [](auto a, auto b) {
        return a > b;
    });
    double sum_value = std::accumulate(sorted_acc.begin(), sorted_acc.end(), 0.0);
    double mean_value = sum_value / (num_rho * num_theta);
    fprintf(stderr, "count %lu mean %f\n", sorted_acc.size(), mean_value);
    double line_countth = std::min(mean_value * 5, sorted_acc[sorted_acc.size()/10]);

    fprintf(stderr, "line_countth %f\n", line_countth);
    std::vector<std::pair<double, std::pair<int, int> > > values1;
    if(line_countth > 0) {
        for (int rho_i = 0; rho_i < num_rho; rho_i++) {
            for (int theta_i = 0; theta_i < num_theta; theta_i++) {
                if (accumulator1[num_theta * rho_i + theta_i] < line_countth) {
                    continue;
                }
                const double allow_max_angle = 15;
                double theta = 180.0 / num_theta * theta_i / 180 * M_PI;
                if((180.0 / num_theta * theta_i > allow_max_angle && 180.0 / num_theta * theta_i < 90 - allow_max_angle) ||
                (180.0 / num_theta * theta_i > allow_max_angle + 90 && 180.0 / num_theta * theta_i < 180 - allow_max_angle) ||
                (180.0 / num_theta * theta_i > allow_max_angle + 180 && 180.0 / num_theta * theta_i < 270 - allow_max_angle) ||
                (180.0 / num_theta * theta_i > allow_max_angle + 270 && 180.0 / num_theta * theta_i < 360 - allow_max_angle)) 
                {
                    continue;
                }
                values1.emplace_back(accumulator1[num_theta * rho_i + theta_i], std::make_pair(rho_i, theta_i));
            }
        }
        std::sort(values1.begin(), values1.end(), [](auto a, auto b) {
            return a.first > b.first;
        });
    }
    fprintf(stderr, "lines %lu\n", values1.size());


    fprintf(stderr, "line find\n");
    std::vector<double> angle_map(width*height, std::nan(""));
    std::vector<int> lineid_map(width*height, -1);
    std::vector<std::vector<int>> lineid_idx;
    std::vector<double> lineangles;
    lineid_idx.push_back(std::vector<int>());
    lineangles.push_back(std::nan(""));
    int lineid_count = 0;
    for(auto v: values1) {
        int rho_i = v.second.first;
        int theta_i = v.second.second;

        double rho = (2 * d) / num_rho * rho_i - d;
        double theta = M_PI * theta_i / num_theta;

        double x0 = (cos_theta[theta_i] * rho) + width / 2.0;
        double y0 = (sin_theta[theta_i] * rho) + height / 2.0;

        int x1 = x0 - sin_theta[theta_i] * -std::max(width, height);
        int y1 = y0 + cos_theta[theta_i] * -std::max(width, height);
        int x2 = x0 - sin_theta[theta_i] * std::max(width, height);
        int y2 = y0 + cos_theta[theta_i] * std::max(width, height);

        int sign_direction = 1;
        if (abs(x1 - x2) < abs(y1 - y2)) {
            // 縦書き
            if(run_mode == 1) continue;
            if (y1 > y2) {
                sign_direction = -1;
            }
        }
        else {
            // 横書き
            if(run_mode == 2) continue;
            if (x1 > x2) {
                sign_direction = -1;
            }
        }
        double direction_theta = theta + M_PI_2;
        if (direction_theta > M_PI_4 * 3) {
            direction_theta -= M_PI;
        }

        // lineに沿ってスキャンする
        double scan_width = 5; // 横にずれてるのを許容
        int line_count = 10; // 連続している閾値
        int discont_count = 10; // 連続の切断許容値

        std::vector<std::pair<float, float>> t1;
        int t0 = -std::max(width, height);
        for (int t = -std::max(width, height); t < std::max(width, height); t++) {
            float xt0 = x0 - sin_theta[theta_i] * t * sign_direction;
            float yt0 = y0 + cos_theta[theta_i] * t * sign_direction;
            int xt = xt0;
            int yt = yt0;
            if (xt < 0 || xt >= width || yt < 0 || yt >= height) {
                continue;
            }

            t1.emplace_back(xt0,yt0);
        }

        std::vector<int> stack_idx;
        int count = 0;
        int discont = 0;
        for (int t = 0; t < t1.size(); t++) {
            int xt = t1[t].first;
            int yt = t1[t].second;
            if (xt < 0 || xt >= width || yt < 0 || yt >= height) {
                continue;
            }

            bool value_ok = false;
            if(lineblocker[yt * width + xt]) {
                value_ok = false;
                discont = 0;
            }
            else {
                for(int s = -scan_width; s <= scan_width; s++) {
                    int xd = t1[t].first + cos_theta[theta_i] * s;
                    int yd = t1[t].second + sin_theta[theta_i] * s;
                    if (xd < 0 || xd >= width || yd < 0 || yd >= height) {
                        continue;
                    }
                    if(lineid_map[yd * width + xd] < 0) {
                        if (lineimage[yd * width + xd] > line_valueth) {
                            value_ok = true;
                            stack_idx.push_back(yd * width + xd);
                        }
                    }
                }
            }
            if(value_ok) {
                count++;
                if(count > line_count) {
                    discont = discont_count;
                }
            }
            else if (--discont <= 0) {
                count = 0;
                stack_idx.clear();
                if(lineid_idx[lineid_count].size() > 0) {
                    lineid_count++;
                    lineid_idx.push_back(std::vector<int>());
                    lineangles.push_back(std::nan(""));
                }
            }

            if(count > line_count) {
                lineangles[lineid_count] = direction_theta;
                for(auto idx: stack_idx) {
                    angle_map[idx] = direction_theta;
                    lineid_map[idx] = lineid_count;
                    lineid_idx[lineid_count].push_back(idx);
                }
                stack_idx.clear();
            }
        }
    }

    // 分離して検出された線を統合する
    fprintf(stderr, "fix line\n");
    std::vector<int> line_area(lineid_idx.size());
    for(int i = 0; i < lineid_idx.size(); i++) {
        line_area[i] = lineid_idx[i].size();
    }
    while(true){
        // for(int i = 0; i < lineid_idx.size(); i++) {
        //     std::cerr << i << " " << line_area[i] << std::endl;
        // }
        std::vector<std::vector<int>> remap(lineid_idx.size());
        for(int i = 0; i < lineid_idx.size(); i++) {
            int maxid = i;
            for(auto idx: lineid_idx[i]) {
                int y = idx / width;
                int x = idx % width;

                for(int yi = y - 2; yi <= y + 2; yi++) {
                    for(int xi = x - 2; xi <= x + 2; xi++) {
                        if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
                            continue;
                        }
                        int id1 = lineid_map[yi * width + xi];
                        if(id1 < 0) continue;
                        if(line_area[id1] > line_area[maxid]) {
                            maxid = id1;                        
                        }
                    }
                }
            }
            if(maxid != i) {
                remap[maxid].push_back(i);
            }
        }
        int remap_count = 0;
        for(int i = 0; i < remap.size(); i++) {
            std::vector<int> remaplist;
            std::vector<int> stack;
            std::copy(remap[i].begin(), remap[i].end(), std::back_inserter(stack));
            while(!stack.empty()) {
                int k = stack.back();
                stack.pop_back();
                remaplist.push_back(k);
                std::copy(remap[k].begin(), remap[k].end(), std::back_inserter(stack));
            }
            for(auto k: remaplist) {
                if(fabs(lineangles[i]) < M_PI_4 && fabs(lineangles[k]) > M_PI_4) continue;
                if(fabs(lineangles[i]) > M_PI_4 && fabs(lineangles[k]) < M_PI_4) continue;
                remap_count++;
                line_area[i] += line_area[k];
                line_area[k] = 0;
                for(auto idx: lineid_idx[k]) {
                    lineid_map[idx] = i;
                    angle_map[idx] = lineangles[i];
                }
                std::copy(lineid_idx[k].begin(), lineid_idx[k].end(), std::back_inserter(lineid_idx[i]));
                lineid_idx[k].clear();
            }
        }
        if(remap_count == 0) break;
    }
    // 小さすぎる線は無視する
    for(int i = 0; i < lineid_idx.size(); i++) {
        std::cerr << i << " " << lineangles[i] << " " << line_area[i] << std::endl;
        if(line_area[i] < linearea_th) {
            for(auto idx: lineid_idx[i]) {
                lineid_map[idx] = -1;
                angle_map[idx] = std::nan("");
            }
        }
    }

    // {
    //     std::ofstream outfile("angle.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << angle_map[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }
    // {
    //     std::ofstream outfile("linemap.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << lineid_map[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }


    // 文字ボックスと線分の接触を検出して、ライン上にboxを並べる
    fprintf(stderr, "chain boxes1\n");
    std::vector<std::vector<int>> line_box_chain(lineid_count);
    for(int boxid = 0; boxid < boxes.size(); boxid++) {
        if((boxes[boxid].subtype & (2+4)) == (2+4)) continue; // ふりがなは後で
        for(int di = 0; di < std::max(boxes[boxid].w, boxes[boxid].h) / 2; di+=2) {
            {
                int yi = boxes[boxid].cy;
                int y = yi / 2;
                int xi = boxes[boxid].cx - di;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy;
                int y = yi / 2;
                int xi = boxes[boxid].cx + di;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy - di;
                int y = yi / 2;
                int xi = boxes[boxid].cx;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy + di;
                int y = yi / 2;
                int xi = boxes[boxid].cx;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
        }
    }

    // 既に検出できた文字の大きさを元に、線を太くして判定範囲を広げる
    fprintf(stderr, "line grow\n");
    std::vector<float> line_width(lineid_count);
    for(int i = 0; i < lineid_count; i++) {
        float max_width = 0;
        for(const auto &boxid: line_box_chain[i]) {
            float w = std::max(boxes[boxid].w, boxes[boxid].h);
            max_width = std::max(max_width, w);
        }
        line_width[i] = max_width / 3;
    }
    std::vector<int> lineid_map2(width*height, -1);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int lineid;
            if((lineid = lineid_map[y * width + x]) < 0) continue;
            lineid_map2[y * width + x] = lineid;
            float direction = angle_map[y * width + x];
            if(fabs(angle_map[y * width + x]) < M_PI_4) {
                // 横書き
                int max_width = line_width[lineid];
                for(int y2 = y; y2 >= std::max(0, y-max_width); y2--) {
                    lineid_map2[y2 * width + x] = lineid;
                    angle_map[y2 * width + x] = direction;
                }
            }
            else {
                // 縦書き
                int max_width = line_width[lineid];
                for(int x2 = x; x2 >= std::max(0, x-max_width/2); x2--) {
                    lineid_map2[y * width + x2] = lineid;
                    angle_map[y * width + x2] = direction;
                }
                for(int x2 = x; x2 < std::min(width, x+max_width/2+1); x2++) {
                    lineid_map2[y * width + x2] = lineid;
                    angle_map[y * width + x2] = direction;
                }
            }
        }
    }

    // {
    //     std::ofstream outfile("linemap.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << lineid_map2[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }

    // 線の切断をチェック
    fprintf(stderr, "line break\n");
    std::vector<bool> breaklines(lineid_count, false);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(lineblocker[y * width + x] && lineid_map2[y * width + x] >= 0) {
                breaklines[lineid_map2[y * width + x]] = true;
                float direction = angle_map[y * width + x];
                if(fabs(angle_map[y * width + x]) < M_PI_4) {
                    // 横書き
                    for(int y2 = y; y2 >= 0; y2--) {
                        if(lineid_map2[y2 * width + x] < 0) break;
                        lineid_map2[y2 * width + x] = -1;
                        angle_map[y2 * width + x] = std::nan("");
                    }
                    for(int y2 = y; y2 < height; y2++) {
                        if(lineid_map2[y2 * width + x] < 0) break;
                        lineid_map2[y2 * width + x] = -1;
                        angle_map[y2 * width + x] = std::nan("");
                    }
                }
                else {
                    // 縦書き
                    for(int x2 = x; x2 >= 0; x2--) {
                        if(lineid_map2[y * width + x2] < 0) break;
                        lineid_map2[y * width + x2] = -1;
                        angle_map[y * width + x2] = std::nan("");
                    }
                    for(int x2 = x; x2 < width; x2++) {
                        if(lineid_map2[y * width + x2] < 0) break;
                        lineid_map2[y * width + x2] = -1;
                        angle_map[y * width + x2] = std::nan("");
                    }
                }
            }
        }
    }
    for(int i = 0; i < breaklines.size(); i++){
        if(!breaklines[i]) continue;
        std::vector<int> areaid;
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(lineid_map2[y * width + x] == i) {
                    areaid.push_back(y * width + x);
                }
            }
        }

        std::vector<std::vector<int>> result_area_chain;
        while(!areaid.empty()) {
            int cur = areaid.back();
            areaid.pop_back();

            std::vector<int> stack;
            stack.push_back(cur);
    
            std::vector<int> area_chain;
            while(!stack.empty()) {
                cur = stack.back();
                stack.pop_back();                
                area_chain.push_back(cur);
                
                int x = cur % width;
                int y = cur / width;

                {
                    int x2 = x + 1;
                    int y2 = y;
                    if(x2 < width) {
                        int next = y2 * width + x2;
                        auto it = std::lower_bound(areaid.begin(), areaid.end(), next);
                        if(it != areaid.end() && *it == next) {
                            areaid.erase(it);
                            if(std::find(stack.begin(), stack.end(), next) == stack.end()) {
                                stack.push_back(next);
                            }
                        }
                    }
                }
                {
                    int x2 = x - 1;
                    int y2 = y;
                    if(x2 >= 0) {
                        int next = y2 * width + x2;
                        auto it = std::lower_bound(areaid.begin(), areaid.end(), next);
                        if(it != areaid.end() && *it == next) {
                            areaid.erase(it);
                            if(std::find(stack.begin(), stack.end(), next) == stack.end()) {
                                stack.push_back(next);
                            }
                        }
                    }
                }
                {
                    int x2 = x;
                    int y2 = y + 1;
                    if(y2 < height) {
                        int next = y2 * width + x2;
                        auto it = std::lower_bound(areaid.begin(), areaid.end(), next);
                        if(it != areaid.end() && *it == next) {
                            areaid.erase(it);
                            if(std::find(stack.begin(), stack.end(), next) == stack.end()) {
                                stack.push_back(next);
                            }
                        }
                    }
                }
                {
                    int x2 = x;
                    int y2 = y - 1;
                    if(y2 >= 0) {
                        int next = y2 * width + x2;
                        auto it = std::lower_bound(areaid.begin(), areaid.end(), next);
                        if(it != areaid.end() && *it == next) {
                            areaid.erase(it);
                            if(std::find(stack.begin(), stack.end(), next) == stack.end()) {
                                stack.push_back(next);
                            }
                        }
                    }
                }
            }
            result_area_chain.push_back(area_chain);
        }

        for(int j = 1; j < result_area_chain.size(); j++) {
            int newid = lineid_count;
            lineid_count++;
            for(auto idx: result_area_chain[j]) {
                lineid_map2[idx] = newid;
            }
        }
    }

    // {
    //     std::ofstream outfile("angle.txt");
    //     for(int y = 0; y < height; y++){
    //         for(int x = 0; x < width; x++){
    //             outfile << angle_map[y * width + x] << " ";
    //         }
    //         outfile << std::endl;
    //     }
    // }
    {
        std::ofstream outfile("linemap.txt");
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                outfile << lineid_map2[y * width + x] << " ";
            }
            outfile << std::endl;
        }
    }

    // 太くした線でもう一度、文字ボックスと線分の接触を検出して、ライン上にboxを並べる
    fprintf(stderr, "chain boxes2\n");
    line_box_chain.clear();
    line_box_chain.resize(lineid_count);
    for(int boxid = 0; boxid < boxes.size(); boxid++) {
        if((boxes[boxid].subtype & (2+4)) == (2+4)) continue; // ふりがなは後で
        for(int di = 0; di < std::max(boxes[boxid].w, boxes[boxid].h) / 2; di+=2) {
            {
                int yi = boxes[boxid].cy;
                int y = yi / 2;
                int xi = boxes[boxid].cx - di;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map2[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy;
                int y = yi / 2;
                int xi = boxes[boxid].cx + di;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map2[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy - di;
                int y = yi / 2;
                int xi = boxes[boxid].cx;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map2[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
            {
                int yi = boxes[boxid].cy + di;
                int y = yi / 2;
                int xi = boxes[boxid].cx;
                int x = xi / 2;
                if(y >= 0 && y < height && x >= 0 && x < width) {
                    int lineid = lineid_map2[y * width + x];
                    float direction = angle_map[y * width + x];
                    if(lineid >= 0 && !std::isnan(direction)) {
                        line_box_chain[lineid].push_back(boxid);
                        boxes[boxid].direction = direction;
                        break;
                    }
                }
            }
        }
    }

    fprintf(stderr, "remove alone\n");
    // 1以下のchainを消す
    for (auto it = line_box_chain.begin(); it != line_box_chain.end();) {
        if (it->size() < 2) {
            it = line_box_chain.erase(it);
        }
        else {
            ++it;
        }
    }

    search_loop(boxes, line_box_chain, lineblocker, idimage_main, sepimage);
    after_search(boxes, line_box_chain, lineblocker, idimage);
}

void space_chack(std::vector<charbox> &boxes)
{
    std::vector<int> sp_idx;
    for(int i = 0; i < boxes.size(); i++) {
        if((boxes[i].subtype & 8) == 8) {
            sp_idx.push_back(i);
        }
    }
    if (sp_idx.size() < 1) return;
    for(int i = 0; i < sp_idx.size() - 1; i++) {
        auto box1 = boxes[sp_idx[i]];
        auto box2 = boxes[sp_idx[i+1]];
        
        if((box1.subtype & 8) != 8) {
            continue;
        }

        float area1_vol = box1.w * box1.h;
        float area2_vol = box2.w * box2.h;
        float inter_xmin = std::max(box1.cx - box1.w / 2, box2.cx - box2.w / 2);
        float inter_ymin = std::max(box1.cy - box1.h / 2, box2.cy - box2.h / 2);
        float inter_xmax = std::min(box1.cx + box1.w / 2, box2.cx + box2.w / 2);
        float inter_ymax = std::min(box1.cy + box1.h / 2, box2.cy + box2.h / 2);
        float inter_w = std::max(inter_xmax - inter_xmin, 0.0f);
        float inter_h = std::max(inter_ymax - inter_ymin, 0.0f);
        float inter_vol = inter_w * inter_h;
        float union_vol = area1_vol + area2_vol - inter_vol;
        float iou = (union_vol > 0)? inter_vol/union_vol: 0;

        if(iou > 0) {
            boxes[sp_idx[i+1]].subtype &= ~8;
        }
    }
}

int main(int argn, char **argv) {
    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);

    fread(&run_mode, sizeof(u_int32_t), 1, stdin);

    fread(&width, sizeof(u_int32_t), 1, stdin);
    fread(&height, sizeof(u_int32_t), 1, stdin);

    std::vector<float> lineimage(width*height);
    fread(lineimage.data(), sizeof(float), width*height, stdin);
    std::vector<float> sepimage(width*height);
    fread(sepimage.data(), sizeof(float), width*height, stdin);

    int boxcount = 0;
    fread(&boxcount, sizeof(u_int32_t), 1, stdin);

    std::cerr << boxcount << std::endl;

    std::vector<charbox> boxes(boxcount);
    for(int i = 0; i < boxcount; i++) {
        boxes[i].id = i;
        boxes[i].block = -1;
        boxes[i].idx = -1;
        boxes[i].subidx = -1;
        boxes[i].subtype = 0;
        boxes[i].direction = 0;
        boxes[i].double_line = 0;
        fread(&boxes[i].cx, sizeof(float), 1, stdin);
        fread(&boxes[i].cy, sizeof(float), 1, stdin);
        fread(&boxes[i].w, sizeof(float), 1, stdin);
        fread(&boxes[i].h, sizeof(float), 1, stdin);
        fread(&boxes[i].code1, sizeof(float), 1, stdin);
        fread(&boxes[i].code2, sizeof(float), 1, stdin);
        fread(&boxes[i].code4, sizeof(float), 1, stdin);
        fread(&boxes[i].code8, sizeof(float), 1, stdin);
        // ルビ親文字
        if(boxes[i].code2 > rubybase_cutoff) {
            boxes[i].subtype |= 2;
        }
        // ルビの文字
        if(boxes[i].code1 > ruby_cutoff) {
            boxes[i].subtype |= 2+4;
        }
        // 空白
        if(boxes[i].code8 > space_cutoff) {
            boxes[i].subtype |= 8;
        }
        // fprintf(stderr, "box %d cx %f cy %f w %f h %f c1 %f c2 %f c4 %f c8 %f\n", 
        //     boxes[i].id, boxes[i].cx, boxes[i].cy, boxes[i].w, boxes[i].h, 
        //     boxes[i].code1, boxes[i].code2, boxes[i].code4, boxes[i].code8);
    }

    fread(&org_width, sizeof(u_int32_t), 1, stdin);
    fread(&org_height, sizeof(u_int32_t), 1, stdin);
    
    hough_linefind(lineimage, sepimage, boxes);

    space_chack(boxes);

    u_int32_t count = boxes.size();
    fwrite(&count, sizeof(int32_t), 1, stdout);

    for(int i = 0; i < boxes.size(); i++) {
        // fprintf(stderr, "box %d cx %f cy %f w %f h %f block %d idx %d sidx %d stype %d c1 %f c2 %f c4 %f c8 %f d %d\n", 
        //     boxes[i].id, boxes[i].cx, boxes[i].cy, boxes[i].w, boxes[i].h, 
        //     boxes[i].block, boxes[i].idx, boxes[i].subidx, boxes[i].subtype,
        //     boxes[i].code1, boxes[i].code2, boxes[i].code4, boxes[i].code8,
        //     boxes[i].double_line);
        
        fwrite(&boxes[i].id, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].block, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].idx, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].subidx, sizeof(int32_t), 1, stdout);
        fwrite(&boxes[i].subtype, sizeof(int32_t), 1, stdout);
    }

    return 0;
}