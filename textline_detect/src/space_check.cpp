#include "space_check.h"
#include <cmath>
#include <numeric>
#include <algorithm>

//#include <iostream>
//#include <iterator>

#include "minpack/minpack.hpp"

std::vector<float> x_data;
std::vector<float> y_data;

int func(int m, int n, double *x, double *fvec)
{
    for(int i = 0; i < m; i++) {
        double xx = 1;
        double yy = 0;
        for(int j = 0; j < n; j++) {
            yy += x[j] * xx;
            xx *= x_data[i];
        }
        fvec[i] = y_data[i] - yy;
    }
    return 0;
}

double fit_curve(double x, const std::vector<double> &c)
{
    double xx = 1;
    double yy = 0;
    for(int j = 0; j < c.size(); j++) {
        yy += c[j] * xx;
        xx *= x;
    }
    return yy;
}

// 文字boxが重なっているときに、先行するBoxにのみspace flagを残す
void remove_dupspace(std::vector<charbox> &boxes)
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

// 行頭の字下げを検出して、spaceフラグを修正する
void find_lostspace(std::vector<charbox> &boxes)
{
    // blockごとに処理する
    std::vector<int> blocks;
    for(int i = 0; i < boxes.size(); i++) {
        if(boxes[i].idx == 0 && boxes[i].subidx == 0) {
            blocks.push_back(boxes[i].block);
        }
    }
    for(auto b: blocks) {
        // 先頭3文字をとってくる
        std::vector<std::vector<int>> lines_box;
        float s0 = 0;
        for(int i = 0; i < boxes.size(); i++) {
            if(boxes[i].block == b && (boxes[i].subtype & (2+4)) != (2+4)) {
                if((boxes[i].subtype & 1) == 0) {
                    // 横書き
                    s0 = std::max(s0, boxes[i].w);
                }
                else {
                    // 縦書き
                    s0 = std::max(s0, boxes[i].h);
                }
                while (lines_box.size() <= boxes[i].idx) {
                    lines_box.push_back(std::vector<int>());
                }
                if(lines_box[boxes[i].idx].size() < 3) {
                    lines_box[boxes[i].idx].push_back(i);
                }
            }
        }
        // 連続で細い文字の場合は除外する
        for(auto it = lines_box.begin(); it != lines_box.end();) {
            bool pass = false;
            for(auto bidx: *it) {
                if((boxes[bidx].subtype & 1) == 0) {
                    // 横書き
                    if(s0 - boxes[bidx].w < s0 * 0.5) {
                        pass = true;
                        break;
                    }
                }
                else {
                    // 縦書き
                    if(s0 - boxes[bidx].h < s0 * 0.5) {
                        pass = true;
                        break;
                    }
                }
            }
            if(pass) {
                ++it;
            }
            else {
                it = lines_box.erase(it);
            }
        }
        // 2行以上ないblockは処理対象外
        if(lines_box.size() < 2) continue;
        
        if((boxes[lines_box.front().front()].subtype & 1) == 0) {
            // 横書き
            float x0 = INFINITY;
            for(int i = 0; i < lines_box.size(); i++) {
                x0 = std::min(x0, boxes[lines_box[i].front()].cx);
            }
            for(auto it = lines_box.begin(); it != lines_box.end();) {
                // 深いインデントの行は処理しない
                if(boxes[it->front()].cx - x0 > s0 * 2.5) {
                    it = lines_box.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        else {
            // 縦書き
            float x0 = INFINITY;
            for(int i = 0; i < lines_box.size(); i++) {
                x0 = std::min(x0, boxes[lines_box[i].front()].cy);
            }
            for(auto it = lines_box.begin(); it != lines_box.end();) {
                // 深いインデントの行は処理しない
                if(boxes[it->front()].cy - x0 > s0 * 2.5) {
                    it = lines_box.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        // 2行以上ないblockは処理対象外
        if(lines_box.size() < 2) continue;

        // 現在のインデントをチェックする
        std::vector<bool> head_indents;
        for(int i = 0; i < lines_box.size(); i++) {
            auto line1 = lines_box[i];
            head_indents.push_back((boxes[line1.front()].subtype & 8) == 8);
        }

        // インデントがおかしい行をチェック
        std::vector<bool> head_skip(head_indents.size());
        std::vector<float> amx(head_indents.size());
        // 細くない文字で、一番上になっている行から開始する
        int k = 0;
        float minx = INFINITY;
        for(int i = 0; i < lines_box.size(); i++) {
            auto line2 = lines_box[i];
            if(line2.size() < 2) continue;
            if((boxes[line2.front()].subtype & 1) == 0) {
                // 横書き
                float w = boxes[line2[0]].w;
                if(w < s0 * 0.6) continue;
                float sx = boxes[line2[0]].cx - boxes[line2[0]].w / 2;
                if(sx < minx) {
                    minx = sx;
                    k = i;
                }
            }
            else {
                // 縦書き
                float w = boxes[line2[0]].h;
                if(w < s0 * 0.6) continue;
                float sx = boxes[line2[0]].cy - boxes[line2[0]].h / 2;
                if(sx < minx) {
                    minx = sx;
                    k = i;
                }
            }
        }
        if(k < lines_box.size() / 2) {
            for(int i = k; i < lines_box.size(); i++) {
                auto line2 = lines_box[i];
                if(line2.size() < 2) continue;
                float sx2,mx2;
                bool skip = false;
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    mx2 = (boxes[line2[0]].cx + boxes[line2[0]].w / 2 + boxes[line2[1]].cx - boxes[line2[1]].w / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cx + boxes[line2[1]].cx) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cx - boxes[line2[0]].cx > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].w + boxes[line2[1]].w < s0 * 0.85) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx < s0) {
                            skip = true;
                        }
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                else {
                    // 縦書き
                    mx2 = (boxes[line2[0]].cy + boxes[line2[0]].h / 2 + boxes[line2[1]].cy - boxes[line2[1]].h / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cy + boxes[line2[1]].cy) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cy - boxes[line2[0]].cy > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].h + boxes[line2[1]].h < s0 * 0.85) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy < s0) {
                            skip = true;
                        }
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                float delta = 0;
                for(int j = k+1; j < i; j++) {
                    if(amx[j] != 0 && amx[j-1] != 0) {
                        delta = (amx[j] - amx[j-1]) * 0.25 + delta * 0.75;
                    }
                }
                if(skip) {
                    head_skip[i] = true;
                }
                else {
                    if(i > 0 && amx[i-1] != 0) {
                        float fmx = amx[i-1] + delta;

                        if(mx2 < fmx && fabs(fmx - mx2) > s0 * 0.25) {
                            head_skip[i] = true;
                        }
                        else {
                            if(fabs(fmx - mx2) < s0 * 0.6) {
                                head_indents[i] = false;
                            }
                            else {
                                if(fabs(fmx - s0 - mx2) < s0 * 0.6) {
                                    head_indents[i] = false;
                                }
                                else if(fabs(fmx - sx2) < s0 * 0.6) {
                                    head_indents[i] = true;
                                }
                                else {
                                    head_skip[i] = true;
                                }
                            }
                        }
                    }
                }
                if(head_skip[i]) {
                    if(i > 0 && amx[i-1] != 0) {
                        float fmx = amx[i-1] + delta;
                        amx[i] = fmx;
                    }
                }
                else {
                    if(head_indents[i]) {
                        amx[i] = sx2;
                    }
                    else {
                        amx[i] = mx2;
                    }
                }
            }
            std::fill(head_skip.begin(), head_skip.end(), false);
            for(int i = (int)lines_box.size() - 1; i >= 0; i--) {
                auto line2 = lines_box[i];
                if(line2.size() < 2) continue;
                float sx2,mx2;
                bool skip = false;
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    mx2 = (boxes[line2[0]].cx + boxes[line2[0]].w / 2 + boxes[line2[1]].cx - boxes[line2[1]].w / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cx + boxes[line2[1]].cx) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cx - boxes[line2[0]].cx > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].w + boxes[line2[1]].w < s0 * 0.85) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx < s0) {
                            skip = true;
                        }
                        if(i > 0 && i < lines_box.size()-1 && boxes[line2[2]].cx - boxes[line2[0]].cx > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                else {
                    // 縦書き
                    mx2 = (boxes[line2[0]].cy + boxes[line2[0]].h / 2 + boxes[line2[1]].cy - boxes[line2[1]].h / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cy + boxes[line2[1]].cy) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cy - boxes[line2[0]].cy > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].h + boxes[line2[1]].h < s0 * 0.85) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy < s0) {
                            skip = true;
                        }
                        if(i > 0 && i < lines_box.size()-1 && boxes[line2[2]].cy - boxes[line2[0]].cy > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                float delta = 0;
                for(int j = (int)lines_box.size() - 2; j > i; j--) {
                    if(amx[j] != 0 && amx[j+1] != 0) {
                        delta = (amx[j] - amx[j+1]) * 0.25 + delta * 0.75;
                    }
                }
                if(skip) {
                    head_skip[i] = true;
                }
                else {
                    if(i < lines_box.size()-1 && amx[i+1] != 0) {
                        float fmx = amx[i] != 0 && delta == 0 ? amx[i] : amx[i+1] + delta;

                        if(mx2 < fmx && fabs(fmx - mx2) > s0 * 0.25) {
                            head_skip[i] = true;
                        }
                        else {
                            if(fabs(fmx - mx2) < s0 * 0.6) {
                                head_indents[i] = false;
                            }
                            else {
                                if(fabs(fmx - s0 - mx2) < s0 * 0.6) {
                                    head_indents[i] = false;
                                }
                                else if(fabs(fmx - sx2) < s0 * 0.6) {
                                    head_indents[i] = true;
                                }
                                else {
                                    head_skip[i] = true;
                                }
                            }
                        }
                    }
                }
                if(head_skip[i]) {
                    if(i < lines_box.size()-1 && amx[i+1] != 0 && amx[i] == 0) {
                        float fmx = amx[i+1] + delta;
                        amx[i] = fmx;
                    }
                }
                else {
                    if(head_indents[i]) {
                        amx[i] = sx2;
                    }
                    else {
                        amx[i] = mx2;
                    }
                }
            }
        }
        else {
            for(int i = k; i >= 0; i--) {
                auto line2 = lines_box[i];
                if(line2.size() < 2) continue;
                float sx2,mx2;
                bool skip = false;
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    mx2 = (boxes[line2[0]].cx + boxes[line2[0]].w / 2 + boxes[line2[1]].cx - boxes[line2[1]].w / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cx + boxes[line2[1]].cx) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cx - boxes[line2[0]].cx > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].w + boxes[line2[1]].w < s0 * 0.75) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx < s0) {
                            skip = true;
                        }
                        if(i > 0 && i < lines_box.size()-1 && boxes[line2[2]].cx - boxes[line2[0]].cx > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                else {
                    // 縦書き
                    mx2 = (boxes[line2[0]].cy + boxes[line2[0]].h / 2 + boxes[line2[1]].cy - boxes[line2[1]].h / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cy + boxes[line2[1]].cy) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cy - boxes[line2[0]].cy > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].h + boxes[line2[1]].h < s0 * 0.75) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy < s0) {
                            skip = true;
                        }
                        if(i > 0 && i < lines_box.size()-1 && boxes[line2[2]].cy - boxes[line2[0]].cy > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                float delta = 0;
                for(int j = k-1; j > i; j--) {
                    if(amx[j] != 0 && amx[j+1] != 0) {
                        delta = (amx[j] - amx[j+1]) * 0.25 + delta * 0.75;
                    }
                }
                if(skip) {
                    head_skip[i] = true;
                }
                else {
                    if(i < lines_box.size()-1 && amx[i+1] != 0) {
                        float fmx = amx[i+1] + delta;

                        if(mx2 < fmx && fabs(fmx - mx2) > s0 * 0.25) {
                            head_skip[i] = true;
                        }
                        else {
                            if(fabs(fmx - mx2) < s0 * 0.6) {
                                head_indents[i] = false;
                            }
                            else {
                                if(fabs(fmx - s0 - mx2) < s0 * 0.6) {
                                    head_indents[i] = false;
                                }
                                else if(fabs(fmx - sx2) < s0 * 0.6) {
                                    head_indents[i] = true;
                                }
                                else {
                                    head_skip[i] = true;
                                }
                            }
                        }
                    }
                }
                if(head_skip[i]) {
                    if(i < lines_box.size()-1 && amx[i+1] != 0) {
                        float fmx = amx[i+1] + delta;
                        amx[i] = fmx;
                    }
                }
                else {
                    if(head_indents[i]) {
                        amx[i] = sx2;
                    }
                    else {
                        amx[i] = mx2;
                    }
                }
            }
            std::fill(head_skip.begin(), head_skip.end(), false);
            for(int i = 0; i < lines_box.size(); i++) {
                auto line2 = lines_box[i];
                if(line2.size() < 2) continue;
                float sx2,mx2;
                bool skip = false;
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    mx2 = (boxes[line2[0]].cx + boxes[line2[0]].w / 2 + boxes[line2[1]].cx - boxes[line2[1]].w / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cx + boxes[line2[1]].cx) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cx - boxes[line2[0]].cx > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].w + boxes[line2[1]].w < s0 * 0.75) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx < s0) {
                            skip = true;
                        }
                        if(boxes[line2[2]].cx - boxes[line2[0]].cx > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                else {
                    // 縦書き
                    mx2 = (boxes[line2[0]].cy + boxes[line2[0]].h / 2 + boxes[line2[1]].cy - boxes[line2[1]].h / 2) / 2;
                    mx2 = std::max(mx2, (boxes[line2[0]].cy + boxes[line2[1]].cy) / 2);
                    sx2 = mx2 - s0;
                    if(boxes[line2[1]].cy - boxes[line2[0]].cy > s0 * 1.15) {
                        skip = true;
                    }
                    if(boxes[line2[0]].h + boxes[line2[1]].h < s0) {
                        skip = true;
                    }
                    if(line2.size() == 3) {
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy < s0 * 0.75) {
                            skip = true;
                        }
                        if(boxes[line2[2]].cy - boxes[line2[0]].cy > s0 * 2.2) {
                            skip = true;
                        }
                    }
                }
                float delta = 0;
                for(int j = 1; j < i; j++) {
                    if(amx[j] != 0 && amx[j-1] != 0) {
                        delta = (amx[j] - amx[j-1]) * 0.25 + delta * 0.75;
                    }
                }
                if(skip) {
                    head_skip[i] = true;
                }
                else {
                    if(i > 0 && amx[i-1] != 0) {
                        float fmx = amx[i] != 0 && delta == 0 ? amx[i] : amx[i-1] + delta;

                        if(mx2 < fmx && fabs(fmx - mx2) > s0 * 0.25) {
                            head_skip[i] = true;
                        }
                        else {
                            if(fabs(fmx - mx2) < s0 * 0.6) {
                                head_indents[i] = false;
                            }
                            else {
                                if(fabs(fmx - s0 - mx2) < s0 * 0.6) {
                                    head_indents[i] = false;
                                }
                                else if(fabs(fmx - sx2) < s0 * 0.6) {
                                    head_indents[i] = true;
                                }
                                else {
                                    head_skip[i] = true;
                                }
                            }
                        }
                    }
                }
                if(head_skip[i]) {
                    if(i > 0 && amx[i-1] != 0 && amx[i] == 0) {
                        float fmx = amx[i-1] + delta;
                        amx[i] = fmx;
                    }
                }
                else {
                    if(head_indents[i]) {
                        amx[i] = sx2;
                    }
                    else {
                        amx[i] = mx2;
                    }
                }
            }
        }

        std::vector<std::vector<float>> sx;
        std::vector<std::vector<float>> sy;
        std::vector<std::vector<float>> cx;
        for(int i = 0; i < lines_box.size(); i++) {
            auto line1 = lines_box[i];
            std::vector<float> tmp_sx;
            std::vector<float> tmp_sy;
            std::vector<float> tmp_cx;
            for(auto bidx: line1) {
                if((boxes[bidx].subtype & 1) == 0) {
                    // 横書き
                    tmp_sx.push_back(boxes[bidx].cx - boxes[bidx].w/2);
                    tmp_sx.push_back(boxes[bidx].cx + boxes[bidx].w/2);
                    tmp_sy.push_back(boxes[bidx].cy);
                    tmp_sy.push_back(boxes[bidx].cy);
                    tmp_cx.push_back(boxes[bidx].cx);
                }
                else {
                    // 縦書き
                    tmp_sx.push_back(boxes[bidx].cy - boxes[bidx].h/2);
                    tmp_sx.push_back(boxes[bidx].cy + boxes[bidx].h/2);
                    tmp_sy.push_back(boxes[bidx].cx);
                    tmp_sy.push_back(boxes[bidx].cx);
                    tmp_cx.push_back(boxes[bidx].cy);
                }
            }
            sx.push_back(tmp_sx);
            sy.push_back(tmp_sy);
            cx.push_back(tmp_cx);
        }

        x_data.clear();
        y_data.clear();
        
        for(int i = 0; i < lines_box.size(); i++) {
            if(sx[i].size() < 2) continue;
            if(head_skip[i]) continue;
            if(head_indents[i]) {
                x_data.push_back(sy[i][0]);
                y_data.push_back(sx[i][0]);
            }
            else {
                x_data.push_back(sy[i][1]);
                y_data.push_back(sx[i][1]);
            }
        }

        int m = (int)y_data.size();
        int n = std::min(4, m);
        std::vector<double> x(n);
        std::vector<double> fvec(m);
        std::fill(x.begin(), x.end(), 0);
        lmdif1(func, m, n, x.data(), fvec.data());

//        std::cout << "x=[" << std::endl;
//        std::copy(x_data.begin(), x_data.end(), std::ostream_iterator<double>(std::cout,","));
//        std::cout << std::endl;
//        std::cout << "]" << std::endl;
//        std::cout << "y=[" << std::endl;
//        std::copy(y_data.begin(), y_data.end(), std::ostream_iterator<double>(std::cout,","));
//        std::cout << std::endl;
//        std::cout << "]" << std::endl;
//        std::cout << "c2 = np.array([" << std::endl;
//        std::copy(x.begin(), x.end(), std::ostream_iterator<double>(std::cout,","));
//        std::cout << std::endl;
//        std::cout << "])[::-1]" << std::endl;

        for(int i = 0; i < lines_box.size(); i++) {
            float lx = fit_curve(sy[i][0], x);
            head_indents[i] = cx[i][0] > lx;
        }
        
        for(int i = 0; i < lines_box.size(); i++) {
            auto line1 = lines_box[i];
            if(head_indents[i]) {
                boxes[line1.front()].subtype |= 8;
            }
            else {
                boxes[line1.front()].subtype &= ~8;
            }
        }
    }
}

void space_chack(std::vector<charbox> &boxes)
{
    remove_dupspace(boxes);
    find_lostspace(boxes);
}
