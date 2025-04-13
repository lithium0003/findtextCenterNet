#include "space_check.h"
#include <cmath>
#include <numeric>

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
                    if(s0 - boxes[bidx].w < s0 * 0.25) {
                        pass = true;
                        break;
                    }
                }
                else {
                    // 縦書き
                    if(s0 - boxes[bidx].h < s0 * 0.25) {
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
                if(boxes[it->front()].cx - x0 > s0 * 2) {
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
                if(boxes[it->front()].cy - x0 > s0 * 2) {
                    it = lines_box.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        // 2行以上ないblockは処理対象外
        if(lines_box.size() < 2) continue;

        std::vector<int> head_offsets;
        std::vector<bool> head_indents;
        for(int i = 0; i < lines_box.size(); i++) {
            auto line1 = lines_box[i];
            head_indents.push_back((boxes[line1.front()].subtype & 8) == 8);
        }
        for(int i = 0; i < lines_box.size() - 1; i++) {
            auto line1 = lines_box[i];
            auto line2 = lines_box[i+1];
            if(line1.size() < 1 || line2.size() < 1) continue;
            std::vector<float> x1;
            std::vector<float> x2;
            float s = 0;
            std::vector<float> s1;
            std::vector<float> s2;
            // 座標とサイズの確認
            for(int j = 0; j < line1.size(); j++) {
                if((boxes[line1.front()].subtype & 1) == 0) {
                    // 横書き
                    s = std::max(s, boxes[line1[j]].w);
                    x1.push_back(boxes[line1[j]].cx - boxes[line1[j]].w/2);
                }
                else {
                    // 縦書き
                    s = std::max(s, boxes[line1[j]].h);
                    x1.push_back(boxes[line1[j]].cy - boxes[line1[j]].h/2);
                }
            }
            for(int j = 0; j < line2.size(); j++) {
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    s = std::max(s, boxes[line2[j]].w);
                    x2.push_back(boxes[line2[j]].cx - boxes[line2[j]].w/2);
                }
                else {
                    // 縦書き
                    s = std::max(s, boxes[line2[j]].h);
                    x2.push_back(boxes[line2[j]].cy - boxes[line2[j]].h/2);
                }
            }
            for(int j = 0; j < line1.size(); j++) {
                if((boxes[line1.front()].subtype & 1) == 0) {
                    // 横書き
                    if((s - boxes[line1[j]].w) / s > 0.6) {
                        // 句読点など小さすぎる文字はサイズを無視する
                        s1.push_back(0);
                    }
                    else {
                        s1.push_back(s - boxes[line1[j]].w);
                    }
                }
                else {
                    // 縦書き
                    if((s - boxes[line1[j]].h) / s > 0.6) {
                        // 句読点など小さすぎる文字はサイズを無視する
                        s1.push_back(0);
                    }
                    else {
                        s1.push_back(s - boxes[line1[j]].h);
                    }
                }
            }
            for(int j = 0; j < line2.size(); j++) {
                if((boxes[line2.front()].subtype & 1) == 0) {
                    // 横書き
                    if((s - boxes[line2[j]].w) / s > 0.6) {
                        // 句読点など小さすぎる文字はサイズを無視する
                        s2.push_back(0);
                    }
                    else {
                        s2.push_back(s - boxes[line2[j]].w);
                    }
                }
                else {
                    // 縦書き
                    if((s - boxes[line2[j]].h) / s > 0.6) {
                        // 句読点など小さすぎる文字はサイズを無視する
                        s2.push_back(0);
                    }
                    else {
                        s2.push_back(s - boxes[line2[j]].h);
                    }
                }
            }
            int offset = -1;
            int th_c = 1;
            while(offset < 0) {
                for(int j = 0; j < x1.size(); j++) {
                    std::vector<float> diff;
                    // line2 がインデント
                    diff.push_back(j + 1 < x1.size() && j < x2.size() ? fabs(x1[j+1] - x2[j]): INFINITY);
                    diff.push_back(j + 1 < x1.size() && j < x2.size() ? fabs(x1[j+1] - s1[j+1] - x2[j]): INFINITY);
                    diff.push_back(j + 1 < x1.size() && j < x2.size() ? fabs(x1[j+1] - x2[j] + s2[j]): INFINITY);
                    // 同じ高さ
                    diff.push_back(j < x2.size() ? fabs(x1[j] - x2[j]): INFINITY);
                    diff.push_back(j < x2.size() ? fabs(x1[j] - s1[j] - x2[j]): INFINITY);
                    diff.push_back(j < x2.size() ? fabs(x1[j] - x2[j] + s2[j]): INFINITY);
                    // line1 がインデント
                    diff.push_back(j + 1 < x2.size() ? fabs(x1[j] - x2[j+1]): INFINITY);
                    diff.push_back(j + 1 < x2.size() ? fabs(x1[j] - s1[j] - x2[j+1]): INFINITY);
                    diff.push_back(j + 1 < x2.size() ? fabs(x1[j] - x2[j+1] + s2[j+1]): INFINITY);

                    std::vector<int> index(diff.size());
                    std::iota(index.begin(), index.end(), 0);
                    std::sort(index.begin(), index.end(), [&](const auto a, const auto b){
                        return diff[a] < diff[b];
                    });
                    if(diff[index.front()] < s * th_c * 0.1) {
                        offset = index.front() / 3;
                        break;
                    }
                }
                th_c++;
            }
            head_offsets.push_back(offset - 1);
        }
        for(int i = 0; i < lines_box.size()-1; i++) {
            if(head_offsets[i] == 0) {
                head_indents[i+1] = head_indents[i];
            }
            else if(head_offsets[i] > 0) {
                head_indents[i+1] = false;
                head_indents[i] = true;
            }
            else if(head_offsets[i] < 0) {
                head_indents[i+1] = true;
                head_indents[i] = false;
            }
        }
        for(int i = (int)lines_box.size()-1; i > 0; i--) {
            if(head_offsets[i-1] == 0) {
                head_indents[i-1] = head_indents[i];
            }
        }
        for(int i = 0; i < lines_box.size()-1; i++) {
            if(head_offsets[i] == 0) {
                head_indents[i+1] = head_indents[i];
            }
            else if(head_offsets[i] > 0) {
                head_indents[i+1] = false;
                head_indents[i] = true;
            }
            else if(head_offsets[i] < 0) {
                head_indents[i+1] = true;
                head_indents[i] = false;
            }
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
