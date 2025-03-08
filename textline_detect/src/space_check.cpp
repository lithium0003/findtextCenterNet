#include "space_check.h"
#include <cmath>

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

void append_lostspace(std::vector<charbox> &boxes)
{
    std::vector<int> head_idx;
    for(int i = 0; i < boxes.size(); i++) {
        if(boxes[i].subidx > 0) continue;
        head_idx.push_back(i);
    }
    for(int i = 0; i < head_idx.size() - 1; i++) {
        int boxid1 = head_idx[i];
        if((boxes[boxid1].subtype & 8) != 8) continue;
        int boxid2 = head_idx[i+1];
        if((boxes[boxid2].subtype & 8) == 8) continue;

        if(boxes[boxid1].block != boxes[boxid2].block) continue;
        if((boxes[boxid1].subtype & 1) != (boxes[boxid2].subtype & 1)) continue;

        if((boxes[boxid1].subtype & 1) == 0) {
            // 横書き
            float allowdiff = std::min(boxes[boxid1].w, boxes[boxid2].w) * 0.5;
            if(fabs((boxes[boxid1].cx - boxes[boxid1].w/2) - (boxes[boxid2].cx - boxes[boxid2].w/2)) < allowdiff) {
                boxes[boxid2].subtype |= 8;
            }
        }
        else {
            // 縦書き
            float allowdiff = std::min(boxes[boxid1].h, boxes[boxid2].h) * 0.5;
            if(fabs((boxes[boxid1].cy - boxes[boxid1].h/2) - (boxes[boxid2].cy - boxes[boxid2].h/2)) < allowdiff) {
                boxes[boxid2].subtype |= 8;
            }            
        }
    }
    for(int i = int(head_idx.size()) - 1; i > 0; i--) {
        int boxid1 = head_idx[i];
        if((boxes[boxid1].subtype & 8) != 8) continue;
        int boxid2 = head_idx[i-1];
        if((boxes[boxid2].subtype & 8) == 8) continue;

        if(boxes[boxid1].block != boxes[boxid2].block) continue;
        if((boxes[boxid1].subtype & 1) != (boxes[boxid2].subtype & 1)) continue;

        if((boxes[boxid1].subtype & 1) == 0) {
            // 横書き
            float allowdiff = std::min(boxes[boxid1].w, boxes[boxid2].w) * 0.5;
            if(fabs((boxes[boxid1].cx - boxes[boxid1].w/2) - (boxes[boxid2].cx - boxes[boxid2].w/2)) < allowdiff) {
                boxes[boxid2].subtype |= 8;
            }
        }
        else {
            // 縦書き
            float allowdiff = std::min(boxes[boxid1].h, boxes[boxid2].h) * 0.5;
            if(fabs((boxes[boxid1].cy - boxes[boxid1].h/2) - (boxes[boxid2].cy - boxes[boxid2].h/2)) < allowdiff) {
                boxes[boxid2].subtype |= 8;
            }            
        }
    }
}

void space_chack(std::vector<charbox> &boxes)
{
    remove_dupspace(boxes);
    append_lostspace(boxes);
}
