#include "number_unbind.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <cmath>
#include <iostream>

// 未接続のboxを入れる
int number_unbind(
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage,
    int next_id)
{
    const double allow_maindiff = 1;
    const double allow_subdiff = 10;
    std::vector<int> unbind_boxes;
    for(const auto &box: boxes) {
        if(box.idx < 0) {
            unbind_boxes.push_back(box.id);
        }
    }
    if(unbind_boxes.size() == 0) {
        return next_id;
    }
    std::cerr << "number_unbind " << unbind_boxes.size() << std::endl;

    if(unbind_boxes.size() > 1) {
        // 水平方向の文字列があるかも
        // cyがどこかに固まっている
        std::vector<std::vector<int>> hori_line_boxid;
        std::vector<int> sortedcy_idx = unbind_boxes;
        std::sort(sortedcy_idx.begin(), sortedcy_idx.end(), [boxes](int a, int b) {
            return boxes[a].cy < boxes[b].cy;
        });

        std::vector<std::vector<int>> agg_idx;
        for(int i = 0; i < sortedcy_idx.size() - 1; i++) {
            int boxid1 = sortedcy_idx[i];
            int boxid2 = sortedcy_idx[i+1];
            float diff_cy = boxes[boxid2].cy - boxes[boxid1].cy;
            float s1 = std::max(boxes[boxid1].w, boxes[boxid1].h);
            float s2 = std::max(boxes[boxid2].w, boxes[boxid2].h);
            float s = std::max(s1, s2);

            if (diff_cy < s * allow_maindiff) {
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

        for(const auto &chain: agg_idx) {
            std::vector<int> sortedcx_idx = chain;
            std::sort(sortedcx_idx.begin(), sortedcx_idx.end(), [boxes](int a, int b) {
                return boxes[a].cx < boxes[b].cx;
            });

            for(int i = 0; i < sortedcx_idx.size() - 1; i++) {
                int boxid1 = sortedcx_idx[i];
                int boxid2 = sortedcx_idx[i+1];
                float diff_cx = boxes[boxid2].cx - boxes[boxid1].cx;
                float diff_cy = boxes[boxid2].cy - boxes[boxid1].cy;
                float s = std::max(boxes[boxid1].w, boxes[boxid1].h);

                if (fabs(diff_cy) > s * allow_maindiff) {
                    // reject
                    continue;
                }
                if (diff_cx > s * allow_subdiff) {
                    // reject
                    continue;
                }

                // 線の切断テスト
                float x1 = boxes[boxid1].cx + boxes[boxid1].w / 2;
                float x2 = boxes[boxid2].cx - boxes[boxid2].w / 2;
                float y1 = boxes[boxid1].cy;
                float y2 = boxes[boxid2].cy;
                int chain_idx = -1;
                if (fabs(x1 - x2) > 0) {
                    float a = (y2 - y1) / (x2 - x1);
                    for(int x = x1; x < x2; x += scale) {
                        float y = a * (x - x1) + y1;
                        int xi = x / scale;
                        int yi = y / scale;
                        if (xi < 0 || xi >= width || yi < 0 || yi >= height) continue;
                        if (lineblocker[yi * width + xi]) {
                            goto chainloop_next1;
                        }
                    }
                }

                // 確定しているBOXとの接触テスト
                for(const auto &box: boxes) {
                    if (box.idx < 0) { continue; }
                    if (boxes[boxid1].cx < box.cx && box.cx < boxes[boxid2].cx) {
                        if (std::min(boxes[boxid1].cy - boxes[boxid1].h / 2, boxes[boxid2].cy - boxes[boxid2].h / 2) < box.cy 
                            && box.cy < std::max(boxes[boxid1].cy + boxes[boxid1].h / 2, boxes[boxid2].cy + boxes[boxid2].h / 2)) {
                                goto chainloop_next1;                        
                        }
                    }
                }

                for(int j = 0; j < hori_line_boxid.size(); j++) {
                    if(std::find(hori_line_boxid[j].begin(), hori_line_boxid[j].end(), boxid1) != hori_line_boxid[j].end()) {
                        chain_idx = j;
                        break;
                    }
                }
                if(chain_idx < 0) {
                    std::vector<int> tmp_chain;
                    tmp_chain.push_back(boxid1);
                    tmp_chain.push_back(boxid2);
                    hori_line_boxid.push_back(tmp_chain);
                }
                else {
                    hori_line_boxid[chain_idx].push_back(boxid2);
                }

                chainloop_next1:
                ;
            }
        }

        // 垂直方向の文字列があるかも
        // cxがどこかに固まっている
        std::vector<std::vector<int>> vert_line_boxid;
        std::vector<int> sortedcx_idx = unbind_boxes;
        std::sort(sortedcx_idx.begin(), sortedcx_idx.end(), [boxes](int a, int b) {
            return boxes[a].cx < boxes[b].cx;
        });

        agg_idx.clear();
        for(int i = 0; i < sortedcx_idx.size() - 1; i++) {
            int boxid1 = sortedcx_idx[i];
            int boxid2 = sortedcx_idx[i+1];
            float diff_cx = boxes[boxid2].cx - boxes[boxid1].cx;
            float s1 = std::max(boxes[boxid1].w, boxes[boxid1].h);
            float s2 = std::max(boxes[boxid2].w, boxes[boxid2].h);
            float s = std::max(s1, s2);
            if (diff_cx < s * allow_maindiff) {
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

        for(const auto &chain: agg_idx) {
            std::vector<int> sortedcy_idx = chain;
            std::sort(sortedcy_idx.begin(), sortedcy_idx.end(), [boxes](int a, int b) {
                return boxes[a].cy < boxes[b].cy;
            });

            for(int i = 0; i < sortedcy_idx.size() - 1; i++) {
                int boxid1 = sortedcy_idx[i];
                int boxid2 = sortedcy_idx[i+1];
                float diff_cx = boxes[boxid2].cx - boxes[boxid1].cx;
                float diff_cy = boxes[boxid2].cy - boxes[boxid1].cy;
                float s = std::max(boxes[boxid1].w, boxes[boxid1].h);
                if (fabs(diff_cx) > s * allow_maindiff) {
                    // reject
                    continue;
                }
                if (diff_cy > s * allow_subdiff) {
                    // reject
                    continue;
                }

                // 線の切断テスト
                float x1 = boxes[boxid1].cx;
                float x2 = boxes[boxid2].cx;
                float y1 = boxes[boxid1].cy + boxes[boxid1].h / 2;
                float y2 = boxes[boxid2].cy - boxes[boxid2].h / 2;
                int chain_idx = -1;
                if (fabs(y1 - y2) > 0) {
                    float a = (x2 - x1) / (y2 - y1);
                    for(int y = y1; y < x2; y += scale) {
                        float x = a * (y - y1) + x1;
                        int xi = x / scale;
                        int yi = y / scale;
                        if (xi < 0 || xi >= width || yi < 0 || yi >= height) continue;
                        if (lineblocker[yi * width + xi]) {
                            goto chainloop_next2;
                        }
                    }
                }

                // 確定しているBOXとの接触テスト
                for(const auto &box: boxes) {
                    if (box.idx < 0) { continue; }
                    if (boxes[boxid1].cy < box.cy && box.cy < boxes[boxid2].cy) {
                        if (std::min(boxes[boxid1].cx - boxes[boxid1].w / 2, boxes[boxid2].cx - boxes[boxid2].w / 2) < box.cx 
                            && box.cx < std::max(boxes[boxid1].cx + boxes[boxid1].w / 2, boxes[boxid2].cx + boxes[boxid2].w / 2)) {
                                goto chainloop_next2;                        
                        }
                    }
                }

                for(int j = 0; j < vert_line_boxid.size(); j++) {
                    if(std::find(vert_line_boxid[j].begin(), vert_line_boxid[j].end(), boxid1) != vert_line_boxid[j].end()) {
                        chain_idx = j;
                        break;
                    }
                }
                if(chain_idx < 0) {
                    std::vector<int> tmp_chain;
                    tmp_chain.push_back(boxid1);
                    tmp_chain.push_back(boxid2);
                    vert_line_boxid.push_back(tmp_chain);
                }
                else {
                    vert_line_boxid[chain_idx].push_back(boxid2);
                }

                chainloop_next2:
                ;
            }
        }
        
        // for(auto line_chain: hori_line_boxid) {
        //     for(auto bid: line_chain) {
        //         std::cerr << bid << ' ';
        //     }
        //     std::cerr << std::endl;
        // }

        // for(auto line_chain: vert_line_boxid) {
        //     for(auto bid: line_chain) {
        //         std::cerr << bid << ' ';
        //     }
        //     std::cerr << std::endl;
        // }

        // 縦横で重複してラインに所属しているboxがあれば、長く取れた方を優先する
        auto flattened_hori_line_boxid = std::reduce(hori_line_boxid.begin(), hori_line_boxid.end(), 
            std::vector<int>(),
            [](auto &&x, auto &&y) {
                x.insert(x.end(), y.begin(), y.end());
                return x;
            });
        auto flattened_vert_line_boxid = std::reduce(vert_line_boxid.begin(), vert_line_boxid.end(), 
            std::vector<int>(),
            [](auto &&x, auto &&y) {
                x.insert(x.end(), y.begin(), y.end());
                return x;
            });
        std::sort(flattened_hori_line_boxid.begin(), flattened_hori_line_boxid.end());
        std::sort(flattened_vert_line_boxid.begin(), flattened_vert_line_boxid.end());
        std::vector<int> dup_box;
        std::set_intersection(flattened_hori_line_boxid.begin(), flattened_hori_line_boxid.end(),
            flattened_vert_line_boxid.begin(), flattened_vert_line_boxid.end(),
            std::back_inserter(dup_box));
        for(auto dupid: dup_box) {
            auto hp = std::find_if(hori_line_boxid.begin(), hori_line_boxid.end(),
                [dupid](auto a){ return std::count(a.begin(), a.end(), dupid) > 0; });
            if(hp == hori_line_boxid.end()) continue;
            auto vp = std::find_if(vert_line_boxid.begin(), vert_line_boxid.end(),
                [dupid](auto a){ return std::count(a.begin(), a.end(), dupid) > 0; });
            if(vp == vert_line_boxid.end()) continue;

            if(hp->size() >= vp->size()) {
                vert_line_boxid.erase(vp);
            }
            else {
                hori_line_boxid.erase(hp);
            }
        }

        // 反映
        for(const auto &chain: hori_line_boxid) {
            int lineidx = next_id++;
            int subidx = 0;
            for(auto boxid: chain) {
                boxes[boxid].idx = lineidx;
                boxes[boxid].subidx = subidx++;
                boxes[boxid].subtype &= ~1;
                boxes[boxid].direction = 0;
            }
        }

        for(const auto &chain: vert_line_boxid) {
            int idx = next_id++;
            int subidx = 0;
            for(const auto boxid: chain) {
                boxes[boxid].idx = idx;
                boxes[boxid].subidx = subidx++;
                boxes[boxid].subtype |= 1;
                boxes[boxid].direction = M_PI_2;
            }
        }
    }

    // 文章全体が、縦書きか横書きかを判定する
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

    // 孤立したboxを足しておく
    for(const auto boxid: unbind_boxes) {
        if(boxes[boxid].idx < 0) {
            boxes[boxid].idx = next_id++;
            boxes[boxid].subidx = 0;
            if (direction < 0) {
                // 縦書き優勢 -> 縦書きにしておく
                boxes[boxid].subtype &= ~1;
                boxes[boxid].direction = 0;
            }
            else {
                // 横書き優勢 -> 横書きにしておく
                boxes[boxid].subtype |= 1;
                boxes[boxid].direction = M_PI_2;
            }
        }
    }
    return next_id;
}
