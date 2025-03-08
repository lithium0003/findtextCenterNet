#include "split_doubleline.h"
#include "search_loop.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <iostream>
#include <cmath>

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
            for(const auto boxid: *chain_it) {
                float x = boxes[boxid].cx;

                if(fabs(x - x1) > 3 * std::max(max_w, max_h)) {
                    split_flag = true;    
                }
                if(split_flag) {
                    splited.push_back(boxid);
                }
                x1 = x;
            }
        }
        else {
            //縦書き
            float y1 = boxes[front_id].cy;
            for(const auto boxid: *chain_it) {
                float y = boxes[boxid].cy;

                if(fabs(y - y1) > 3 * std::max(max_w, max_h)) {
                    split_flag = true;    
                }
                if(split_flag) {
                    splited.push_back(boxid);
                }
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
                            if(boxes[last_idx].cy + boxes[last_idx].h / 2 * 0.75 < boxes[boxid].cy - boxes[boxid].h / 2 * 0.75) {
                                boxes[last_idx].double_line = 1;
                                boxes[boxid].double_line = 2;
                            }
                        }
                        else {
                            if(boxes[boxid].cy + boxes[boxid].h / 2 * 0.75 < boxes[last_idx].cy - boxes[last_idx].h / 2 * 0.75) {
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
                float cy1_s = (cy1.size() > 0) ? std::accumulate(cy1.begin(), cy1.end(), 0.0) / cy1.size() : -1;
                float cy2_s = (cy2.size() > 0) ? std::accumulate(cy2.begin(), cy2.end(), 0.0) / cy2.size() : -1;
                int splitcount = 0;
                for(const auto boxid: *chain_it) {
                    if((boxes[boxid].subtype & (2+4)) == 2+4) continue;
                    if(boxes[boxid].double_line > 0) {
                        splitcount++;
                    }
                    if(splitcount > 1 && boxes[boxid].double_line == 0) {
                        if(fabs(boxes[boxid].cy - cy1_s) < h_s / 5) {
                            boxes[boxid].double_line = 1;
                        }
                        else if(fabs(boxes[boxid].cy - cy2_s) < h_s / 5) {
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
                            if(boxes[boxid].cx + boxes[boxid].w / 2 * 0.75 < boxes[last_idx].cx - boxes[last_idx].w / 2 * 0.75) {
                                boxes[last_idx].double_line = 1;
                                boxes[boxid].double_line = 2;
                            }
                        }
                        else {
                            if(boxes[last_idx].cx + boxes[last_idx].w / 2 * 0.75 < boxes[boxid].cx - boxes[boxid].w / 2 * 0.75) {
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
                    if((boxes[boxid].subtype & (2+4)) == 2+4) continue;
                    if(boxes[boxid].double_line > 0) {
                        splitcount++;
                    }
                    else if(splitcount > 2) {
                        if(std::max(boxes[boxid].h, boxes[boxid].w) > s_s * 1.5) {
                            splitcount = 0;
                            continue;
                        }

                        if(boxes[boxid].cx - boxes[boxid].w / 2 * 0.75 > cx2_s + w_s / 2 * 0.75) {
                            boxes[boxid].double_line = 1;
                        }
                        else if(boxes[boxid].cx + boxes[boxid].w / 2 * 0.75 < cx1_s - w_s / 2 * 0.75) {
                            boxes[boxid].double_line = 2;
                        }
                        else {
                            splitcount = 0;
                        }
                    }
                }
            }
        }

        // while(std::count_if(chain_it->begin(), chain_it->end(), [boxes](int i){ return boxes[i].double_line > 0; }) > 0) {
        //     std::vector<int> splited1;
        //     std::vector<int> splited2;
        //     std::vector<int> remain;

        //     for(auto it = chain_it->begin(); it != chain_it->end();++it) {
        //         if(boxes[*it].double_line == 1) {
        //             splited1.push_back(*it);
        //         }
        //         else if (boxes[*it].double_line == 2) {
        //             splited2.push_back(*it);
        //         }
        //         else {
        //             if (splited1.size() > 1 && splited2.size() > 1) {
        //                 std::copy(it, chain_it->end(), back_inserter(remain));
        //                 break;
        //             }
        //             else {
        //                 splited1.clear();
        //                 splited2.clear();
        //             }
        //         }
        //     }
        //     if(splited1.size() > 1) {
        //         for(auto it = chain_it->begin(); it != chain_it->end();) {
        //             if(std::find(splited1.begin(), splited1.end(), *it) != splited1.end()) {
        //                 it = chain_it->erase(it);
        //             }
        //             else {
        //                 ++it;
        //             }
        //         }
        //     }
        //     if(splited2.size() > 1) {
        //         for(auto it = chain_it->begin(); it != chain_it->end();) {
        //             if(std::find(splited2.begin(), splited2.end(), *it) != splited2.end()) {
        //                 it = chain_it->erase(it);
        //             }
        //             else {
        //                 ++it;
        //             }
        //         }
        //     }
        //     if(remain.size() > 0) {
        //         for(auto it = chain_it->begin(); it != chain_it->end();) {
        //             if(std::find(remain.begin(), remain.end(), *it) != remain.end()) {
        //                 it = chain_it->erase(it);
        //             }
        //             else {
        //                 ++it;
        //             }
        //         }
        //     }
        //     if(splited1.size() > 1) {
        //         sort_chain(splited1, boxes);
        //         chain_it = line_box_chain.insert(chain_it, splited1);
        //     }
        //     if(splited2.size() > 1) {
        //         sort_chain(splited2, boxes);
        //         chain_it = line_box_chain.insert(chain_it, splited2);
        //     }
        //     if(remain.size() > 0) {
        //         sort_chain(remain, boxes);
        //         chain_it = line_box_chain.insert(chain_it, remain);
        //     }

        //     if(remain.size() == 0) {
        //         break;
        //     }
        // }
    }
}
