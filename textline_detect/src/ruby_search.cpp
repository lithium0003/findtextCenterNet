#include "ruby_search.h"
#include "search_loop.h"

#include <algorithm>
#include <numeric>
#include <iterator>

#include <stdio.h>
#include <iostream>
#include <cmath>

void search_ruby(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage)
{
    std::cerr << "search_ruby" << std::endl;

    auto chain_map = create_chainid_map(boxes, line_box_chain, lineblocker);

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        if(line_box_chain[chainid].size() < 2) continue;
        
        // chain内をソート
        sort_chain(line_box_chain[chainid], boxes);

        std::vector<int> x;
        std::vector<int> y;
        float direction;
        double w, h;
        make_track_line(x,y, direction, w, h, boxes, line_box_chain, lineblocker, chainid, 1);

        if(fabs(direction) < M_PI_4) {
            // 横書き

            // ルビの所属chainを検索
            std::vector<int> ruby_boxid;
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int k = 1; k < h * 1.25 / scale; k++) {
                    int yp = yi - k;
                    if(yp < 0 || yp >= height) continue;
                    if(lineblocker[yp * width + xi]) break;
                    int other_chain = chain_map[yp * width + xi];
                    if (other_chain >= 0 && other_chain != chainid) break;
                    int other_id = idimage[yp * width + xi];
                    if(other_id < 0) continue;
                    if((boxes[other_id].subtype & (2+4)) != 2+4) continue;                        
                    if(std::find(ruby_boxid.begin(), ruby_boxid.end(), other_id) != ruby_boxid.end()) continue;
                    if(boxes[other_id].idx >= 0) continue;
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
        }
        else {
            // 縦書き

            // ルビの所属chainを検索
            std::vector<int> ruby_boxid;
            for(int i = 0; i < x.size(); i++) {
                int xi = x[i] / scale;
                int yi = y[i] / scale;
                if(xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
                for(int k = 1; k < w * 1.25 / scale; k++) {
                    int xp = xi + k;
                    if(xp < 0 || xp >= width) continue;
                    if(lineblocker[yi * width + xp]) break;
                    int other_chain = chain_map[yi * width + xp];
                    if (other_chain >= 0 && other_chain != chainid) break;
                    int other_id = idimage[yi * width + xp];
                    if(other_id < 0) continue;
                    if((boxes[other_id].subtype & (2+4)) != 2+4) continue;
                    if(std::find(ruby_boxid.begin(), ruby_boxid.end(), other_id) != ruby_boxid.end()) continue;
                    if(boxes[other_id].idx >= 0) continue;
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
        }
    }

    fix_chain_info(boxes, line_box_chain);
    // print_chaininfo(boxes, line_box_chain);

    for(int chainid = 0; chainid < line_box_chain.size(); chainid++) {
        sort_chain(line_box_chain[chainid], boxes);

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

        // ルビがない行は飛ばす
        if(rubyid.empty()) {
            int subidx = 0;
            for(auto boxid: line_box_chain[chainid]) {
                boxes[boxid].subtype &= ~(2+4);
                boxes[boxid].idx = chainid;
                boxes[boxid].subidx = subidx++;
            }
            continue;
        }

        // ルビの親文字を探す
        std::vector<std::vector<int>> rubybaseBlock;
        std::vector<int> tmp;
        for(auto boxid: baseid) {
            if((boxes[boxid].subtype & (2+4)) == 2) {
                tmp.push_back(boxid);
            }
            else {
                if(!tmp.empty()) {
                    rubybaseBlock.push_back(tmp);
                    tmp.clear();
                }
            }
        }
        if(!tmp.empty()) {
            rubybaseBlock.push_back(tmp);
            tmp.clear();
        }

        std::vector<int> fixlist;
        bool horizontal = false;
        if(fabs(boxes[line_box_chain[chainid].front()].direction) < M_PI_4) {
            // 横書き
            horizontal = true;

            double w1 = std::transform_reduce(
                baseid.begin(), 
                baseid.end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].w; });
            double w2 = std::transform_reduce(
                rubyid.begin(), 
                rubyid.end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].w; });

            // ルビの親文字のブロック座標
            std::vector<std::pair<float, float>> rubybaseBlock_x;
            for(auto rubyblock: rubybaseBlock) {
                auto x1 = boxes[rubyblock.front()].cx - w1;
                auto x2 = boxes[rubyblock.back()].cx + w1;
                rubybaseBlock_x.emplace_back(x1,x2);
            }

            // ルビは通常、ルビ付き文字のブロックの間に付いている
            std::vector<int> ruby_to_base(rubyid.size(), -1);
            for(int i = 0; i < rubyid.size(); i++) {
                int boxid = rubyid[i];
                for(int j = 0; j < rubybaseBlock_x.size(); j++) {
                    if(rubybaseBlock_x[j].first < boxes[boxid].cx && boxes[boxid].cx < rubybaseBlock_x[j].second) {
                        ruby_to_base[i] = j;
                        break;
                    }
                }
            }

            // ルビの方が長い場合などは、ブロックから外れることがある
            if(std::count(ruby_to_base.begin(), ruby_to_base.end(), -1) > 0 
                && std::count_if(ruby_to_base.begin(), ruby_to_base.end(), [](auto x){ return x >= 0; }) > 0) {
                
                for(int i = 0; i < rubyid.size(); i++) {
                    if(ruby_to_base[i] >= 0) continue;

                    // 隣のルビとの距離
                    float dx1 = INFINITY;
                    float dx2 = INFINITY;
                    if (i > 0) {
                        dx1 = boxes[rubyid[i]].cx - boxes[rubyid[i-1]].cx;
                    }
                    if (i < rubyid.size() - 1) {
                        dx2 = boxes[rubyid[i+1]].cx - boxes[rubyid[i]].cx;
                    }

                    if (dx1 <= w2 * 2 && dx2 <= w2 * 2) {
                        // 前後と連結している
                        // おそらく前後は連続のふりがな(多分どちらを選択してもよい)
                        
                        // 前後とも未確定
                        if (ruby_to_base[i-1] < 0 && ruby_to_base[i+1] < 0) {
                            // まず前方に検索する
                            for(int k = i-1; k > 0; k--) {
                                float dx11 = boxes[rubyid[k]].cx - boxes[rubyid[k-1]].cx;
                                if (dx11 <= w2 * 2) {
                                    if(ruby_to_base[k-1] < 0) {
                                        // まだ未確定
                                        continue;
                                    }
                                    ruby_to_base[i] = ruby_to_base[k-1];
                                    break;
                                }
                                else {
                                    // ルビが途切れたので探索失敗
                                    break;
                                }
                            }
                            // 前方で失敗した場合後方に検索する
                            if(ruby_to_base[i] < 0) {
                                for(int k = i+1; k < rubyid.size()-1; k++) {
                                    float dx22 = boxes[rubyid[k+1]].cx - boxes[rubyid[k]].cx;
                                    if (dx22 <= w2 * 2) {
                                        if(ruby_to_base[k+1] < 0) {
                                            // まだ未確定
                                            continue;
                                        }
                                        ruby_to_base[i] = ruby_to_base[k+1];
                                        break;
                                    }
                                    else {
                                        // ルビが途切れたので探索失敗
                                        break;
                                    }
                                }
                            }
                            continue;
                        }

                        // どちらも確定している
                        if (ruby_to_base[i-1] >= 0 && ruby_to_base[i+1] >= 0) {
                            // 違うブロックが、近接している
                            if (ruby_to_base[i-1] != ruby_to_base[i+1]) {
                                // 近い方のブロックに所属させる
                                if (dx1 < dx2) {
                                    ruby_to_base[i] = ruby_to_base[i-1];
                                }
                                else {
                                    ruby_to_base[i] = ruby_to_base[i+1];
                                }
                            }
                            else {
                                // 同じブロックなので前後どちらでも同じ
                                ruby_to_base[i] = ruby_to_base[i-1];
                            }
                            continue;
                        }

                        // 確定している方を取る
                        if (ruby_to_base[i-1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i-1];
                            continue;
                        }
                        if (ruby_to_base[i+1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i+1];
                            continue;
                        }
                    }

                    // なんか孤立しているので、おそらく誤認識
                    if (dx1 > w2 * 2 && dx2 > w2 * 2) {
                        continue;
                    }

                    // 前に繋がるふりがな
                    if (dx1 <= w2 * 2) {
                        // 確定していれば所属させる
                        if (ruby_to_base[i-1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i-1];
                            continue;
                        }

                        // 前方に検索する
                        for(int k = i-1; k > 0; k--) {
                            float dx11 = boxes[rubyid[k]].cx - boxes[rubyid[k-1]].cx;
                            if (dx11 <= w2 * 2) {
                                if(ruby_to_base[k-1] < 0) {
                                    // まだ未確定
                                    continue;
                                }
                                ruby_to_base[i] = ruby_to_base[k-1];
                                break;
                            }
                            else {
                                // ルビが途切れたので探索失敗
                                break;
                            }
                        }
                        continue;
                    }

                    // 後ろに繋がるふりがな
                    // 確定していれば所属させる
                    if (ruby_to_base[i+1] >= 0) {
                        ruby_to_base[i] = ruby_to_base[i+1];
                        continue;
                    }

                    // 後方に検索する
                    for(int k = i+1; k < rubyid.size()-1; k++) {
                        float dx22 = boxes[rubyid[k+1]].cx - boxes[rubyid[k]].cx;
                        if (dx22 <= w2 * 2) {
                            if(ruby_to_base[k+1] < 0) {
                                // まだ未確定
                                continue;
                            }
                            ruby_to_base[i] = ruby_to_base[k+1];
                            break;
                        }
                        else {
                            // ルビが途切れたので探索失敗
                            break;
                        }
                    }
                    continue;
                }
            }

            // rubybaseの直後にrubyを挿入する
            std::vector<int> baselast;
            for(int i = 0; i < rubybaseBlock.size(); i++) {
                baselast.push_back(rubybaseBlock[i].back());
            }
            for(auto boxid: baseid) {
                fixlist.push_back(boxid);
                auto p = std::find(baselast.begin(), baselast.end(), boxid);
                if(p == baselast.end()) continue;
                auto idx = std::distance(baselast.begin(), p);
                for(int i = 0; i < rubyid.size(); i++) {
                    if(ruby_to_base[i] == idx) {
                        fixlist.push_back(rubyid[i]);
                    }
                }
            }

            // ふりがながいない親文字は扱いを外す
            for(int i = 0; i < rubybaseBlock.size(); i++) {
                if(std::find(ruby_to_base.begin(), ruby_to_base.end(), i) == ruby_to_base.end()) {
                    for(auto bidx: rubybaseBlock[i]) {
                        boxes[bidx].subtype &= ~(2+4);
                    }
                }
            }

            // 親を見つけられなかったルビの処理
            for(int i = 0; i < rubyid.size(); i++) {
                if(ruby_to_base[i] == -1) {
                    // ルビ扱いを外して、行のそれっぽい場所に挿入する
                    int ridx = rubyid[i];

                    boxes[ridx].subtype &= ~(2+4);
                    boxes[ridx].subtype |= 32;
                    int x = boxes[ridx].cx;
                    std::vector<int> baselist;
                    for(auto bidx: fixlist) {
                        if((boxes[bidx].subtype & (2+4)) != 2+4) {
                            baselist.push_back(bidx);
                        }
                    }
                    auto pbefore = std::find_if(fixlist.begin(), fixlist.end(), [x,boxes](int j){ return x < boxes[j].cx - boxes[j].w / 2; });
                    fixlist.insert(pbefore, ridx);
                }
            }
        }
        else {
            // 縦書き

            double h1 = std::transform_reduce(
                baseid.begin(), 
                baseid.end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].h; });
            double h2 = std::transform_reduce(
                rubyid.begin(), 
                rubyid.end(), 
                0.0, 
                [&](double acc, double i) { return std::max(acc, i); },
                [&](int x) { return boxes[x].h; });

            // ルビの親文字のブロック座標
            std::vector<std::pair<float, float>> rubybaseBlock_y;
            for(auto rubyblock: rubybaseBlock) {
                auto y1 = boxes[rubyblock.front()].cy - h1;
                auto y2 = boxes[rubyblock.back()].cy + h1;
                rubybaseBlock_y.emplace_back(y1,y2);
            }

            // ルビは通常、ルビ付き文字のブロックの間に付いている
            std::vector<int> ruby_to_base(rubyid.size(), -1);
            for(int i = 0; i < rubyid.size(); i++) {
                int boxid = rubyid[i];
                for(int j = 0; j < rubybaseBlock_y.size(); j++) {
                    if(rubybaseBlock_y[j].first < boxes[boxid].cy && boxes[boxid].cy < rubybaseBlock_y[j].second) {
                        ruby_to_base[i] = j;
                        break;
                    }
                }
            }

            // ルビの方が長い場合などは、ブロックから外れることがある
            if(std::count(ruby_to_base.begin(), ruby_to_base.end(), -1) > 0 
                && std::count_if(ruby_to_base.begin(), ruby_to_base.end(), [](auto x){ return x >= 0; }) > 0) {
                
                for(int i = 0; i < rubyid.size(); i++) {
                    if(ruby_to_base[i] >= 0) continue;

                    // 隣のルビとの距離
                    float dy1 = INFINITY;
                    float dy2 = INFINITY;
                    if (i > 0) {
                        dy1 = boxes[rubyid[i]].cy - boxes[rubyid[i-1]].cy;
                    }
                    if (i < rubyid.size() - 1) {
                        dy2 = boxes[rubyid[i+1]].cy - boxes[rubyid[i]].cy;
                    }

                    if (dy1 <= h2 * 2 && dy2 <= h2 * 2) {
                        // 前後と連結している
                        // おそらく前後は連続のふりがな(多分どちらを選択してもよい)
                        
                        // 前後とも未確定
                        if (ruby_to_base[i-1] < 0 && ruby_to_base[i+1] < 0) {
                            // まず前方に検索する
                            for(int k = i-1; k > 0; k--) {
                                float dy11 = boxes[rubyid[k]].cy - boxes[rubyid[k-1]].cy;
                                if (dy11 <= h2 * 2) {
                                    if(ruby_to_base[k-1] < 0) {
                                        // まだ未確定
                                        continue;
                                    }
                                    ruby_to_base[i] = ruby_to_base[k-1];
                                    break;
                                }
                                else {
                                    // ルビが途切れたので探索失敗
                                    break;
                                }
                            }
                            // 前方で失敗した場合後方に検索する
                            if(ruby_to_base[i] < 0) {
                                for(int k = i+1; k < rubyid.size()-1; k++) {
                                    float dy22 = boxes[rubyid[k+1]].cy - boxes[rubyid[k]].cy;
                                    if (dy22 <= h2 * 2) {
                                        if(ruby_to_base[k+1] < 0) {
                                            // まだ未確定
                                            continue;
                                        }
                                        ruby_to_base[i] = ruby_to_base[k+1];
                                        break;
                                    }
                                    else {
                                        // ルビが途切れたので探索失敗
                                        break;
                                    }
                                }
                            }
                            continue;
                        }

                        // どちらも確定している
                        if (ruby_to_base[i-1] >= 0 && ruby_to_base[i+1] >= 0) {
                            // 違うブロックが、近接している
                            if (ruby_to_base[i-1] != ruby_to_base[i+1]) {
                                // 近い方のブロックに所属させる
                                if (dy1 < dy2) {
                                    ruby_to_base[i] = ruby_to_base[i-1];
                                }
                                else {
                                    ruby_to_base[i] = ruby_to_base[i+1];
                                }
                            }
                            else {
                                // 同じブロックなので前後どちらでも同じ
                                ruby_to_base[i] = ruby_to_base[i-1];
                            }
                            continue;
                        }

                        // 確定している方を取る
                        if (ruby_to_base[i-1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i-1];
                            continue;
                        }
                        if (ruby_to_base[i+1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i+1];
                            continue;
                        }
                    }

                    // なんか孤立しているので、おそらく誤認識
                    if (dy1 > h2 * 2 && dy2 > h2 * 2) {
                        continue;
                    }

                    // 前に繋がるふりがな
                    if (dy1 <= h2 * 2) {
                        // 確定していれば所属させる
                        if (ruby_to_base[i-1] >= 0) {
                            ruby_to_base[i] = ruby_to_base[i-1];
                            continue;
                        }

                        // 前方に検索する
                        for(int k = i-1; k > 0; k--) {
                            float dy11 = boxes[rubyid[k]].cy - boxes[rubyid[k-1]].cy;
                            if (dy11 <= h2 * 2) {
                                if(ruby_to_base[k-1] < 0) {
                                    // まだ未確定
                                    continue;
                                }
                                ruby_to_base[i] = ruby_to_base[k-1];
                                break;
                            }
                            else {
                                // ルビが途切れたので探索失敗
                                break;
                            }
                        }
                        continue;
                    }

                    // 後ろに繋がるふりがな
                    // 確定していれば所属させる
                    if (ruby_to_base[i+1] >= 0) {
                        ruby_to_base[i] = ruby_to_base[i+1];
                        continue;
                    }

                    // 後方に検索する
                    for(int k = i+1; k < rubyid.size()-1; k++) {
                        float dy22 = boxes[rubyid[k+1]].cy - boxes[rubyid[k]].cy;
                        if (dy22 <= h2 * 2) {
                            if(ruby_to_base[k+1] < 0) {
                                // まだ未確定
                                continue;
                            }
                            ruby_to_base[i] = ruby_to_base[k+1];
                            break;
                        }
                        else {
                            // ルビが途切れたので探索失敗
                            break;
                        }
                    }
                    continue;
                }
            }

            // rubybaseの直後にrubyを挿入する
            std::vector<int> baselast;
            for(int i = 0; i < rubybaseBlock.size(); i++) {
                baselast.push_back(rubybaseBlock[i].back());
            }
            for(auto boxid: baseid) {
                fixlist.push_back(boxid);
                auto p = std::find(baselast.begin(), baselast.end(), boxid);
                if(p == baselast.end()) continue;
                auto idx = std::distance(baselast.begin(), p);
                for(int i = 0; i < rubyid.size(); i++) {
                    if(ruby_to_base[i] == idx) {
                        fixlist.push_back(rubyid[i]);
                    }
                }
            }

            // ふりがながいない親文字は扱いを外す
            for(int i = 0; i < rubybaseBlock.size(); i++) {
                if(std::find(ruby_to_base.begin(), ruby_to_base.end(), i) == ruby_to_base.end()) {
                    for(auto bidx: rubybaseBlock[i]) {
                        boxes[bidx].subtype &= ~(2+4);
                    }
                }
            }

            // 親を見つけられなかったルビの処理
            for(int i = 0; i < rubyid.size(); i++) {
                if(ruby_to_base[i] == -1) {
                    // ルビ扱いを外して、行のそれっぽい場所に挿入する
                    int ridx = rubyid[i];

                    boxes[ridx].subtype &= ~(2+4);
                    boxes[ridx].subtype |= 32;
                    int x = boxes[ridx].cx;
                    std::vector<int> baselist;
                    for(auto bidx: fixlist) {
                        if((boxes[bidx].subtype & (2+4)) != 2+4) {
                            baselist.push_back(bidx);
                        }
                    }
                    auto pbefore = std::find_if(fixlist.begin(), fixlist.end(), [x,boxes](int j){ return x < boxes[j].cy - boxes[j].h / 2; });
                    fixlist.insert(pbefore, ridx);
                }
            }
        }

        // 付番する
        int subidx = 0;
        for(auto boxid: fixlist) {
            if (horizontal) {
                boxes[boxid].subtype &= ~1;
            }
            else {
                boxes[boxid].subtype |= 1;
            }
            boxes[boxid].idx = chainid;
            boxes[boxid].subidx = subidx++;
        }
    }

    // この時点で列に入っていないルビは無視する
    for(auto &box: boxes) {
        if (box.idx < 0) {
            box.subtype &= ~(2+4);
        }
    }
}
