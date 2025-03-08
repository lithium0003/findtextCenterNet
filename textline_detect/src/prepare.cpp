#include "prepare.h"
#include <cstdio>
#include <vector>

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

void prepare_id_image(
    std::vector<int> &idimage,
    std::vector<int> &idimage_main,
    std::vector<charbox> &boxes)
{
    fprintf(stderr, "prepare_id_image\n");
    idimage.resize(width*height, -1);
    idimage_main.resize(width*height, -1);
    for(const auto &box: boxes) {
        //fprintf(stderr, "box %d cx %f cy %f w %f h %f c1 %f c2 %f c4 %f c8 %f\n", box.id, box.cx, box.cy, box.w, box.h, box.code1, box.code2, box.code4, box.code8);
        int left = (box.cx - box.w / 2) / 4;
        int right = (box.cx + box.w / 2) / 4 + 1;
        int top = (box.cy - box.h / 2) / 4;
        int bottom = (box.cy + box.h / 2) / 4 + 1;
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
}

void make_lineblocker(
    std::vector<bool> &lineblocker,
    const std::vector<float> &sepimage)
{
    fprintf(stderr, "make_lineblocker\n");
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
}
