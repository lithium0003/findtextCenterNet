#include "line_detect.h"
#include "process.h"
#include "prepare.h"
#include "hough_linefind.h"
#include "search_loop.h"
#include "after_search.h"
#include "space_check.h"

#include <stdio.h>
#include <iostream>
#include <iterator>
#include <cmath>

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
            fprintf(stderr, "    %d %d %d, %d %d %d, %f x %f y %f w %f h %f t %d\n",
                i, j, boxid,
                boxes[boxid].idx, boxes[boxid].subidx, boxes[boxid].subtype, 
                boxes[boxid].direction / M_PI * 180, 
                boxes[boxid].cx, boxes[boxid].cy, boxes[boxid].w, boxes[boxid].h,
                boxes[boxid].subtype);
        }
    }
    fprintf(stderr, "****************\n");
}

void process(
    const std::vector<float> &lineimage, 
    const std::vector<float> &sepimage,
    std::vector<charbox> &boxes)
{
    std::vector<int> idimage;
    std::vector<int> idimage_main;
    prepare_id_image(idimage, idimage_main, boxes);

    std::vector<bool> lineblocker;
    make_lineblocker(lineblocker, sepimage);

    auto line_box_chain = linefind(boxes, lineimage, lineblocker);
    // print_chaininfo(boxes, line_box_chain);

    search_loop(boxes, line_box_chain, lineblocker, idimage_main, sepimage);
    // //print_chaininfo(boxes, line_box_chain);

    after_search(boxes, line_box_chain, lineblocker, idimage);

    space_chack(boxes);
}
