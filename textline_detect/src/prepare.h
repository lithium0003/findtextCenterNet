#pragma once

#include "line_detect.h"
#include <vector>

void prepare_id_image(
    std::vector<int> &idimage,
    std::vector<int> &idimage_main,
    std::vector<charbox> &boxes);

void make_lineblocker(
    std::vector<bool> &lineblocker,
    const std::vector<float> &sepimage);
