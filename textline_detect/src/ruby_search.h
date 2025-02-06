#pragma once

#include "line_detect.h"

void search_ruby(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage);
