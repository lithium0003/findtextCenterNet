#pragma once

#include "line_detect.h"
#include <vector>

int number_unbind(
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage,
    int next_id);
