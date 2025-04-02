#pragma once

#include "line_detect.h"
#include <vector>

void make_block(
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker);
