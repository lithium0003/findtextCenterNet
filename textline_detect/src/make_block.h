#pragma once

#include "line_detect.h"

void make_block(
    int id_max,
    std::vector<charbox> &boxes,
    const std::vector<bool> &lineblocker);
