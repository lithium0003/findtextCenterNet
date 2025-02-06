#pragma once

#include "line_detect.h"

std::vector<std::vector<int>> linefind(
    std::vector<charbox> &boxes,
    const std::vector<float> &lineimage, 
    const std::vector<bool> &lineblocker);
