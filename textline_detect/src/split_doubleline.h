#pragma once

#include "line_detect.h"
#include <vector>

void split_doubleline1(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain);

void split_doubleline2(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain);

void split_doubleline3(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain);
