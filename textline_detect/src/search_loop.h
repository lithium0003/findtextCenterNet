#pragma once

#include "line_detect.h"
#include <vector>

void sort_chain(
    std::vector<int> &chain,
    const std::vector<charbox> &boxes);

void fix_chain_info(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain);

void search_chain(
    const std::vector<int> &chain,
    const std::vector<charbox> &boxes,
    float &direction,
    double &w, double &h,
    float &start_cx, float &start_cy, 
    float &end_cx, float &end_cy);

void make_track_line(
    std::vector<int> &x,
    std::vector<int> &y,
    float &direction,
    double &w, double &h,
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    int chainid,
    int extra_len = 0);

std::vector<int> create_chainid_map(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    double ratio = 1.0,
    int extra_len = 0);

void search_loop(
    std::vector<charbox> &boxes,
    std::vector<std::vector<int>> &line_box_chain,
    const std::vector<bool> &lineblocker,
    const std::vector<int> &idimage,
    const std::vector<float> &sepimage);
