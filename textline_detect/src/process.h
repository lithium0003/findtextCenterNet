//
//  process.h
//  linedetector
//
//  Created by rei9 on 2025/03/03.
//

#ifndef process_h
#define process_h

#include <vector>

void process(
    const std::vector<float> &lineimage,
    const std::vector<float> &sepimage,
    std::vector<charbox> &boxes);

void print_chaininfo(
    const std::vector<charbox> &boxes,
    const std::vector<std::vector<int>> &line_box_chain);

#endif /* process_h */
