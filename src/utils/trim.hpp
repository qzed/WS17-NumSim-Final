#pragma once

#include <string>
#include <locale>
#include <algorithm>


namespace utils {

inline auto trim(std::string str) -> std::string {
    // trim start
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int c) {
        return !std::isspace(c);
    }));

    // trim end
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int c) {
        return !std::isspace(c);
    }).base(), str.end());

    return str;
}

}   /* namespace utils */
