#include "core/parameters.hpp"
#include "utils/trim.hpp"

#include <fstream>
#include <sstream>
#include <iostream>


namespace core {

void Parameters::load(char const* file) {
    std::ifstream in;
    in.open(file);

    while (in) {
        std::string line;
        if (!std::getline(in, line)) { break; }

        std::stringstream tokenstr{line};
        std::string key;
        if (!std::getline(tokenstr, key, '=')) { continue; };
        utils::trim(key);

        if (key == "re") {
            tokenstr >> re;
        } else if (key == "omg") {
            tokenstr >> omega;
        } else if (key == "alpha") {
            tokenstr >> alpha;
        } else if (key == "dt") {
            tokenstr >> dt;
        } else if (key == "tend") {
            tokenstr >> t_end;
        } else if (key == "iter") {
            tokenstr >> itermax;
        } else if (key == "eps") {
            tokenstr >> eps;
        } else if (key == "tau") {
            tokenstr >> tau;
        } else {
            std::cout << "WARNING: unknown key `" << key << "` in file `" << file << "`\n";
        }
    }
}

}   /* namespace core */
