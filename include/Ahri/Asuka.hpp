/**
 * @file Asuka.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef ASUKA_HPP
#define ASUKA_HPP

#include <cstdlib>
#include <iostream>
#include <string>

namespace Ahri {
inline std::string _get_sokyoei_data_dir() {
    auto const SOKYOEI_DATA_DIR = std::getenv("SOKYOEI_DATA_DIR");
    if (SOKYOEI_DATA_DIR) {
        return std::string(SOKYOEI_DATA_DIR);
    } else {
        std::cerr << "Please install https://github.com/Sokyoei/data" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static auto const SOKYOEI_DATA_DIR = _get_sokyoei_data_dir();
}  // namespace Ahri

#define SOKYOEI_DATA_DIR Ahri::SOKYOEI_DATA_DIR

#endif  // !ASUKA_HPP
