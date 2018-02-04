#include "core/geometry.hpp"
#include "utils/trim.hpp"

#include <fstream>
#include <algorithm>

#include <iostream>


namespace core {
namespace geometry {

void set_neighbor_bits(ivec2 const& size, std::vector<std::uint8_t>& data) {
    auto idx = [&](ivec2 pos) -> std::size_t {
        return pos.y * size.x + pos.x;
    };

    auto is_fluid = [&](ivec2 pos) -> bool {
        return (data[idx(pos)] & CELL_MASK_SELF) == static_cast<std::uint8_t>(CellType::Fluid);
    };

    for (int_t y = 0; y < size.y; y++) {
        for (int_t x = 0; x < size.x; x++) {
            std::uint8_t neighbors = 0b0000;

            // left neighbor
            if (x != 0 && is_fluid({x - 1, y})) {
                neighbors |= CELL_MASK_NEIGHBOR_LEFT;
            }

            // right neighbor
            if (x != (size.x - 1) && is_fluid({x + 1, y})) {
                neighbors |= CELL_MASK_NEIGHBOR_RIGHT;
            }

            // bottom neighbor
            if (y != 0 && is_fluid({x, y - 1})) {
                neighbors |= CELL_MASK_NEIGHBOR_BOTTOM;
            }

            // top neighbor
            if (y != (size.y - 1) && is_fluid({x, y + 1})) {
                neighbors |= CELL_MASK_NEIGHBOR_TOP;
            }

            // clear and set
            data[idx({x, y})] = (data[idx({x, y})] & CELL_MASK_SELF) | neighbors;
        }
    }
}

}   /* namespace geometry */

void Geometry::make_lid_driven_cavity() {
    auto idx = [&](ivec2 pos) -> std::size_t {
        return pos.y * m_size.x + pos.x;
    };

    std::fill(m_data.begin(), m_data.end(), geometry::cell_type_to_bits(CellType::Fluid));

    // set left boundary: no-slip
    for (int_t y = 0; y < m_size.y; y++) {
        m_data[idx({0, y})] = geometry::cell_type_to_bits(CellType::NoSlip);
    }

    // set right boundary: no-slip
    for (int_t y = 0; y < m_size.y; y++) {
        m_data[idx({m_size.x - 1, y})] = geometry::cell_type_to_bits(CellType::NoSlip);
    }

    // set bottom boundary: no-slip
    for (int_t x = 0; x < m_size.x; x++) {
        m_data[idx({x, 0})] = geometry::cell_type_to_bits(CellType::NoSlip);
    }

    // set top boundary: horizontal velocity inflow
    for (int_t x = 0; x < m_size.x; x++) {
        m_data[idx({x, m_size.y - 1})] = geometry::cell_type_to_bits(CellType::InflowHoriz);
    }

    geometry::set_neighbor_bits(m_size, m_data);
}

auto Geometry::lid_driven_cavity(ivec2 size, rvec2 length, real_t u) -> Geometry {
    auto geom = Geometry{size, length, {u, 0.0}, 0.0, std::vector<uint8_t>(size.x * size.y)};
    geom.make_lid_driven_cavity();

    return geom;
}

void Geometry::load(char const* file) {
    // load file
    std::ifstream in;
    in.open(file);

    std::vector<char> freeform;

    while (in) {
        std::string line;
        if (!std::getline(in, line)) { break; }

        std::stringstream tokenstr{line};
        std::string key;
        if (!std::getline(tokenstr, key, '=')) { continue; };
        key = utils::trim(key);

        if (key == "size") {
            tokenstr >> m_size.x;
            tokenstr >> m_size.y;
        }

        if (key == "length") {
            tokenstr >> m_length.x;
            tokenstr >> m_length.y;
        }

        if (key == "velocity") {
            tokenstr >> m_velocity.x;
            tokenstr >> m_velocity.y;
        }

        if (key == "pressure") {
            tokenstr >> m_pressure;
        }

        if (key == "geometry") {
            std::string value;
            tokenstr >> value;
            value = utils::trim(value);

            if (value == "free") {      // load 'm_size.y' lines
                std::string line;

                for (int_t i = 0; i < m_size.y; i++) {
                    if (!std::getline(in, line)) { break; }
                    std::copy(line.begin(), line.end(), std::back_inserter(freeform));
                }
            }
        }
    }

    // update mesh-width
    m_mesh.x = m_length.x / m_size.x;
    m_mesh.y = m_length.y / m_size.y;

    // set data bitfield to free-form
    if (!freeform.empty()) {
        // check validity
        if (static_cast<std::size_t>(m_size.x * m_size.y) != freeform.size()) {
            throw std::invalid_argument{"Geometry size does not match free-form data"};
        }

        // transform charecter-data to bit-field
        std::vector<std::uint8_t> data(m_size.x * m_size.y, geometry::cell_type_to_bits(CellType::Fluid));
        for (int_t y = 0; y < m_size.y; y++) {
            for (int_t x = 0; x < m_size.x; x++) {
                int_t idx_chars = (m_size.y - y - 1) * m_size.x + x;
                int_t idx_cells = y * m_size.x + x;

                char c = freeform[idx_chars];
                data[idx_cells] = geometry::cell_type_to_bits(geometry::cell_type_from_char(c));
            }
        }

        m_data = std::move(data);

        // reset neighbor bits
        geometry::set_neighbor_bits(m_size, m_data);

    // else make/update the lid driven cavity form
    } else {
        make_lid_driven_cavity();
    }
}

auto Geometry::num_fluid_cells() const -> uint_t {
    return std::count_if(m_data.begin(), m_data.end(), [](std::uint8_t c) {
        return (c & CELL_MASK_SELF) == geometry::cell_type_to_bits(CellType::Fluid);
    });
}

}   /* namespace core */