#pragma once

#include "types.hpp"

#include <sstream>
#include <bitset>
#include <stdexcept>
#include <vector>


namespace core {

enum class CellType : std::uint8_t {
    Fluid       = 0b0000,
    NoSlip      = 0b1100,
    Inflow      = 0b1101,
    InflowHoriz = 0b0101,
    InflowVert  = 0b1001,
    SlipHoriz   = 0b0110,
    SlipVert    = 0b1010,
    Outflow     = 0b1110,
};

const std::uint8_t CELL_MASK_NEIGHBOR_LEFT   = 0b10000000;
const std::uint8_t CELL_MASK_NEIGHBOR_RIGHT  = 0b01000000;
const std::uint8_t CELL_MASK_NEIGHBOR_BOTTOM = 0b00100000;
const std::uint8_t CELL_MASK_NEIGHBOR_TOP    = 0b00010000;
const std::uint8_t CELL_MASK_SELF            = 0b00001111;


namespace geometry {

inline auto cell_type_to_bits(CellType type) -> std::uint8_t;
inline auto cell_type_from_char(char c) -> CellType;
inline auto cell_type_to_char(CellType type) -> char;

void set_neighbor_bits(ivec2 const& size, std::vector<std::uint8_t>& data);

}   /* namespace geometry */


class Geometry {
public:
    static auto lid_driven_cavity(ivec2 size = {128, 128}, rvec2 length = {1.0, 1.0}, real_t u = 1.0)
        -> Geometry;

    inline Geometry(ivec2 size, rvec2 length, rvec2 velocity, real_t pressure, std::vector<std::uint8_t> data);
    inline Geometry(Geometry const&) = default;
    inline Geometry(Geometry&&) = default;

    inline auto operator=(Geometry const&) -> Geometry& = default;
    inline auto operator=(Geometry&&) -> Geometry& = default;

    void load(char const* file);

    inline auto size() const -> ivec2 const&;
    inline auto mesh() const -> rvec2 const&;
    inline auto length() const -> rvec2 const&;

    inline auto boundary_pressure() const -> real_t;
    inline auto boundary_velocity() const -> rvec2 const&;

    inline auto data() const -> std::vector<std::uint8_t> const&;

private:
    void make_lid_driven_cavity();

private:
    ivec2 m_size;
    rvec2 m_mesh;
    rvec2 m_length;

    rvec2 m_velocity;
    real_t m_pressure;

    std::vector<std::uint8_t> m_data;
};


namespace geometry {

inline auto cell_type_to_bits(CellType type) -> std::uint8_t {
    return static_cast<std::uint8_t>(type);
}

inline auto cell_type_from_char(char c) -> CellType {
    switch (c) {
    case ' ': return CellType::Fluid;
    case '#': return CellType::NoSlip;
    case 'I': return CellType::Inflow;
    case 'H': return CellType::InflowHoriz;
    case 'V': return CellType::InflowVert;
    case 'O': return CellType::Outflow;
    case '-': return CellType::SlipHoriz;
    case '|': return CellType::SlipVert;
    }

    std::stringstream msg;
    msg << "Character `" << c << "` is not a valid cell type";
    throw std::invalid_argument(msg.str());
}

inline auto cell_type_to_char(CellType type) -> char {
    switch (type) {
    case CellType::Fluid:       return ' ';
    case CellType::NoSlip:      return '#';
    case CellType::Inflow:      return 'I';
    case CellType::InflowHoriz: return 'H';
    case CellType::InflowVert:  return 'V';
    case CellType::Outflow:     return 'O';
    case CellType::SlipHoriz:   return '-';
    case CellType::SlipVert:    return '|';
    }

    std::stringstream msg;
    msg << "Invalid cell type `0b" << std::bitset<8>(static_cast<std::uint8_t>(type)) << "`";
    throw std::invalid_argument(msg.str());
}

}   /* namespace geometry */


Geometry::Geometry(ivec2 size, rvec2 length, rvec2 velocity, real_t pressure, std::vector<std::uint8_t> data)
    : m_size{size}
    , m_mesh{length.x / size.x, length.y / size.y}
    , m_length{length}
    , m_velocity{velocity}
    , m_pressure{pressure}
    , m_data{std::move(data)} {}

auto Geometry::size() const -> ivec2 const& {
    return m_size;
}

auto Geometry::mesh() const -> rvec2 const& {
    return m_mesh;
}

auto Geometry::length() const -> rvec2 const& {
    return m_length;
}

auto Geometry::boundary_pressure() const -> real_t {
    return m_pressure;
}

auto Geometry::boundary_velocity() const -> rvec2 const& {
    return m_velocity;
}

auto Geometry::data() const -> std::vector<std::uint8_t> const& {
    return m_data;
}

}   /* namespace core */
