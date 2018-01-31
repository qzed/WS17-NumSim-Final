#pragma once

#include <cstdint>


namespace utils {

class Resource {
public:
    inline Resource(std::uint8_t const* data, std::size_t len);
    inline Resource(Resource const& other);

    template<std::size_t N>
    explicit inline Resource(const std::uint8_t(&data)[N]);

    inline auto operator= (Resource const& other) -> Resource&;

    inline auto data() const -> std::uint8_t const*;
    inline auto size() const -> std::size_t;

    inline auto begin() const -> std::uint8_t const*;
    inline auto cbegin() const -> std::uint8_t const*;

    inline auto end() const -> std::uint8_t const*;
    inline auto cend() const -> std::uint8_t const*;

private:
    std::uint8_t const* m_data;
    std::size_t m_len;
};


Resource::Resource(std::uint8_t const* data, std::size_t len)
    : m_data{data}, m_len{len} {}

Resource::Resource(Resource const& other)
    : m_data{other.m_data}, m_len{other.m_len} {}

template<std::size_t N>
Resource::Resource(const std::uint8_t(&data)[N])
    : m_data{data}, m_len{sizeof(data)} {}

auto Resource::operator= (Resource const& other) -> Resource& {
    m_data = other.m_data;
    m_len = other.m_len;
    return *this;
}

auto Resource::data() const -> std::uint8_t const* {
    return m_data;
}

auto Resource::size() const -> std::size_t {
    return m_len;
}

auto Resource::begin() const -> std::uint8_t const* {
    return m_data;
}

auto Resource::cbegin() const -> std::uint8_t const* {
    return m_data;
}

auto Resource::end() const -> std::uint8_t const* {
    return m_data + m_len;
}

auto Resource::cend() const -> std::uint8_t const* {
    return m_data + m_len;
}

}   /* namespace utils */