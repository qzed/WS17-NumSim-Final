#pragma once


namespace utils {

template <typename T>
class Cached {
public:
    inline Cached();
    inline Cached(Cached<T> const& t);
    inline Cached(Cached<T>&& t);
    inline Cached(T const& t);
    inline Cached(T&& t);

    inline auto operator= (Cached<T> const& t) -> Cached<T>&;
    inline auto operator= (Cached<T>&& t) -> Cached<T>&;

    inline auto operator= (T const& t) -> Cached<T>&;
    inline auto operator= (T&& t) -> Cached<T>&;

    inline auto set(T const& t) -> Cached<T>&;
    inline auto set(T&& t) -> Cached<T>&;

    inline auto get() const -> T const&;
    inline auto get() -> T&;

    template <typename F>
    inline void when_dirty(F f);

private:
    T m_value;
    bool m_dirty;
};

template <typename T>
Cached<T>::Cached()
    : m_value{}, m_dirty{true} {}

template <typename T>
Cached<T>::Cached(Cached<T> const& other)
    : m_value{other.m_value}, m_dirty{true} {}

template <typename T>
Cached<T>::Cached(Cached<T>&& other)
    : m_value{std::move(other.m_value)}, m_dirty{true} {}

template <typename T>
Cached<T>::Cached(T const& t)
    : m_value{t}, m_dirty{true} {}

template <typename T>
Cached<T>::Cached(T&& t)
    : m_value{std::move(t)}, m_dirty{true} {}

template <typename T>
auto Cached<T>::operator= (Cached<T> const& t) -> Cached<T>& {
    m_value = t;
    m_dirty = true;
}

template <typename T>
auto Cached<T>::operator= (Cached<T>&& t) -> Cached<T>& {
    m_value = std::move(t);
    m_dirty = true;
    return *this;
}

template <typename T>
auto Cached<T>::operator= (T const& t) -> Cached<T>& {
    m_value = t;
    m_dirty = true;
    return *this;
}

template <typename T>
auto Cached<T>::operator= (T&& t) -> Cached<T>& {
    m_value = std::move(t);
    m_dirty = true;
    return *this;
}

template <typename T>
auto Cached<T>::set(T const& t) -> Cached<T>& {
    m_value = t;
    m_dirty = true;
    return *this;
}

template <typename T>
auto Cached<T>::set(T&& t) -> Cached<T>& {
    m_value = std::move(t);
    m_dirty = true;
    return *this;
}

template <typename T>
auto Cached<T>::get() const -> T const& {
    return m_value;
}

template <typename T>
auto Cached<T>::get() -> T& {
    return m_value;
}

template <typename T>
template <typename Fn>
void Cached<T>::when_dirty(Fn fn) {
    if (m_dirty) {
        fn(m_value);
        m_dirty = false;
    }
}

}   /* namespace utils */
