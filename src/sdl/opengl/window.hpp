#pragma once

#include "opengl/opengl.hpp"
#include "opengl/errors.hpp"

#include "sdl/errors.hpp"
#include "sdl/init.hpp"

#include <SDL2/SDL.h>

#include <vector>
#include <cinttypes>
#include <limits>


namespace sdl {
namespace opengl {

class WindowBuilder;


class Context {
public:
    inline Context(SDL_GLContext handle);
    inline Context(Context const&) = delete;
    inline Context(Context&& other);

    inline ~Context();

    inline auto operator= (Context const&) -> Context& = delete;
    inline auto operator= (Context&& other) -> Context&;

    inline auto handle() const -> SDL_GLContext;
    inline auto operator* () const -> SDL_GLContext;

private:
    SDL_GLContext m_handle;
};


class Window {
public:
    static inline auto builder(std::string title, int width, int height) -> WindowBuilder;

    inline Window(sdl::InitGuard init, SDL_Window* handle, Context context);
    inline Window(Window const&) = delete;
    inline Window(Window&& other);

    inline ~Window();

    inline auto operator= (Window const&) -> Window& = delete;
    inline auto operator= (Window&& other) -> Window&;

    inline auto handle() const -> SDL_Window*;
    inline auto operator* () const -> SDL_Window*;

    inline auto id() const -> std::uint32_t;

    inline auto context() const -> Context const&;

    inline void swap_buffers() const;

    inline void show() const;
    inline void hide() const;
    inline void raise() const;

private:
    sdl::InitGuard m_sdl_init;
    SDL_Window* m_handle;
    Context m_context;

    std::uint32_t m_window_id;
};


class WindowBuilder {
public:
    inline WindowBuilder(std::string title, int width, int height);

    inline auto position(int x, int y) -> WindowBuilder&;

    inline auto set(std::uint32_t flag) -> WindowBuilder&;
    inline auto unset(std::uint32_t flag) -> WindowBuilder&;

    inline auto set(SDL_GLattr attribute, int value) -> WindowBuilder&;

    inline auto build() const -> Window;

private:
    std::string m_title;
    std::pair<int, int> m_size;
    std::pair<int, int> m_pos;

    std::uint32_t m_flags;
    std::vector<std::pair<SDL_GLattr, int>> m_gl_attribs;
};


Context::Context(SDL_GLContext handle)
    : m_handle{handle} {}

Context::Context(Context&& other)
    : m_handle{std::exchange(other.m_handle, nullptr)} {}

Context::~Context() {
    if (m_handle != nullptr) {
        SDL_GL_DeleteContext(m_handle);
    }
}

auto Context::operator= (Context&& other) -> Context& {
    m_handle = std::exchange(other.m_handle, nullptr);
    return *this;
}

auto Context::handle() const -> SDL_GLContext {
    return m_handle;
}

auto Context::operator* () const -> SDL_GLContext {
    return m_handle;
}


auto Window::builder(std::string title, int width, int height) -> WindowBuilder {
    return WindowBuilder{std::move(title), width, height};
}

Window::Window(sdl::InitGuard init, SDL_Window* handle, Context context)
    : m_sdl_init{std::move(init)}
    , m_handle{handle}
    , m_context{std::move(context)}
    , m_window_id{SDL_GetWindowID(handle)} {}

Window::Window(Window&& other)
    : m_sdl_init{std::move(other.m_sdl_init)}
    , m_handle{std::exchange(other.m_handle, nullptr)}
    , m_context{std::move(other.m_context)}
    , m_window_id{std::exchange(other.m_window_id, std::numeric_limits<std::uint32_t>::max())} {}

Window::~Window() {
    if (m_handle != nullptr) {
        SDL_DestroyWindow(m_handle);
    }
}

auto Window::operator= (Window&& other) -> Window& {
    m_handle = std::exchange(other.m_handle, nullptr);
    m_context = std::move(other.m_context);
    m_sdl_init = std::move(other.m_sdl_init);

    m_window_id = std::exchange(other.m_window_id, std::numeric_limits<std::uint32_t>::max());

    return *this;
}

auto Window::handle() const -> SDL_Window* {
    return m_handle;
}

auto Window::operator* () const -> SDL_Window* {
    return m_handle;
}

auto Window::id() const -> std::uint32_t {
    return m_window_id;
}

auto Window::context() const -> Context const& {
    return m_context;
}

void Window::swap_buffers() const {
    SDL_GL_SwapWindow(m_handle);
}

void Window::hide() const {
    SDL_HideWindow(m_handle);
}

void Window::show() const {
    SDL_ShowWindow(m_handle);
}

void Window::raise() const {
    SDL_RaiseWindow(m_handle);
}


WindowBuilder::WindowBuilder(std::string title, int width, int height)
    : m_title{std::move(title)}
    , m_size{width, height}
    , m_pos{SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED}
    , m_flags{SDL_WINDOW_OPENGL} {}

auto WindowBuilder::position(int x, int y) -> WindowBuilder& {
    m_pos = {x, y};
    return *this;
}

auto WindowBuilder::set(std::uint32_t flag) -> WindowBuilder& {
    m_flags |= flag;
    return *this;
}

auto WindowBuilder::unset(std::uint32_t flag) -> WindowBuilder& {
    m_flags &= ~flag;
    return *this;
}

auto WindowBuilder::set(SDL_GLattr attribute, int value) -> WindowBuilder& {
    m_gl_attribs.emplace_back(attribute, value);
    return *this;
}

auto WindowBuilder::build() const -> Window {
    // initialize sdl video subsystem
    auto sdl_init = sdl::init(SDL_INIT_VIDEO);

    // set attributes for window and context creation
    SDL_GL_ResetAttributes();

    for (auto const& attrib : m_gl_attribs) {
        sdl::except(SDL_GL_SetAttribute(attrib.first, attrib.second));
    }

    // crate window
    auto window = [&](){
        auto window_ptr = SDL_CreateWindow(
            m_title.c_str(),
            m_pos.first, m_pos.second,
            m_size.first, m_size.second,
            m_flags
        );

        return std::unique_ptr<SDL_Window, void(*)(SDL_Window*)>{window_ptr, SDL_DestroyWindow};
    }();

    // create context
    auto context = Context{sdl::except_null(SDL_GL_CreateContext(window.get()))};

    // reset state
    SDL_GL_ResetAttributes();

    return {std::move(sdl_init), window.release(), std::move(context)};
}

}   /* namespace opengl */
}   /* namespace sdl */
