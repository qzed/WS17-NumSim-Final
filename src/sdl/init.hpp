//! Facilities to safely initialize SDL2 across multiple modules.
//!

#pragma once

#include "sdl/sdl.hpp"
#include "sdl/errors.hpp"

#include <cstdint>
#include <memory>
#include <mutex>


namespace sdl {
namespace detail {

class InitState {
public:
    inline InitState();
    inline ~InitState();

    inline void push(std::uint32_t flags);
    inline void pop(std::uint32_t flags);

private:
    std::mutex m_mutex;

    bool m_init;
    std::uint32_t m_timer;
    std::uint32_t m_audio;
    std::uint32_t m_video;
    std::uint32_t m_joystick;
    std::uint32_t m_haptic;
    std::uint32_t m_gamecontroller;
    std::uint32_t m_events;
};


InitState::InitState()
    : m_init{false}
    , m_timer{0}
    , m_audio{0}
    , m_video{0}
    , m_joystick{0}
    , m_haptic{0}
    , m_gamecontroller{0}
    , m_events{0} {}

InitState::~InitState() {
    if (m_init) {
        SDL_Quit();
    }
}

void InitState::push(std::uint32_t flags) {
    std::lock_guard<std::mutex> guard{m_mutex};

    std::uint32_t init_flags = 0;

    if (flags & SDL_INIT_TIMER) {
        if (m_timer == 0) { init_flags |= SDL_INIT_TIMER; }
        m_timer++;
    }

    if (flags & SDL_INIT_AUDIO) {
        if (m_audio == 0) { init_flags |= SDL_INIT_AUDIO; }
        m_audio++;
    }

    if (flags & SDL_INIT_VIDEO) {
        if (m_video == 0) { init_flags |= SDL_INIT_VIDEO; }
        m_video++;
    }

    if (flags & SDL_INIT_JOYSTICK) {
        if (m_joystick == 0) { init_flags |= SDL_INIT_JOYSTICK; }
        m_joystick++;
    }

    if (flags & SDL_INIT_HAPTIC) {
        if (m_haptic == 0) { init_flags |= SDL_INIT_HAPTIC; }
        m_haptic++;
    }

    if (flags & SDL_INIT_GAMECONTROLLER) {
        if (m_gamecontroller == 0) { init_flags |= SDL_INIT_GAMECONTROLLER; }
        m_gamecontroller++;
    }

    if (flags & SDL_INIT_EVENTS) {
        if (m_events == 0) { init_flags |= SDL_INIT_EVENTS; }
        m_events++;
    }

    m_init = true;
    sdl::except(SDL_InitSubSystem(init_flags));
}

void InitState::pop(std::uint32_t flags) {
    std::lock_guard<std::mutex> guard{m_mutex};

    std::uint32_t init_flags = 0;

    if ((flags & SDL_INIT_TIMER) && m_timer > 0) {
        if (m_timer == 0) { init_flags |= SDL_INIT_TIMER; }
        m_timer--;
    }

    if ((flags & SDL_INIT_AUDIO) && m_audio > 0) {
        m_audio--;
        if (m_audio == 0) { init_flags |= SDL_INIT_AUDIO; }
    }

    if ((flags & SDL_INIT_VIDEO) && m_video > 0) {
        m_video--;
        if (m_video == 0) { init_flags |= SDL_INIT_VIDEO; }
    }

    if ((flags & SDL_INIT_JOYSTICK) && m_joystick > 0) {
        m_joystick--;
        if (m_joystick == 0) { init_flags |= SDL_INIT_JOYSTICK; }
    }

    if ((flags & SDL_INIT_HAPTIC) && m_haptic > 0) {
        m_haptic--;
        if (m_haptic == 0) { init_flags |= SDL_INIT_HAPTIC; }
    }

    if ((flags & SDL_INIT_GAMECONTROLLER) && m_gamecontroller > 0) {
        m_gamecontroller--;
        if (m_gamecontroller == 0) { init_flags |= SDL_INIT_GAMECONTROLLER; }
    }

    if ((flags & SDL_INIT_EVENTS) && m_events > 0) {
        m_events--;
        if (m_events == 0) { init_flags |= SDL_INIT_EVENTS; }
    }

    if (init_flags != 0) {
        SDL_QuitSubSystem(init_flags);
    }
}

}   /* namespace detail */

class InitGuard {
    friend auto init(std::uint32_t flags) -> InitGuard;

public:
    inline InitGuard(InitGuard const&) = delete;
    inline InitGuard(InitGuard&&);

    inline auto operator= (InitGuard const&) const -> InitGuard& = delete;
    inline auto operator= (InitGuard&&) -> InitGuard&;

    inline ~InitGuard();

    inline auto clone() const -> InitGuard;

private:
    InitGuard(std::shared_ptr<detail::InitState> state, std::uint32_t flags);

private:
    std::shared_ptr<detail::InitState> m_state;
    std::uint32_t m_flags;
};


InitGuard::InitGuard(std::shared_ptr<detail::InitState> state, std::uint32_t flags)
    : m_state{std::move(state)}
    , m_flags{flags}
{
    m_state->push(m_flags);
}

InitGuard::InitGuard(InitGuard&& from)
    : m_state{std::move(from.m_state)}
    , m_flags{from.m_flags} {}

InitGuard::~InitGuard() {
    if (m_state) {
        m_state->pop(m_flags);
    }
}

auto InitGuard::operator= (InitGuard&& from) -> InitGuard& {
    m_flags = from.m_flags;
    m_state = std::move(from.m_state);
    return *this;
}

auto InitGuard::clone() const -> InitGuard {
    return InitGuard{m_state, m_flags};
}


inline auto init(std::uint32_t flags) -> InitGuard {
    static auto state = std::make_shared<detail::InitState>();
    return InitGuard{state, flags};
}

}   /* namespace sdl */
