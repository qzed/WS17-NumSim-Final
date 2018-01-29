#include "sdl/opengl/window.hpp"

#include <iostream>
#include <iomanip>


int main(int argc, char** argv) {
    auto window = sdl::opengl::Window::builder("Numerical Simulations Course 2017/18", 800, 600)
        .set(SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        .set(SDL_GL_CONTEXT_MINOR_VERSION, 2)
        .set(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
        .set(SDL_GL_DOUBLEBUFFER, 1)
        .set(SDL_WINDOW_RESIZABLE)
        .build();

    bool running = true;
    while (running) {
        SDL_Event e;

        // handle input
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {   // received on SIGINT or when all windows have been closed
                running = false;
                std::cout << "Terminating\n";
            }

            else if (e.type == SDL_WINDOWEVENT && e.window.windowID == window.id()) {
                if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
                    window.hide();      // hide on close
                }
            }

            else if (e.type == SDL_KEYDOWN && e.key.windowID == window.id()) {
                std::cout << "Key pressed: "
                    << "0x" << std::hex << std::setfill('0') << std::setw(2) << e.key.keysym.sym
                    << "\n";
            }
        }

        window.swap_buffers();
    }
}
