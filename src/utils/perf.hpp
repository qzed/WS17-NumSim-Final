#pragma once

#include "opencl/opencl.hpp"
#include "json/json.hpp"

#include <chrono>
#include <string>
#include <unordered_map>


// Note: this implementation is janky af. as it relies on the key (function name)
//       being static (const char* as map key)! Generally, the key should be the
//       static when it has been obtained via the compiler macros (__func__ etc.).


#define PERF_RECORD_START() \
    ::utils::perf::Record::start(__PRETTY_FUNCTION__)

#define PERF_RECORD_SCOPE() \
    auto _utils_perf_record = PERF_RECORD_START()


namespace utils {
namespace perf {

template <typename Duration>
struct Entry {
    size_t executions;
    Duration duration;
};


class Registry {
public:
    using duration_type = std::chrono::duration<double, std::milli>;
    using entry_type = Entry<duration_type>;
    using store_type = std::unordered_map<char const*, entry_type>;

public:
    inline static auto get() -> Registry&;

    inline auto store() -> store_type&;
    inline auto store() const -> store_type const&;

private:
    Registry() {}

private:
    store_type m_store;
};


class Record {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

public:
    inline static auto start(char const* name) -> Record;

    inline Record(Record const& other) = delete;
    inline Record(Record&& other);
    inline ~Record();

    inline auto operator= (Record const& rhs) -> Record& = delete;
    inline auto operator= (Record&& rhs) -> Record&;

    inline void stop();

private:
    inline Record(char const* name);

private:
    char const* m_name;
    bool m_active;
    time_point m_start;
};


template <typename D>
inline auto add_cl_event_record(char const* name, D&& duration) {
    auto d_nanos = std::chrono::nanoseconds{std::forward<D>(duration)};
    auto d_mills = std::chrono::duration_cast<Registry::duration_type>(d_nanos);

    auto r = Registry::get().store().insert({name, {1ull, d_mills}});
    if (!r.second) {
        r.first->second.duration += d_mills;
        r.first->second.executions += 1;
    }
}

template <typename Rep, typename Period>
inline auto add_cl_event_record(char const* name, std::chrono::duration<Rep, Period> duration) {
    auto d_mills = std::chrono::duration_cast<Registry::duration_type>(duration);

    auto r = Registry::get().store().insert({name, {1ull, d_mills}});
    if (!r.second) {
        r.first->second.duration += d_mills;
        r.first->second.executions += 1;
    }
}

template <>
inline auto add_cl_event_record(char const* name, cl::Event& event) {
    event.wait();
    auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    add_cl_event_record(name, end - start);
}


auto Registry::get() -> Registry& {
    static Registry registry;
    return registry;
}

auto Registry::store() -> store_type& {
    return m_store;
}

auto Registry::store() const -> store_type const& {
    return m_store;
}


auto Record::start(char const* name) -> Record {
    return {name};
}

Record::Record(char const* name)
    : m_name{name}
    , m_active{true}
    , m_start{clock::now()} {}

Record::Record(Record&& other) {
    m_active = other.m_active;
    m_start = other.m_start;

    other.m_active = false;
}

Record::~Record() {
    stop();     // Note: technically, this might throw...
}

auto Record::operator= (Record&& rhs) -> Record& {
    m_active = rhs.m_active;
    m_start = rhs.m_start;

    rhs.m_active = false;

    return *this;
}

void Record::stop() {
    if (!m_active) { return; }

    // calculate duration
    Registry::duration_type d = clock::now() - m_start;

    // store duration
    auto r = Registry::get().store().insert({m_name, {1ull, d}});
    if (!r.second) {
        r.first->second.duration += d;
        r.first->second.executions += 1;
    }

    m_active = false;
}


inline void to_json(nlohmann::json& json, const Registry& reg) {
    for (auto const& entry : reg.store()) {
        json[entry.first]["executions"] = entry.second.executions;
        json[entry.first]["duration"] = entry.second.duration.count();
    }
}

}   /* namespace perf */
}   /* namespace utils */
