#pragma once
// DIAGNOSTIC, not for main as it stands. LEONIDA_GPU_PROFILE=1: where the GPU's time goes, per pipeline.
// Every compute dispatch the replay records and every draw's rendering scope is bracketed by two timestamp queries
// of one device-wide pool (a ring; each use resets its own two queries in the command buffer first). Results are
// read back a few seconds later from whichever thread records the next bracket, summed per pipeline name, and the
// 25 largest are printed every 10 s:
//   [rt] gpu profile: <ms> ms <n> dispatches avg <ms> max <ms> groups <g> gap before <ms> (max <ms>)
//        fragments avg <n> max <n> mesh invocations avg <n> max <n> <name>
// "dispatches" counts draws too. "groups" is x*y*z of a direct dispatch, 0 of an indirect one, and for a draw the
// maximum record count of an indirect draw (0 for a direct one). "gap before" is the time from the end of the
// bracket before it in the same command buffer to its own start (whatever the list records in between), with one
// histogram line of all gaps. LEONIDA_GPU_PROFILE_FRAGMENTS=1 adds a pipeline-statistics query per draw (fragment
// shader invocations); =2 also counts mesh shader invocations (needs the device's meshShaderQueries feature).
// A statistics query of the title's own that is active around a draw would overlap with it (invalid): not handled.
// The difference of two bottom-of-pipe timestamps is when the work before and after finished, so work running on
// other queues is counted too: a ranking, not an exact cost. Ticks are taken as nanoseconds (timestampPeriod 1.0,
// as on NVIDIA). Names are handed over although the fault diagnostics are off, and stay in the local log.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace leonida::gpu_profile {

inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_GPU_PROFILE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

struct State {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    VkQueryPool pool = VK_NULL_HANDLE;
    bool mesh_counter = false;
    VkQueryPool stat_pool = VK_NULL_HANDLE;  // fragment-shader invocations of a draw, query q / 2
    PFN_vkGetQueryPoolResults results = nullptr;
    std::unordered_map<std::uintptr_t, std::string> names;
    struct Pending {
        std::uint32_t query;
        std::uint32_t previous;
        bool draw;  // the dispatch before it in the same command buffer, or UINT32_MAX
        std::uintptr_t id;
        std::uint64_t groups;
        std::chrono::steady_clock::time_point at;
    };
    std::deque<Pending> pending;
    struct Sum {
        std::uint64_t count = 0, ns = 0, max_ns = 0, groups = 0, gap_ns = 0, gaps = 0, gap_max = 0, fragments = 0, fragment_max = 0, mesh = 0, mesh_max = 0;
    };
    std::unordered_map<std::string, Sum> sums;
    std::uint64_t lost = 0;
    std::uint64_t gap_count[5] = {}, gap_sum[5] = {};  // under 0.1 ms, 1 ms, 10 ms, 100 ms, and above
    std::chrono::steady_clock::time_point last_read{}, last_print = std::chrono::steady_clock::now();
    std::atomic<std::uint32_t> next{0};
};
constexpr std::uint32_t kQueries = 1u << 17;

inline State& state() {
    static State* s = new State;
    return *s;
}

// From VulkanDeviceFunctions::load: the pool, once per device.
inline void init(PFN_vkGetDeviceProcAddr get, VkDevice device) {
    if (!enabled() || !get || !device) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    if (s.pool) return;
    const auto create = reinterpret_cast<PFN_vkCreateQueryPool>(get(device, "vkCreateQueryPool"));
    s.results = reinterpret_cast<PFN_vkGetQueryPoolResults>(get(device, "vkGetQueryPoolResults"));
    if (!create || !s.results) return;
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kQueries;
    if (create(device, &info, nullptr, &s.pool) != VK_SUCCESS) {
        s.pool = VK_NULL_HANDLE;
        return;
    }
    s.device = device;
    VkQueryPoolCreateInfo stat{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    stat.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
    stat.queryCount = kQueries / 2;
    stat.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
    const char* fragments_mode = std::getenv("LEONIDA_GPU_PROFILE_FRAGMENTS");
    s.mesh_counter = fragments_mode && fragments_mode[0] == '2';  // needs meshShaderQueries on the device
    if (s.mesh_counter) stat.pipelineStatistics |= VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
    if (std::getenv("LEONIDA_GPU_PROFILE_FRAGMENTS") && create(device, &stat, nullptr, &s.stat_pool) != VK_SUCCESS)
        s.stat_pool = VK_NULL_HANDLE;
    std::fprintf(stderr, "[rt] gpu profile (local diagnostic): timestamp pool of %u queries ready\n", kQueries);
}

inline void name(std::uintptr_t id, const char* shader) {
    if (!enabled() || !shader) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    s.names[id] = shader;
}

inline void collect_locked(State& s) {
    const auto now = std::chrono::steady_clock::now();
    if (now - s.last_read < std::chrono::seconds(1)) return;
    s.last_read = now;
    while (!s.pending.empty() && now - s.pending.front().at > std::chrono::seconds(4)) {
        const State::Pending p = s.pending.front();
        s.pending.pop_front();
        std::uint64_t data[4] = {};
        const VkResult r = s.results(s.device, s.pool, p.query, 2, sizeof(data), data, 2 * sizeof(std::uint64_t),
                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS || !data[1] || !data[3] || data[2] < data[0]) {
            ++s.lost;
            continue;
        }
        const auto it = s.names.find(p.id);
        State::Sum& sum = s.sums[it == s.names.end() ? std::string("(unnamed)") : it->second];
        const std::uint64_t ns = data[2] - data[0];
        ++sum.count;
        sum.ns += ns;
        sum.max_ns = (std::max)(sum.max_ns, ns);
        sum.groups += p.groups;
        if (p.draw && s.stat_pool) {
            std::uint64_t f[3] = {};
            const std::size_t words = s.mesh_counter ? 3 : 2;
            if (s.results(s.device, s.stat_pool, p.query / 2, 1, words * 8, f, words * 8,
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS && f[words - 1]) {
                sum.fragments += f[0];
                sum.fragment_max = (std::max)(sum.fragment_max, f[0]);
                if (s.mesh_counter) sum.mesh += f[1], sum.mesh_max = (std::max)(sum.mesh_max, f[1]);
            }
        }
        if (p.previous != UINT32_MAX) {  // from the end of the dispatch before it to this one's start
            std::uint64_t before[2] = {};
            if (s.results(s.device, s.pool, p.previous + 1, 1, sizeof(before), before, sizeof(before),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS &&
                before[1] && data[0] >= before[0]) {
                sum.gap_ns += data[0] - before[0];
                sum.gap_max = (std::max)(sum.gap_max, data[0] - before[0]);
                const std::uint64_t g = data[0] - before[0];
                const int band = g < 100000 ? 0 : g < 1000000 ? 1 : g < 10000000 ? 2 : g < 100000000 ? 3 : 4;
                ++s.gap_count[band];
                s.gap_sum[band] += g;
                ++sum.gaps;
            }
        }
    }
    if (now - s.last_print < std::chrono::seconds(10)) return;
    const double seconds = std::chrono::duration<double>(now - s.last_print).count();
    s.last_print = now;
    std::vector<std::pair<std::string, State::Sum>> rows(s.sums.begin(), s.sums.end());
    s.sums.clear();
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns + a.second.gap_ns > b.second.ns + b.second.gap_ns; });
    std::uint64_t total = 0, count = 0, gap = 0, gaps = 0;
    for (const auto& row : rows)
        total += row.second.ns, count += row.second.count, gap += row.second.gap_ns, gaps += row.second.gaps;
    std::fprintf(stderr, "[rt] gpu profile: %.0f s, %llu dispatches of %zu pipelines read, %.1f ms of GPU time in all, "
                         "%.1f ms in %llu gaps between dispatches of one command buffer, %llu not read\n",
                 seconds, static_cast<unsigned long long>(count), rows.size(), double(total) / 1e6,
                 double(gap) / 1e6, static_cast<unsigned long long>(gaps), static_cast<unsigned long long>(s.lost));
    s.lost = 0;
    std::fprintf(stderr, "[rt] gpu profile: gaps under 0.1 ms: %llu (%.0f ms); 0.1-1 ms: %llu (%.0f ms); 1-10 ms: %llu (%.0f ms); "
                         "10-100 ms: %llu (%.0f ms); over 100 ms: %llu (%.0f ms)\n",
                 (unsigned long long)s.gap_count[0], s.gap_sum[0] / 1e6, (unsigned long long)s.gap_count[1], s.gap_sum[1] / 1e6,
                 (unsigned long long)s.gap_count[2], s.gap_sum[2] / 1e6, (unsigned long long)s.gap_count[3], s.gap_sum[3] / 1e6,
                 (unsigned long long)s.gap_count[4], s.gap_sum[4] / 1e6);
    for (int b = 0; b < 5; ++b) s.gap_count[b] = s.gap_sum[b] = 0;
    for (std::size_t i = 0; i < rows.size() && i < 25; ++i)
        std::fprintf(stderr, "[rt] gpu profile:   %9.1f ms  %6llu dispatches  avg %8.3f ms  max %8.1f ms  groups %llu  gap before %8.1f ms (max %7.1f)  fragments avg %10.0f max %10llu  mesh invocations avg %9.0f max %9llu  %s\n",
                     double(rows[i].second.ns) / 1e6, static_cast<unsigned long long>(rows[i].second.count),
                     double(rows[i].second.ns) / 1e6 / double(rows[i].second.count), double(rows[i].second.max_ns) / 1e6,
                     static_cast<unsigned long long>(rows[i].second.groups), double(rows[i].second.gap_ns) / 1e6, double(rows[i].second.gap_max) / 1e6,
                     double(rows[i].second.fragments) / double(rows[i].second.count),
                     static_cast<unsigned long long>(rows[i].second.fragment_max),
                     double(rows[i].second.mesh) / double(rows[i].second.count),
                     static_cast<unsigned long long>(rows[i].second.mesh_max),
                     rows[i].first.c_str());
}

// Before a dispatch: returns the first of its two queries, or UINT32_MAX when the profile is off.
template<typename Functions>
inline std::uint32_t begin(const Functions& vk, VkCommandBuffer cb) {
    if (!enabled()) return UINT32_MAX;
    State& s = state();
    if (!s.pool || !vk.reset_query_pool || !vk.write_timestamp) return UINT32_MAX;
    const std::uint32_t q = (s.next.fetch_add(2, std::memory_order_relaxed)) % kQueries;
    vk.reset_query_pool(cb, s.pool, q, 2);
    vk.write_timestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, q);
    return q;
}

template<typename Functions>
inline void end(const Functions& vk, VkCommandBuffer cb, std::uint32_t q, std::uintptr_t id, std::uint64_t groups,
                bool draw = false) {
    if (q == UINT32_MAX) return;
    State& s = state();
    vk.write_timestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, q + 1);
    thread_local VkCommandBuffer last_cb = VK_NULL_HANDLE;
    thread_local std::uint32_t last_q = UINT32_MAX;
    const std::uint32_t previous = last_cb == cb ? last_q : UINT32_MAX;
    last_cb = cb;
    last_q = q;
    std::lock_guard lock(s.mutex);
    s.pending.push_back({q, previous, draw, id, groups, std::chrono::steady_clock::now()});
    collect_locked(s);
}

// Around a draw's rendering scope: the timestamps, and with LEONIDA_GPU_PROFILE_FRAGMENTS its fragment-shader invocations.
template<typename Functions>
inline std::uint32_t begin_draw(const Functions& vk, VkCommandBuffer cb) {
    const std::uint32_t q = begin(vk, cb);
    State& s = state();
    if (q != UINT32_MAX && s.stat_pool && vk.begin_query && vk.end_query) {
        vk.reset_query_pool(cb, s.stat_pool, q / 2, 1);
        vk.begin_query(cb, s.stat_pool, q / 2, 0);
    }
    return q;
}
template<typename Functions>
inline void end_draw(const Functions& vk, VkCommandBuffer cb, std::uint32_t q, std::uintptr_t id, std::uint64_t pixels) {
    State& s = state();
    if (q != UINT32_MAX && s.stat_pool && vk.begin_query && vk.end_query) vk.end_query(cb, s.stat_pool, q / 2);
    end(vk, cb, q, id, pixels, true);
}

}  // namespace leonida::gpu_profile
