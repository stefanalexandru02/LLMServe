// src/core/conversation_pool_test.cpp - standalone test of ConversationPool / ConversationBuffer (no GPU, NOT part of CMake).
//   g++ -std=c++17 -O1 -g -Wall -Wextra -fsanitize=address,undefined -Iinclude src/core/conversation_pool_test.cpp src/core/conversation_pool.cpp -pthread -o /tmp/pool_test && /tmp/pool_test
#include "strata/core/conversation_buffer.hpp"
#include "strata/core/conversation_memory.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <atomic>
using namespace strata::core;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
constexpr size_t MiB = 1 << 20;

static uint8_t pat(size_t i, int salt) { return (uint8_t) ((i * 131 + salt) ^ (i >> 7)); }
static void fill(ConversationBuffer& b, int salt) {
    b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t at) { for (size_t i = 0; i < n; ++i) p[i] = pat(at + i, salt); return true; });
}
static bool verify(const ConversationBuffer& b, int salt) {
    return b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t at) { for (size_t i = 0; i < n; ++i) if (p[i] != pat(at + i, salt)) return false; return true; });
}

int main() {
    // ---- pool off: vector-identical capacities
    {
        CHECK(!ConversationPool::enabled());
        CHECK(ConversationPool::round(5 * MiB + 17) == 5 * MiB + 17);
        ConversationBuffer b;
        const size_t est = b.allocation_peak(40 * MiB + 123);
        b.resize(40 * MiB + 123);
        CHECK(b.bytes() == est);
        fill(b, 1); CHECK(verify(b, 1));
        ConversationBuffer c = b;                       // deep copy
        CHECK(c == b); CHECK(verify(c, 1));
        ConversationBuffer d(std::move(c)); CHECK(c.size() == 0); CHECK(d == b);
        std::puts("off ok");
    }
    // ---- pool on, prefault
    std::atomic<uint64_t> pre_bytes{0}; std::atomic<bool> pre_done{false};
    CHECK(ConversationPool::init(96 * MiB, 32 * MiB, [] { return true; },
          [&](uint64_t b, double s, bool ok) { std::printf("prefault %llu MiB in %.3f s ok=%d\n", (unsigned long long) (b >> 20), s, (int) ok); pre_bytes = b; pre_done = true; }));
    CHECK(ConversationPool::enabled());
    for (int i = 0; i < 2000 && !pre_done; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(pre_done);
    CHECK(pre_bytes == 32 * MiB);
    auto st = ConversationPool::stats();
    CHECK(st.region_bytes == 96 * MiB && st.free_resident_bytes == 32 * MiB && st.free_cold_bytes == 64 * MiB && st.used_bytes == 0);
    CHECK(conversation_pool_credit() == 32 * MiB);
    // admission: an allocation the credit covers needs no RAM at all (even with unknown telemetry); beyond it MemAvailable is judged
    CHECK(conversation_memory_admit(std::nullopt, 30 * MiB, 1u << 30));
    CHECK(!conversation_memory_admit(std::nullopt, 33 * MiB, 1u << 30));
    CHECK(conversation_memory_admit(std::optional<uint64_t>(2000 * MiB), 33 * MiB + (500 * MiB), 1000 * MiB));       // 501 MiB fresh vs 1000 spare
    CHECK(!conversation_memory_admit(std::optional<uint64_t>(1100 * MiB), 33 * MiB + (500 * MiB), 1000 * MiB));      // 501 MiB fresh vs 100 spare
    CHECK(!conversation_memory_admit(std::optional<uint64_t>(500 * MiB), 0, 1000 * MiB));                            // 0 bytes: the floor itself
    CHECK(!conversation_memory_admit(std::optional<uint64_t>(5000 * MiB), UINT64_MAX, 1000 * MiB));
    // rounding
    CHECK(ConversationPool::round(5 * MiB + 17) == 5 * MiB + 256 * 1024);
    CHECK(ConversationPool::round(512 * 1024) == 512 * 1024);   // below kMinPooled: heap, exact
    {
        ConversationBuffer b;
        const size_t est = b.allocation_peak(20 * MiB + 1000);
        b.resize(20 * MiB + 1000);
        CHECK(b.bytes() == est);                     // the estimate equals what the allocation took
        CHECK(b.size() == 20 * MiB + 1000);
        fill(b, 2); CHECK(verify(b, 2));
        st = ConversationPool::stats();
        CHECK(st.hits == 2);                         // 16 MiB + 4.0004 MiB -> 2 segments, both from the pool
        CHECK(st.free_resident_bytes == 32 * MiB - (20 * MiB + 256 * 1024) + 0 || st.free_resident_bytes <= 12 * MiB);
        // growth by appends keeps content
        b.resize(30 * MiB); CHECK(b.size() == 30 * MiB);
        for (size_t i = 0; i < 20 * MiB + 1000; ++i) { /* sampled check below */ if (i % 4099 == 0) { uint8_t x; b.read(&x, i, 1); CHECK(x == pat(i, 2)); } }
        b.resize(7 * MiB + 3);
        CHECK(b.size() == 7 * MiB + 3);
        for (size_t i = 0; i < b.size(); i += 4099) { uint8_t x; b.read(&x, i, 1); CHECK(x == pat(i, 2)); }
        ConversationBuffer cp = b; CHECK(cp == b);   // copy rounds its own capacity
    }
    st = ConversationPool::stats();
    CHECK(st.used_bytes == 0);
    CHECK(st.free_resident_bytes >= 20 * MiB);       // freed segments came back resident (a segment shrunk to 7 MiB reports only that as warm)
    std::printf("after free: resident %llu MiB cold %llu MiB hits %llu\n", (unsigned long long) (st.free_resident_bytes >> 20), (unsigned long long) (st.free_cold_bytes >> 20), (unsigned long long) st.hits);

    // ---- misses: more than the region -> heap, free resident handed back
    {
        std::vector<ConversationBuffer> v(7);
        for (auto& b : v) { b.resize(15 * MiB); fill(b, 3); }       // 7 x 15 MiB = 105 MiB > 96 MiB
        st = ConversationPool::stats();
        CHECK(st.misses >= 1);
        CHECK(st.free_resident_bytes == 0);
        for (auto& b : v) CHECK(verify(b, 3));
    }
    st = ConversationPool::stats();
    CHECK(st.used_bytes == 0);
    std::printf("after misses: resident %llu MiB cold %llu MiB misses %llu trims %llu\n", (unsigned long long) (st.free_resident_bytes >> 20), (unsigned long long) (st.free_cold_bytes >> 20), (unsigned long long) st.misses, (unsigned long long) st.trims);

    // ---- first-touch cost: cold park vs warm park (same RSS accounting)
    {
        auto rss_mib = [] { std::ifstream f("/proc/self/statm"); size_t a, b; f >> a >> b; return b * 4096 / MiB; };
        ConversationPool::trim();
        const size_t r0 = rss_mib();
        auto cycle = [&] {
            auto t0 = std::chrono::steady_clock::now();
            { ConversationBuffer b; b.resize(48 * MiB); fill(b, 4); }
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        const double cold = cycle(); const size_t r1 = rss_mib();
        const double warm = cycle(); const size_t r2 = rss_mib();
        std::printf("cold %.1f ms warm %.1f ms; RSS %zu -> %zu -> %zu MiB\n", cold, warm, r0, r1, r2);
        CHECK(r2 <= r1 + 2);                        // the warm cycle took no new memory
        CHECK(r1 >= r0 + 40);                       // the pool keeps what it faulted (the heap would have returned it)
        CHECK(ConversationPool::free_resident_bytes() >= 40 * MiB);
        ConversationPool::trim();
        CHECK(ConversationPool::free_resident_bytes() == 0);
        CHECK(rss_mib() <= r0 + 8);                 // trim gave it back
    }
    // ---- fragmentation: free every other 2 MiB block (resident holes), then ask for 8 MiB: miss -> heap, holes handed back
    {
        ConversationPool::trim();
        std::vector<ConversationBuffer> v(44);
        for (auto& b : v) { b.resize(2 * MiB); fill(b, 5); }          // 88 MiB of the 96 MiB
        for (size_t i = 0; i < v.size(); i += 2) v[i] = ConversationBuffer();
        const auto before = ConversationPool::stats();
        CHECK(before.free_resident_bytes == 22 * 2 * MiB);
        ConversationBuffer big; big.resize(12 * MiB); fill(big, 6);   // 8 MiB of holes at the end (cold) + 22 resident 2 MiB holes: no 12 MiB run? the tail is 8 MiB
        const auto after = ConversationPool::stats();
        CHECK(after.misses == before.misses + 1);
        CHECK(after.trims == before.trims + 1);
        CHECK(after.free_resident_bytes == 0);                         // the idle resident holes went back to the OS
        CHECK(verify(big, 6));
        for (size_t i = 1; i < v.size(); i += 2) CHECK(verify(v[i], 5));
    }
    // ---- threads
    {
        std::atomic<int> bad{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < 6; ++t) ts.emplace_back([&, t] {
            for (int i = 0; i < 60; ++i) {
                ConversationBuffer b; b.resize((1 + (i * 7 + t) % 5) * MiB + 12345 * t); fill(b, t);
                if (!verify(b, t)) ++bad;
            }
        });
        for (auto& t : ts) t.join();
        CHECK(bad == 0);
        CHECK(ConversationPool::stats().used_bytes == 0);
    }
    ConversationPool::shutdown();
    CHECK(!ConversationPool::enabled());
    std::puts("ALL OK");
    return 0;
}
