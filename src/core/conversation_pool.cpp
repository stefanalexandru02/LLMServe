// src/core/conversation_pool.cpp - see include/strata/core/conversation_pool.hpp.
#include "strata/core/conversation_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace strata::core {

namespace {

constexpr uint8_t kCold = 0;      // free; not known to be resident
constexpr uint8_t kResident = 1;  // free; pages resident (the admission credit)
constexpr uint8_t kUsed = 2;      // owned by a segment
constexpr uint8_t kClaimed = 3;   // being faulted in by the background thread
constexpr size_t kNone = ~size_t(0);
constexpr size_t kPrefaultSlots = 4;        // 1 MiB per populate call: a stall of other threads' mmap calls stays short
constexpr size_t kCheckEverySlots = 256;    // 64 MiB between RAM checks

struct Pool {
    std::mutex mu;
    uint8_t* base = nullptr;
    size_t region = 0, nslots = 0;
    std::vector<uint8_t> st;
    uint64_t n_cold = 0, n_res = 0, n_used = 0;
    uint64_t hits = 0, misses = 0, trims = 0, prefaulted = 0;
    std::atomic<uint64_t> credit{0};   // n_res * kGrain, readable without the lock
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};
    std::thread th;
    std::function<bool()> may_continue;
    std::function<void(uint64_t, double, bool)> done;
};

Pool& pool() {
    static Pool* p = new Pool;   // never destroyed: segments may be freed during static destruction
    return *p;
}
std::atomic<bool> g_on{false};
std::atomic<uintptr_t> g_lo{0}, g_hi{0};   // the region's address range, for release() to tell its blocks from heap ones without the lock

void set_state(Pool& p, size_t i, uint8_t to) {
    const uint8_t from = p.st[i];
    if (from == to) return;
    // kClaimed (held by the prefault thread) is in no count
    if (from == kCold) --p.n_cold; else if (from == kResident) --p.n_res; else if (from == kUsed) --p.n_used;
    if (to == kCold) ++p.n_cold; else if (to == kResident) ++p.n_res; else if (to == kUsed) ++p.n_used;
    p.st[i] = to;
}
void publish_credit(Pool& p) { p.credit.store(p.n_res * ConversationPool::kGrain, std::memory_order_relaxed); }

// first run of `need` free slots (all resident when `resident_only`)
size_t find_run(const Pool& p, size_t need, bool resident_only) {
    size_t run = 0;
    for (size_t i = 0; i < p.nslots; ++i) {
        const uint8_t s = p.st[i];
        if (resident_only ? s == kResident : (s == kCold || s == kResident)) {
            if (++run == need) return i + 1 - need;
        } else {
            run = 0;
        }
    }
    return kNone;
}

// caller holds the lock
uint64_t trim_locked(Pool& p) {
    uint64_t freed = 0;
    size_t i = 0;
    while (i < p.nslots) {
        if (p.st[i] != kResident) { ++i; continue; }
        size_t j = i;
        while (j < p.nslots && p.st[j] == kResident) ++j;
#if defined(__linux__)
        (void) madvise(p.base + i * ConversationPool::kGrain, (j - i) * ConversationPool::kGrain, MADV_DONTNEED);
#endif
        for (size_t k = i; k < j; ++k) set_state(p, k, kCold);
        freed += (j - i) * ConversationPool::kGrain;
        i = j;
    }
    if (freed) ++p.trims;
    publish_credit(p);
    return freed;
}

#if defined(__linux__)
bool populate(uint8_t* at, size_t bytes) {
    static std::atomic<int> mode{0};   // 0 untried, 1 MADV_POPULATE_WRITE works, 2 touch by writing
    if (mode.load() != 2) {
        if (madvise(at, bytes, 23 /* MADV_POPULATE_WRITE (Linux 5.14) */) == 0) {
            mode.store(1);
            return true;
        }
        if (errno == EINVAL || errno == ENOSYS) mode.store(2);   // an older kernel: write the pages instead
        else return false;                                       // ENOMEM etc.: the memory is not there; leave the slots cold
    }
    std::memset(at, 0, bytes);   // the slots are claimed: nobody reads them
    return true;
}

void prefault_main(size_t slots_wanted) {
    Pool& p = pool();
    // lowest priority we can get: this thread must never compete with serving for a core
    sched_param sp{};
    (void) pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp);
    (void) setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 19);
    (void) pthread_setname_np(pthread_self(), "conv-prefault");
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t done_bytes = 0;
    bool completed = true;
    size_t i = 0, since_check = 0;
    while (i < slots_wanted) {
        if (p.stop.load()) { completed = false; break; }
        if (since_check >= kCheckEverySlots) {
            since_check = 0;
            if (p.may_continue && !p.may_continue()) { completed = false; break; }
        }
        size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(p.mu);
            while (n < kPrefaultSlots && i + n < slots_wanted && p.st[i + n] == kCold) { set_state(p, i + n, kClaimed); ++n; }
        }
        if (n == 0) { ++i; ++since_check; continue; }   // busy or already resident: skip it
        const bool ok = populate(p.base + i * ConversationPool::kGrain, n * ConversationPool::kGrain);
        {
            std::lock_guard<std::mutex> lk(p.mu);
            for (size_t k = 0; k < n; ++k) set_state(p, i + k, ok ? kResident : kCold);
            if (ok) p.prefaulted += n * ConversationPool::kGrain;
            publish_credit(p);
        }
        if (!ok) { completed = false; break; }
        done_bytes += n * ConversationPool::kGrain;
        i += n;
        since_check += n;
        std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    p.running.store(false);
    if (p.done) p.done(done_bytes, secs, completed);
}
#endif

}  // namespace

bool ConversationPool::init(size_t region_bytes, size_t prefault_bytes, std::function<bool()> may_continue,
                            std::function<void(uint64_t, double, bool)> done) {
#if defined(__linux__)
    if (const char* e = std::getenv("STRATA_CONV_POOL"); e != nullptr && e[0] == '0') return false;
    Pool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    if (p.base != nullptr) return g_on.load();
    const size_t slots = region_bytes / kGrain;
    if (slots == 0) return false;
    void* m = mmap(nullptr, slots * kGrain, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) return false;
    p.base = (uint8_t*) m;
    p.region = slots * kGrain;
    p.nslots = slots;
    p.st.assign(slots, kCold);
    p.n_cold = slots;
    p.n_res = p.n_used = 0;
    p.hits = p.misses = p.trims = p.prefaulted = 0;
    p.credit.store(0);
    p.stop.store(false);
    p.may_continue = std::move(may_continue);
    p.done = std::move(done);
    g_lo.store((uintptr_t) p.base);
    g_hi.store((uintptr_t) p.base + p.region);
    g_on.store(true, std::memory_order_release);
    const size_t want = std::min(slots, prefault_bytes / kGrain);
    if (want > 0) {
        p.running.store(true);
        try {
            p.th = std::thread(prefault_main, want);
        } catch (...) {
            p.running.store(false);   // no thread: the pool still works, its pages just fault on first use
        }
    }
    return true;
#else
    (void) region_bytes; (void) prefault_bytes; (void) may_continue; (void) done;
    return false;
#endif
}

void ConversationPool::shutdown() {
#if defined(__linux__)
    Pool& p = pool();
    p.stop.store(true);
    if (p.th.joinable()) p.th.join();
    std::lock_guard<std::mutex> lk(p.mu);
    g_on.store(false, std::memory_order_release);
    g_lo.store(0);
    g_hi.store(0);
    if (p.base != nullptr) munmap(p.base, p.region);
    p.base = nullptr;
    p.region = p.nslots = 0;
    p.st.clear();
    p.n_cold = p.n_res = p.n_used = 0;
    p.credit.store(0);
    p.running.store(false);
#endif
}

bool ConversationPool::enabled() { return g_on.load(std::memory_order_acquire); }

size_t ConversationPool::round(size_t n) {
    if (!enabled() || n < kMinPooled || n > SIZE_MAX - kGrain) return n;
    return (n + kGrain - 1) / kGrain * kGrain;
}

void* ConversationPool::allocate(size_t capacity) {
#if defined(__linux__)
    if (enabled() && capacity >= kMinPooled && capacity % kGrain == 0) {
        Pool& p = pool();
        const size_t need = capacity / kGrain;
        std::lock_guard<std::mutex> lk(p.mu);
        if (p.base != nullptr) {
            size_t at = find_run(p, need, true);
            if (at == kNone) at = find_run(p, need, false);
            if (at != kNone) {
                for (size_t k = 0; k < need; ++k) set_state(p, at + k, kUsed);
                ++p.hits;
                publish_credit(p);
                return p.base + at * kGrain;
            }
            // No run: this allocation goes to the heap and grows the process by its full size, so the free resident slots
            // (counted as credit by the admission) are given back first.
            ++p.misses;
            const uint64_t freed = trim_locked(p);
            if (p.misses <= 5 || p.misses % 50 == 0)
                std::fprintf(stderr, "strata serve: conversation pool: no free run of %zu MiB in the %zu MiB pool (%llu MiB in use); "
                                     "heap instead, %llu MiB of idle pool memory handed back to the OS (miss %llu)\n",
                             capacity >> 20, p.region >> 20, (unsigned long long) (p.n_used * kGrain >> 20),
                             (unsigned long long) (freed >> 20), (unsigned long long) p.misses);
        }
    }
#endif
    return ::operator new(capacity);
}

void ConversationPool::release(void* ptr, size_t capacity, size_t used) noexcept {
    if (ptr == nullptr) return;
#if defined(__linux__)
    const uintptr_t a = (uintptr_t) ptr;
    if (a >= g_lo.load() && a < g_hi.load()) {
        Pool& p = pool();
        uint8_t* b = (uint8_t*) ptr;
        std::lock_guard<std::mutex> lk(p.mu);
        if (p.base != nullptr && b >= p.base && b < p.base + p.region) {
            const size_t at = (size_t) (b - p.base) / kGrain, n = capacity / kGrain;
            const size_t warm = std::min(n, used / kGrain);   // slots written whole stay resident; the rest count as cold
            for (size_t k = 0; k < n; ++k) set_state(p, at + k, k < warm ? kResident : kCold);
            publish_credit(p);
            return;
        }
    }
#else
    (void) capacity; (void) used;
#endif
    ::operator delete(ptr);
}

uint64_t ConversationPool::trim() {
    if (!enabled()) return 0;
    Pool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.base != nullptr ? trim_locked(p) : 0;
}

uint64_t ConversationPool::free_resident_bytes() {
    return enabled() ? pool().credit.load(std::memory_order_relaxed) : 0;
}

ConversationPoolStats ConversationPool::stats() {
    ConversationPoolStats s;
    if (!enabled()) return s;
    Pool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    s.region_bytes = p.region;
    s.used_bytes = p.n_used * kGrain;
    s.free_resident_bytes = p.n_res * kGrain;
    s.free_cold_bytes = p.n_cold * kGrain;
    s.prefaulted_bytes = p.prefaulted;
    s.hits = p.hits;
    s.misses = p.misses;
    s.trims = p.trims;
    s.prefault_running = p.running.load();
    return s;
}

uint64_t conversation_pool_credit() { return ConversationPool::free_resident_bytes(); }

}  // namespace strata::core
