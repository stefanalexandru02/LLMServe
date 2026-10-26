// strata/core/conversation_pool.hpp - a process-wide pool of pre-faulted host memory for the conversation cache's segments.
//
// Why: parking a conversation copies ~240 MB of K/V and running state into fresh ConversationBuffer segments.  Fresh
// anonymous memory is faulted in on first touch, which inside a WSL2 VM ran at ~100 MB/s (2.2-2.5 s for one park, counted
// in the next request's prompt time); once the cache reused memory the VM already owned the same park took 34-207 ms
// (docs/BENCHMARKS.md).  Plain vector segments hand their pages back to the kernel on every eviction, so every park was
// "fresh" again.  This pool keeps a bounded region of memory and gives segments memory out of it; a segment that is freed
// goes back to the region with its pages still resident, and a background thread can fault the first K MiB in before
// the first request.
//
// What it is: one mmap'd region, divided into kGrain-byte slots (256 KiB).  A segment of >= kMinPooled bytes takes a run of
// whole slots (first fit, a fully resident run preferred); anything smaller, or anything that finds no run, is an ordinary
// heap allocation exactly as before.  Capacities are rounded up to whole slots while the pool is on (round()); with the
// pool off round() is the identity and allocate()/release() are operator new/delete, so behaviour is unchanged
// (STRATA_CONV_POOL=0, no init, or a non-Linux build).
//
// RAM accounting.  A free slot that is still resident is RAM this process already owns and the next conversation allocation
// takes without asking the kernel for more: free_resident_bytes().  conversation_memory_admit() (conversation_memory.hpp)
// subtracts it from an allocation before it checks MemAvailable, because evicting a parked conversation now returns
// memory to the pool and not to MemAvailable.  When a request cannot be served from the region (fragmentation, region full)
// the pool first gives every free resident slot back to the kernel (madvise DONTNEED), then allocates from the heap, so
// resident memory never grows by more than the admission assumed: RSS growth <= allocation - credit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::core {

struct ConversationPoolStats {
    uint64_t region_bytes = 0;          ///< the reserved region (0: pool off)
    uint64_t used_bytes = 0;            ///< slots handed out to segments
    uint64_t free_resident_bytes = 0;   ///< free slots whose pages are resident (the admission credit)
    uint64_t free_cold_bytes = 0;       ///< free slots not (known to be) resident
    uint64_t prefaulted_bytes = 0;      ///< what the background thread has faulted in so far
    uint64_t hits = 0;                  ///< segments served from the region
    uint64_t misses = 0;                ///< pooled-size requests the region could not serve (heap instead)
    uint64_t trims = 0;                 ///< times a miss gave the free resident slots back to the kernel
    bool prefault_running = false;
};

class ConversationPool {
public:
    static constexpr size_t kGrain = 256 * 1024;       ///< slot size; capacities round up to it
    static constexpr size_t kMinPooled = 1024 * 1024;  ///< smaller segments are plain heap allocations

    /// Reserves `region_bytes` of address space (rounded down to whole slots; untouched pages cost nothing) and, when
    /// `prefault_bytes` > 0, starts a low-priority thread that makes the first `prefault_bytes` of it resident.  The
    /// thread asks `may_continue` (when set) before every 64 MiB and stops for good when it says no (RAM short).
    /// `done` (when set) is called from the thread with the bytes faulted in and the seconds taken once it ends.  Call at
    /// most once, before the first conversation buffer exists.  false: pool left off (not Linux, mmap refused, size 0).
    static bool init(size_t region_bytes, size_t prefault_bytes, std::function<bool()> may_continue = {},
                     std::function<void(uint64_t, double, bool)> done = {});
    /// Stops the thread and unmaps the region.  Tests only: every pooled segment must be gone.
    static void shutdown();

    static bool enabled();
    /// The capacity a segment asked to hold `n` bytes gets: `n` with the pool off or n < kMinPooled, else `n` rounded up
    /// to whole slots.
    static size_t round(size_t n);
    /// `capacity` bytes (a value round() returned); throws std::bad_alloc.  Never returns nullptr for capacity > 0.
    static void* allocate(size_t capacity);
    /// Gives a block from allocate() back.  `used` is how many of its bytes were ever written: whole slots below it stay
    /// resident in the pool, the rest are counted as cold.
    static void release(void* p, size_t capacity, size_t used) noexcept;

    /// Gives every free resident slot back to the kernel; returns the bytes.
    static uint64_t trim();
    static uint64_t free_resident_bytes();
    static ConversationPoolStats stats();
};

/// The RAM (bytes) the next conversation allocation can take from the pool without growing this process (0 with the pool off).
uint64_t conversation_pool_credit();

}  // namespace strata::core
