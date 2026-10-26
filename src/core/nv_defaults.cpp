// src/core/nv_defaults.cpp - see include/strata/core/nv_defaults.hpp.
#include "strata/core/nv_defaults.hpp"

#include "strata/core/emulate.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
bool off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}
}  // namespace

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor) {
    std::vector<std::pair<std::string, std::string>> t;
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    if ((mj == 8 && mn == 6) || (mj == 8 && mn == 9)) {
        const bool ada = mn == 9;
        if (off(ada ? "STRATA_SM89_PERF" : "STRATA_SM86_PERF")) return t;
        // sm_89 (Ada: RTX 4060-4090, RTX 6000 Ada, L4/L40) takes the sm_86 table unchanged.  Measured on one RTX 4090 24 GB
        // (PCIe 4.0 x16, Ryzen 9 9900X, WSL2/Docker Desktop, engine 0.1.51, Qwen3.8-Flash-Next Q2_0, MTP on, KV int8, 128K context):
        // decode +7% / +13% / +31% at ~124 / 4.3K / 16.9K prompt tokens (docs/BENCHMARKS.md; the nine variables alone: +8% / +13% /
        // +25%), prompt speed unchanged (+1% / -2%).  1 engine start per arm, 3 runs per case: a single-card, single-start result.
        // Output identity on sm_89 is NOT yet verified (sm_86 was gated IDENTICAL; here the bench did not compare texts), so
        // STRATA_SM89_PERF=0 is the way back.  The sm_86-only kernel tables (dev_class() in native_mmvq.cu, the dequant-wide
        // choice) stay sm_86-only: nothing was measured for them on Ada.
        t = {
            // 148-nv, RTX 3060 + RTX A4000 (IQ3_XXS, default flags, 8 pairs): the stack +9.6% / +14.6% (8/8 each), identity gate IDENTICAL.
            // Each is in because removing it from the full stack cost speed on a card (drop-one, 4 rounds) and saved none on the other.
            {"STRATA_ADAPT_NOWAIT", "1"},        // A4000 -6.3 points when removed (with ADAPT_LAG), 3060 -0.8
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_CPU_IQ_ALL_EXACT1", "1"},   // A4000 -3.3, 3060 -0.6
            {"STRATA_SIDE_MIXER", "2"},          // A4000 -1.2, 3060 -2.3
            {"STRATA_ATTN_DAILY", "1"},          // -1.2, -1.3
            {"STRATA_GR_UP75", "1"},             // -0.9, -1.8
            {"STRATA_GU_SWIGLU", "1"},           // -0.8, -1.4
            {"STRATA_QSA_ROWS", "1"},            // -0.5, -1.2
            {"STRATA_B6_MMVQ_ROWS", "1"},        // A4000 0, 3060 -1.7
        };
    } else if (mj == 6 && mn == 0) {
        if (off("STRATA_SM60_PERF")) return t;
        t = {
            // 148-nv, Tesla P100 (IQ3_XXS, default flags, 8 pairs): the stack +17.8% (8/8), identity gate IDENTICAL.
            {"STRATA_ADAPT_NOWAIT", "1"},        // -10 points when removed (with ADAPT_LAG)
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_SIDE_MIXER", "2"},          // -1.8
            {"STRATA_FUSE_WAIT", "2"},           // -1.6
            {"STRATA_B6_MMVQ_ROWS", "1"},        // -1.3
            {"STRATA_FUSE_DECODE", "1"},         // -1.3
            {"STRATA_GR_UP75", "1"},             // -1.2
        };
    }
    return t;
}

std::vector<std::string> apply_nv_defaults(int major, int minor) {
    std::vector<std::string> set;
    for (const auto& kv : nv_default_env(major, minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
        const char* arch = mj == 6 ? "sm_60 (Pascal)" : mn == 9 ? "sm_89 (Ada)" : "sm_86 (Ampere)";
        const char* off_name = mj == 6 ? "STRATA_SM60_PERF" : mn == 9 ? "STRATA_SM89_PERF" : "STRATA_SM86_PERF";
        std::fprintf(stderr, "strata: %s: %zu %sspeed switches on by default (%s=0 turns them off; a switch you set is kept): ",
                     arch, set.size(), mj == 8 && mn == 9 ? "(output identity not yet verified on Ada) " : "exact ", off_name);
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

void apply_spec_adapt_default(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    // sm_86 only.  sm_89 is deliberately not extended: this is opt-in even on sm_86 and nothing was measured on Ada.
    // The P100 (sm_60) gained 4-16% on JSON/code/chat/tool calls but lost 3% on the 4K agent transcript (5/5 pairs), so there it stays opt-in
    // Owner round 9 (2026-10-22): on the z440 A4000s the 5-pair A/B against 0.2.2 was 0.995 (code 0.996, json 1.019, tool 0.973,
    // agent 0.991) and the 3060 was length-confounded, so the earlier +9..17% did not reproduce: opt-in (STRATA_SPEC_ADAPT_SM86=1)
    // until a clean A/B shows it.
    const bool sm86 = mj == 8 && mn == 6;
    if (!sm86) return;
    if (const char* v = std::getenv("STRATA_SPEC_ADAPT_SM86"); v == nullptr || v[0] != '1') return;
    if (off("STRATA_SM86_PERF")) return;
    const std::pair<const char*, const char*> kv[2] = {{"STRATA_SPEC_ADAPT_EVERY", "2"}, {"STRATA_SPEC_ADAPT_WARM", "200"}};
    for (const auto& p : kv) {
        if (std::getenv(p.first) != nullptr) continue;
#if defined(_WIN32)
        _putenv_s(p.first, p.second);
#else
        setenv(p.first, p.second, 0);
#endif
    }
}

}  // namespace strata::core
