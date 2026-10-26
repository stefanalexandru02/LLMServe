"""Unit tests for llmserve.prepare and llmserve.catalog.  No GPU, no network (a local HTTP server stands in for
Hugging Face), no model files.

    python3 -m venv .venv && .venv/bin/pip install pytest numpy jinja2 markupsafe regex psutil
    .venv/bin/python -m pytest llmserve/test_prepare.py -q
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmserve import catalog as C          # noqa: E402
from llmserve import prepare as P          # noqa: E402

APP = Path("/opt/llmserve")                # a path in the plan only; nothing is read from it


def hw(ram=64.0, vram=24.0, wsl=False, avx512=True, rotational=None, gpus=1, fs="ext4", **kw) -> P.Hardware:
    return P.Hardware(ram_gib=ram, host_ram_gib=ram, gpu_name="Test GPU", vram_gib=vram, compute_cap="8.9", driver="581.1",
                      gpu_count=gpus, wsl=wsl, avx2=True, avx512=avx512, cpu_name="Test CPU", cpu_threads=16,
                      rotational=rotational, data_fs=fs, **kw)


def plan(quant="Q2_0", machine=None, **kw) -> P.Plan:
    kw.setdefault("context", 65536)
    s = P.Settings(quant=quant, data_dir=Path("/data"), app_dir=APP, **kw)
    return P.build_plan(s, machine or hw())


# ------------------------------------------------------------------------------------------- catalog
def test_catalog_pins_match_setup_py():
    q = C.QUANTS["Q2_0"]
    f1, f2 = C.shard_files(q)
    assert f1.repo == "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    assert f1.revision == "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    assert (f1.size, f1.sha256) == (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720")
    assert f1.path == "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
    assert f1.url() == ("https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/"
                        "ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")
    assert C.LLAMA_CPP_COMMIT == "3cf03257f219afbe7334045ff7c6a06ac68c627d"
    assert C.MTP_REVISION == "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def test_every_quant_resolves_to_pinned_files_and_the_published_size():
    for q in C.QUANTS.values():
        files = C.shard_files(q)
        assert len(files) == q.shards
        assert all(len(f.sha256) == 64 for f in files)
        assert abs(sum(f.size for f in files) / 1e9 - q.download_gb) < 0.06, q.name
        assert C.mmproj_file(q).size == 907543008
    # the n-gram table (shard 2 of every ISTA size and of the Coder) is one and the same file
    tables = {C.shard_files(C.QUANTS[n])[1].sha256 for n in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ1_M")}
    assert len(tables) == 1


def test_resolve_quant_is_forgiving_and_lists_choices():
    assert C.resolve_quant("q2_0").name == "Q2_0"
    assert C.resolve_quant("ud-q4_k_xl").name == "UD-Q4_K_XL"
    with pytest.raises(ValueError, match="Q2_0"):
        C.resolve_quant("Q9_9")


# ------------------------------------------------------------------------------------------- engine arguments
EXPECTED_NATIVE = (
    "--pack /data/packs/q2_0 "
    "--native /data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf "
    "--ple-gguf /data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf "
    "--expert-profile /opt/llmserve/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 "
    "--spec-min-p 0.5 --mtp /data/mtp/rt --max-context 65536 --kv int8 --kv-resident 32768")


def test_q2_0_64k_native_linux_64gb_24gb():
    p = plan(machine=hw(ram=64, vram=24))
    assert shlex.join(p.args) == EXPECTED_NATIVE
    assert not p.low_ram and p.kv_streaming and p.pack_variant == "q2_avx512"
    assert p.warnings() == []


def test_q2_0_64k_wsl2_64gb_16gb_has_no_kv_resident_and_no_ple_io_ram():
    p = plan(machine=hw(ram=64, vram=16, wsl=True))
    assert shlex.join(p.args) == EXPECTED_NATIVE.replace(" --kv-resident 32768", "")
    big = plan(machine=hw(ram=256, vram=16, wsl=True), context=131072)
    assert "--kv-resident" not in big.args and "--ple-io" not in big.args


def test_native_with_lots_of_ram_keeps_the_ngram_table_in_ram():
    assert P.arg_after(plan(machine=hw(ram=128)).args, "--ple-io") == "ram"
    assert "--ple-io" not in plan(machine=hw(ram=64)).args


def test_rotational_disk_keeps_the_table_in_ram_only_with_room():
    assert P.arg_after(plan(machine=hw(ram=96, rotational="sda")).args, "--ple-io") == "ram"
    p = plan(machine=hw(ram=64, rotational="sda"))
    assert "--ple-io" not in p.args and any("rotational" in w for w in p.warnings())
    assert "--ple-io" not in plan(machine=hw(ram=96, rotational="sda", wsl=True)).args   # WSL's disk says rotational


def test_kv_streaming_needs_64k_and_ram():
    assert "--kv-resident" not in plan(context=32768).args
    p = plan(context=131072, machine=hw(ram=48))        # 48 GiB: the experts + 1.8 GB + 1 do not fit -> off
    assert "--kv-resident" not in p.args
    assert "--kv-resident" in plan(context=131072, kv_streaming="on", machine=hw(ram=48)).args
    assert "--kv-resident" not in plan(context=131072, kv_streaming="off").args


def test_8k_context_uses_fp16_and_no_kv_flag():
    a = plan(context=8192).args
    assert P.arg_after(a, "--max-context") == "8192" and "--kv" not in a


def test_rope_scaling_only_past_the_trained_context():
    assert "--rope-scaling" not in plan(context=262144).args
    a = plan(context=393216).args
    assert P.arg_after(a, "--rope-scaling") == "yarn" and float(P.arg_after(a, "--rope-scale")) == 1.5
    with pytest.raises(P.Fail):
        plan(context=393216, rope_scaling="none")


def test_context_auto_follows_vram():
    assert plan(context="auto", machine=hw(vram=12)).ctx == 32768
    assert plan(context="auto", machine=hw(vram=16)).ctx == 65536
    assert plan(context="auto", machine=hw(vram=24)).ctx == 131072


def test_parse_context():
    assert P.parse_context("128k") == 131072 and P.parse_context("65536") == 65536 and P.parse_context("auto") == "auto"
    with pytest.raises(ValueError):
        P.parse_context("lots")


def test_mtp_off_drops_the_draft_layer_options():
    p = plan(mtp=False)
    assert "--mtp" not in p.args and "--spec-min-p" not in p.args and P.arg_after(p.args, "--spec") == "4"
    assert p.cfg["mtp"] is False


def test_vision_choices():
    g = plan(vision="on", machine=hw(vram=24))
    assert g.vision == "gpu" and "--vision" in g.args and P.arg_after(g.args, "--vram-reserve-mib") == "700"
    assert g.cfg["vision"]["gpu"] is True and g.cfg["vision"]["max_tokens"] == 1024
    assert g.cfg["vision"]["exe"].endswith("/strata-vision") and g.cfg["vision"]["model"] == str(g.shards[0])
    c = plan(vision="on", machine=hw(vram=16))
    assert c.vision == "cpu" and "--vram-reserve-mib" not in c.args
    assert c.cfg["vision"]["gpu"] is False and c.cfg["vision"]["max_tokens"] == 300 and c.cfg["vision"]["threads"] == 8
    assert "vision" not in plan().cfg and "--vision" not in plan().args
    assert abs(g.download_gb - (66.4 + 0.9)) < 0.1


def test_extra_args_win_over_generated_ones():
    p = plan(extra_args=("--spec", "3", "--pool-workers", "8"))
    assert P.arg_after(p.args, "--spec") == "3" and P.arg_after(p.args, "--pool-workers") == "8"
    assert p.args.count("--spec") == 1
    assert p.cfg["user_args"] == ["--spec", "3", "--pool-workers", "8"]


def test_vram_reserve_override_and_engine_env():
    p = plan(vram_reserve_mib=1500, vision="gpu", engine_env={"STRATA_X": "1"})
    assert P.arg_after(p.args, "--vram-reserve-mib") == "1500" and p.cfg["env"] == {"STRATA_X": "1"}


def test_parallel_goes_to_the_config_not_the_args():
    p = plan(parallel=3)
    assert p.cfg["parallel"] == 3 and "--batch" not in p.args
    assert plan(parallel=20).cfg["parallel"] == P.PARALLEL_MAX
    assert "parallel" not in plan().cfg


# ------------------------------------------------------------------------------------------- parallel slots and batch settings
def test_parallel_recommendation_on_a_24gb_card_is_not_a_knife_edge():
    arena = C.QUANTS["Q2_0"].arena_gb
    for vram in (23.5, 23.99, 24.0, 24.5):                 # what a 4090 reports (23.99 GiB) and a little either side
        assert P.parallel_recommend(vram, arena, 65536, "int8", False) == 2
        assert P.parallel_recommend(vram, arena, 49152, "int8", False) == 3
        assert P.parallel_recommend(vram, arena, 32768, "int8", False) == 4
    assert P.parallel_recommend(24.0, arena, 131072, "int8", False) == 0      # 1.8 GB a slot: not worth it
    assert P.parallel_recommend(24.0, arena, 65536, "int8", True) == 4        # streaming: a slot keeps 32K of KV in VRAM
    assert P.parallel_recommend(24.0, arena, 65536, "q4_0", False) == 3       # half the KV memory
    assert P.parallel_recommend(32.0, arena, 65536, "int8", False) == 3
    for vram in (8.0, 12.0, 16.0):                          # a cache under 40% of the experts: no recommendation
        assert P.parallel_recommend(vram, arena, 32768, "int8", False) == 0


def test_parallel_slot_cost_matches_the_report_estimates():
    gb = lambda ctx: P.parallel_slot_gb(ctx, "int8", False)
    assert gb(32768) == pytest.approx(0.59, abs=0.01) and gb(49152) == pytest.approx(0.79, abs=0.01)
    assert gb(65536) == pytest.approx(1.0, abs=0.01) and gb(131072) == pytest.approx(1.83, abs=0.01)
    cache = P.parallel_cache_gb(24.0)
    assert 0.04 < gb(65536) / cache < 0.07                 # about 5% of the expert cache a slot


def test_parallel_two_on_a_24gb_wsl_card_gets_no_warning_but_says_estimate():
    p = plan(machine=hw(vram=23.99, wsl=True), parallel=2)
    assert p.warnings() == []
    line = next(t for lvl, t in p.notes if t.startswith("parallel requests: 2"))
    assert "estimate" in line and "1.0 GB" in line and "% of the" in line
    assert p.cfg["parallel"] == 2


def test_parallel_warnings_say_estimate_and_follow_the_card():
    more = plan(machine=hw(vram=23.99, wsl=True), parallel=4)            # 4 x 1.0 GB at 64K: over 13% of the cache
    assert any("recommended for this card at this context: 2" in w and "estimate" in w for w in more.warnings())
    big = plan(machine=hw(vram=23.99, wsl=True), parallel=2, context=131072)   # 2 x 1.8 GB: no recommendation at all
    assert any("estimate" in w and "5% of the single-request speed" in w and "19%" in w for w in big.warnings())
    over = plan(machine=hw(vram=23.99, wsl=True), parallel=3, context=49152)
    assert over.warnings() == []                                            # 3 slots at 48K is inside the recommendation
    over = plan(machine=hw(vram=23.99, wsl=True), parallel=4, context=49152)
    assert any("recommended for this card at this context: 3" in w and "estimate" in w for w in over.warnings())
    small = plan(machine=hw(vram=12.0, wsl=True), parallel=2, context=32768)
    assert any("10-25%" in w and "estimate" in w for w in small.warnings())


def test_batch_mtp_only_with_two_slots_and_the_draft_layer():
    on = dict(batch_mtp=True)
    assert "--batch-mtp" in plan(parallel=2, **on).args
    assert "--batch-mtp" not in plan(parallel=1, **on).args and "--batch-mtp" not in plan(**on).args
    assert "--batch-mtp" not in plan(parallel=2, mtp=False, **on).args
    assert any("needs the draft layer" in w for w in plan(parallel=2, mtp=False, **on).warnings())
    assert "--batch-mtp" not in plan(parallel=2).args                   # off by default
    assert "--batch-mtp" not in plan(parallel=1, batch_mtp=True, extra_args=("--spec", "3")).args


def test_batch_switches_only_with_two_slots():
    wsl = hw(wsl=True)                                                    # WSL: no KV streaming, so ATTN_ROWS is allowed
    p = plan(machine=wsl, parallel=2, batch_switches=True)
    assert p.cfg["env"] == {"STRATA_BATCH_GDN_GROUPS": "1", "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_SLOT_COPY_DIRECT": "1",
                            "STRATA_SLOT_COPY_BATCHSYNC": "1"}
    assert "env" not in plan(machine=wsl, parallel=1, batch_switches=True).cfg
    assert "env" not in plan(machine=wsl, parallel=2).cfg
    streamed = plan(machine=hw(), parallel=2, batch_switches=True)       # native Linux, 64 GB: KV streaming (--kv-resident)
    assert streamed.kv_streaming and "--kv-resident" in streamed.args
    assert "STRATA_BATCH_ATTN_ROWS" not in streamed.cfg["env"] and streamed.cfg["env"]["STRATA_BATCH_GDN_GROUPS"] == "1"
    mine = plan(machine=wsl, parallel=2, batch_switches=True, extra_args=("--kv-resident", "32768"))
    assert "STRATA_BATCH_ATTN_ROWS" not in mine.cfg["env"]


def test_engine_env_and_extra_args_win_over_the_new_defaults():
    p = plan(machine=hw(wsl=True), parallel=2, batch_switches=True, agent=True, batch_mtp=True,
             engine_env={"STRATA_SLOT_COPY_DIRECT": "0", "STRATA_X": "1"},
             extra_args=("--conversation-cache-mib", "1024"))
    assert p.cfg["env"]["STRATA_SLOT_COPY_DIRECT"] == "0" and p.cfg["env"]["STRATA_X"] == "1"
    assert p.cfg["env"]["STRATA_BATCH_GDN_GROUPS"] == "1" and p.cfg["env"]["STRATA_CACHE_MESSAGE_BOUNDARY"] == "1"
    assert P.arg_after(p.args, "--conversation-cache-mib") == "1024" and p.args.count("--conversation-cache-mib") == 1
    assert "--batch-mtp" in p.args


def test_agent_mode_settings():
    p = plan(machine=hw(ram=64.0, wsl=True), agent=True)
    assert p.cfg["env"] == {"STRATA_CACHE_MESSAGE_BOUNDARY": "1"}
    assert P.arg_after(p.args, "--conversation-cache-mib") == "4096"
    assert not any("ADMIT_SHORT_FIRST" in t for lvl, t in p.notes)       # one slot: no admission order to set
    p2 = plan(machine=hw(ram=64.0, wsl=True), agent=True, parallel=2)
    assert any("STRATA_ADMIT_SHORT_FIRST=4096" in t for lvl, t in p2.notes)
    assert "STRATA_ADMIT_SHORT_FIRST" not in p2.cfg["env"]                # the server reads it, not the engine
    assert "--conversation-cache-mib" not in plan().args                  # not without the setting
    assert "env" not in plan().cfg


def test_agent_conversation_cache_is_capped_by_the_ram_left():
    q = C.QUANTS["Q2_0"]
    tight = plan(machine=hw(ram=q.ram_gb + 4 + 2, wsl=True), agent=True)        # 2 GiB left beside the quant
    assert P.arg_after(tight.args, "--conversation-cache-mib") == "2048"
    none = plan(machine=hw(ram=q.ram_gb + 4, wsl=True), agent=True)             # nothing left: the engine sizes it
    assert "--conversation-cache-mib" not in none.args and any("parking" in w for w in none.warnings())
    assert none.cfg["env"] == {"STRATA_CACHE_MESSAGE_BOUNDARY": "1"}


def test_conversation_cache_mib_setting():
    p = plan(conversation_cache_mib=2048)
    assert P.arg_after(p.args, "--conversation-cache-mib") == "2048"
    assert P.arg_after(plan(conversation_cache_mib=0).args, "--conversation-cache-mib") == "0"
    assert P.arg_after(plan(conversation_cache_mib=512, agent=True).args, "--conversation-cache-mib") == "512"   # explicit wins


def test_new_settings_change_the_fingerprint():
    base = plan(machine=hw(wsl=True), parallel=2).fingerprint
    for kw in (dict(batch_mtp=True), dict(batch_switches=True), dict(agent=True), dict(conversation_cache_mib=2048)):
        assert plan(machine=hw(wsl=True), parallel=2, **kw).fingerprint != base, kw
    assert plan(machine=hw(wsl=True), parallel=2).fingerprint == base


def test_new_settings_from_the_environment(monkeypatch):
    for k, v in (("LLMSERVE_BATCH_MTP", "on"), ("LLMSERVE_BATCH_SWITCHES", "on"), ("LLMSERVE_AGENT", "on"),
                 ("LLMSERVE_CONVERSATION_CACHE_MIB", "3000"), ("LLMSERVE_PARALLEL", "2")):
        monkeypatch.setenv(k, v)
    s = P.settings_from(P.build_parser().parse_args([]))
    assert (s.batch_mtp, s.batch_switches, s.agent, s.conversation_cache_mib, s.parallel) == (True, True, True, 3000, 2)
    s = P.settings_from(P.build_parser().parse_args(["--batch-mtp", "off", "--agent", "off", "--conversation-cache-mib", "0"]))
    assert (s.batch_mtp, s.agent, s.conversation_cache_mib) == (False, False, 0)
    for k in ("LLMSERVE_BATCH_MTP", "LLMSERVE_BATCH_SWITCHES", "LLMSERVE_AGENT", "LLMSERVE_CONVERSATION_CACHE_MIB"):
        monkeypatch.delenv(k)
    s = P.settings_from(P.build_parser().parse_args([]))
    assert (s.batch_mtp, s.batch_switches, s.agent, s.conversation_cache_mib) == (False, False, False, None)
    with pytest.raises(P.Fail):
        P.settings_from(P.build_parser().parse_args(["--conversation-cache-mib", "-5"]))


def test_multi_gpu_host_pins_one_card():
    p = plan(machine=hw(gpus=2))
    assert p.cfg["gpu"] == 0 and p.cfg["gpus_asked"] is True
    assert "gpu" not in plan().cfg


# ------------------------------------------------------------------------------------------- RAM fit
def test_low_ram_decisions():
    q = C.QUANTS["Q2_0"]
    assert not P.low_ram_needed(q, 64) and not P.low_ram_needed(q, 44)
    assert P.low_ram_needed(q, 43.9)
    assert P.low_ram_wanted(q, 64, "mmap") and not P.low_ram_wanted(q, 32, "off")
    assert not P.low_ram_wanted(C.QUANTS["UD-Q4_K_XL"], 16)           # Unsloth files have their own mode


def test_low_ram_resident_vs_mmap():
    p = plan(machine=hw(ram=32))                      # the GPU holds ~17 GB, the other 17 + 10 fit 32 GiB
    assert p.low_ram and p.resident and "--resident-experts" in p.args and p.need_experts_bin is False   # AVX-512 pack
    assert any("low-RAM" in w for w in p.warnings())
    m = plan(machine=hw(ram=26, avx512=False))
    assert m.low_ram and not m.resident and "--mmap-experts" in m.args and m.need_experts_bin
    assert "--mmap-experts" in plan(low_ram="mmap").args
    # streaming the KV cache needs the RAM beside the experts: not on a 32 GiB machine
    assert "--kv-resident" not in plan(context=131072, machine=hw(ram=32)).args


def test_wsl_with_too_little_memory_points_at_wslconfig():
    p = plan(machine=hw(ram=31.2, wsl=True))
    assert p.low_ram and any(".wslconfig" in w for w in p.warnings())


def test_unsloth_ram_budget_mode():
    p = plan("UD-Q4_K_XL", machine=hw(ram=64), context=8192)
    assert p.budget == 40 and P.arg_after(p.args, "--resident-budget-gib") == "40"
    assert "--ple-gguf" not in p.args and p.cfg["env"]["STRATA_HC_Q8"] == "1" and "STRATA_ROUTE_TAIL_SKIP" not in p.cfg["env"]
    p64k = plan("UD-Q4_K_XL", machine=hw(ram=64))          # KV streaming's RAM comes out of the budget
    assert p64k.budget == 39 and "--kv-resident" in p64k.args
    assert any("asks for 48 GB" in w or "keeps" in w for w in plan("UD-Q4_K_XL", machine=hw(ram=40)).warnings())
    big = plan("UD-Q4_K_XL", machine=hw(ram=128), context=8192)       # 77 GB of experts + 16 fit: all in RAM
    assert big.budget is None and big.cfg["env"]["STRATA_ROUTE_TAIL_SKIP"] == "0"
    assert plan("UD-IQ4_XS", machine=hw(ram=64)).quant.shards == 3
    assert plan("UD-Q5_K_XL", machine=hw(ram=256), context=8192).cfg["env"]["STRATA_MMQ_KQ"] == "1"


def test_coder_and_other_ista_sizes_use_their_own_files():
    c = plan("IQ1_M", machine=hw(ram=40))
    assert c.pack.name == "coder-iq1_m" and P.arg_after(c.args, "--expert-profile").endswith("expert-profile-coder.bin")
    assert not c.low_ram and c.pack_variant == "iq"           # 23.4 GB of experts + 10 < 40
    assert plan("IQ3_S", machine=hw(ram=64)).pack_variant == "iq"
    assert plan("IQ3_S", machine=hw(ram=56)).low_ram          # 50.3 + 10 > 56


def test_slow_data_mount_is_warned_about():
    p = plan(machine=hw(fs="9p"))
    assert any("9p" in w and "named Docker volume" in w for w in p.warnings())
    assert not any("mount" in w for w in plan().warnings())


def test_no_gpu_degrades_to_a_warning():
    p = plan(machine=hw(vram=0.0, gpus=0))
    assert any("no NVIDIA GPU" in w for w in p.warnings()) and p.args


def test_cgroup_limit_is_the_smaller_ram():
    assert P.cgroup_limit_gib("max\n", None) is None
    assert P.cgroup_limit_gib(None, str(2 ** 63 - 4096)) is None
    assert P.cgroup_limit_gib("34359738368\n", None) == 32.0
    assert P.cgroup_limit_gib("max", "17179869184") == 16.0


# ------------------------------------------------------------------------------------------- detection helpers
def test_parse_nvidia_smi():
    out = "0, NVIDIA GeForce RTX 5090, 32607, 12.0, 581.57\n1, NVIDIA GeForce RTX 4060, 8188, 8.9, 581.57\n"
    g = P.parse_nvidia_smi(out)
    assert [x["index"] for x in g] == [0, 1] and round(g[0]["vram_gib"], 1) == 31.8 and g[1]["cc"] == "8.9"
    assert P.parse_nvidia_smi("No devices were found") == []


def test_meminfo_cpuinfo_and_mounts():
    assert round(P.read_meminfo_gib("MemTotal:       65536000 kB\nMemFree: 1 kB\n"), 1) == 62.5
    name, avx2, avx512 = P.parse_cpuinfo("model name\t: Test\nflags\t\t: fpu avx2 avx512f avx512bw avx512vl avx512_vnni avx512vbmi\n")
    assert (name, avx2, avx512) == ("Test", True, True)
    assert P.parse_cpuinfo("flags : avx2 avx512f\n")[2] is False
    mounts = "overlay / overlay rw 0 0\n/dev/sdc /data ext4 rw 0 0\nC:\\134 /mnt/c 9p rw 0 0\ndrvfs /mnt/d/x\\040y 9p rw 0 0\n"
    assert P.mount_fs_type("/data/models", mounts) == "ext4"
    assert P.mount_fs_type("/mnt/c/Users", mounts) == "9p"
    assert P.mount_fs_type("/mnt/d/x y/z", mounts) == "9p"
    assert P.mount_fs_type("/tmp", mounts) == "overlay"


def test_wsl_detection():
    assert P.detect_wsl("5.15.167.4-microsoft-standard-WSL2", False)
    assert P.detect_wsl("6.6.0-generic", True)
    assert not P.detect_wsl("6.8.0-45-generic", False)


# ------------------------------------------------------------------------------------------- cfg.json and the server
def written_cfg(tmp_path, **kw):
    s = P.Settings(quant="Q2_0", data_dir=tmp_path, app_dir=APP, context=65536, **kw)
    p = P.build_plan(s, hw())
    P.step_config(p, None)
    return p, json.loads((tmp_path / "cfg.json").read_text())


def test_cfg_json_has_the_keys_the_server_reads(tmp_path):
    p, cfg = written_cfg(tmp_path, parallel=2)
    for key in ("exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs"):
        assert key in cfg, key
    assert cfg["exe"] == "/opt/llmserve/bin/strata" and cfg["tokenizer"] == f"{tmp_path}/packs/q2_0/tokenizer"
    assert all(isinstance(a, str) for a in cfg["args"])
    assert (tmp_path / "logs").is_dir()


def test_the_server_loads_cfg_json_and_adds_batch(tmp_path):
    """serve/server.py's own loader and engine_args() read what prepare wrote."""
    try:
        from serve import server as srv
    except Exception as e:                          # the server's imports are not all there (a trimmed checkout)
        pytest.skip(f"serve.server cannot be imported here: {e}")
    p, _ = written_cfg(tmp_path, parallel=3)
    path = tmp_path / "cfg.json"
    cfg = srv.parse_config_text(path.read_text(encoding="utf-8-sig"), str(path))
    srv.split_joined_args(cfg)
    assert cfg["args"] == p.args and "--kv-resident" in cfg["args"]
    assert srv.engine_args(cfg)[-2:] == ["--batch", "3"]
    assert (cfg.get("model_name") or "").startswith("qwen3.8-flash-next")


def test_up_to_date_follows_settings_and_hardware(tmp_path):
    p, _ = written_cfg(tmp_path)
    for f in (p.shards[0], p.pack / "tokenizer" / "vocab.json", p.rt / "experts.bin"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    assert P.up_to_date(p)
    s2 = P.Settings(quant="Q2_0", data_dir=tmp_path, app_dir=APP, context=131072)
    assert not P.up_to_date(P.build_plan(s2, hw()))                       # another context
    assert not P.up_to_date(P.build_plan(p.settings, hw(ram=32)))         # the WSL memory was changed: re-plan
    assert P.up_to_date(P.build_plan(p.settings, hw(ram=64.2)))           # noise is rounded away
    (p.pack / "tokenizer" / "vocab.json").unlink()
    assert not P.up_to_date(p)


# ------------------------------------------------------------------------------------------- command line
def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path, capsys):
    target = tmp_path / "never-created"
    rc = P.main(["--dry-run", "--quant", "Q2_0", "--data-dir", str(target), "--context", "64k", "--ram-gib", "64",
                 "--vram-gib", "24", "--wsl", "off", "--avx512", "on", "--disk", "ssd", "--app-dir", "/opt/llmserve"])
    out = capsys.readouterr().out
    assert rc == 0 and not target.exists()
    assert "engine arguments:" in out and "--kv-resident 32768" in out and "--mtp " in out
    assert "dry run" in out


def test_dry_run_reads_the_environment(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLMSERVE_QUANT", "IQ2_XS")
    monkeypatch.setenv("LLMSERVE_CONTEXT", "32768")
    monkeypatch.setenv("LLMSERVE_EXTRA_ARGS", "--pool-workers 6")
    rc = P.main(["--dry-run", "--data-dir", str(tmp_path / "d"), "--ram-gib", "64", "--vram-gib", "12"])
    out = capsys.readouterr().out
    assert rc == 0 and "IQ2_XS-00001-of-00002.gguf" in out and "--max-context 32768" in out and "--pool-workers 6" in out


def test_bad_quant_and_bad_context_exit_2(tmp_path, capsys):
    assert P.main(["--dry-run", "--quant", "nope", "--data-dir", str(tmp_path)]) == 2
    assert P.main(["--dry-run", "--context", "huge", "--data-dir", str(tmp_path)]) == 2


def test_module_runs_as_a_script_for_the_container(tmp_path):
    r = subprocess.run([sys.executable, "-m", "llmserve.prepare", "--dry-run", "--data-dir", str(tmp_path / "x"),
                        "--ram-gib", "64", "--vram-gib", "24"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--max-context 65536" in r.stdout


# ------------------------------------------------------------------------------------------- downloads and markers
class _Range(http.server.BaseHTTPRequestHandler):
    payload = b""
    fail_once = False
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        type(self).hits.append(start)
        body = self.payload[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        self.end_headers()
        if type(self).fail_once and start == 0:
            type(self).fail_once = False
            self.wfile.write(body[: len(body) // 3])          # drop the connection a third of the way in
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture()
def server():
    payload = os.urandom(300_000)
    handler = type("H", (_Range,), {"payload": payload, "fail_once": False, "hits": []})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield handler, f"http://127.0.0.1:{httpd.server_address[1]}", payload
    httpd.shutdown()


def _remote(payload: bytes, name="m.gguf") -> C.RemoteFile:
    return C.RemoteFile("o/r", "a" * 40, f"Q/{name}", len(payload), hashlib.sha256(payload).hexdigest())


def test_download_resumes_after_a_dropped_connection_and_verifies(server, tmp_path, monkeypatch):
    handler, url, payload = server
    handler.fail_once = True
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    rf = _remote(payload)
    s = P.Settings(data_dir=tmp_path, app_dir=APP, hf_endpoint=url)
    dst = tmp_path / "models" / "Q" / "m.gguf"
    P.fetch(rf, dst, s, backend="urllib")
    assert dst.read_bytes() == payload and not dst.with_name("m.gguf.part").exists()
    assert len(handler.hits) >= 2 and handler.hits[1] > 0                      # the second request used a Range
    assert P.verified(dst, rf)
    hits = len(handler.hits)
    P.fetch(rf, dst, s, backend="urllib")                                      # verified marker: no request, no hashing
    assert len(handler.hits) == hits


def test_wrong_bytes_are_deleted_not_used(server, tmp_path):
    handler, url, payload = server
    rf = C.RemoteFile("o/r", "a" * 40, "Q/m.gguf", len(payload), "0" * 64)
    s = P.Settings(data_dir=tmp_path, app_dir=APP, hf_endpoint=url)
    dst = tmp_path / "models" / "Q" / "m.gguf"
    with pytest.raises(P.Fail, match="SHA-256"):
        P.fetch(rf, dst, s, backend="urllib")
    assert not dst.exists()


def test_identical_shard_is_hardlinked_from_another_quant(tmp_path):
    payload = b"n-gram table" * 100
    rf = _remote(payload, "table.gguf")
    first = tmp_path / "models" / "Q2_0" / "table.gguf"
    first.parent.mkdir(parents=True)
    first.write_bytes(payload)
    P.verify_file(first, rf)
    s = P.Settings(data_dir=tmp_path, app_dir=APP)
    second = tmp_path / "models" / "IQ3_S" / "table.gguf"
    P.fetch(rf, second, s, backend="urllib")                                   # no server: it must not be asked
    assert second.read_bytes() == payload and os.stat(first).st_ino == os.stat(second).st_ino


def test_markers_skip_finished_steps(tmp_path, monkeypatch):
    p = plan(machine=hw())
    s = P.Settings(quant="Q2_0", data_dir=tmp_path, app_dir=APP, context=65536)
    p = P.build_plan(s, hw())
    (p.pack / "tokenizer").mkdir(parents=True)
    (p.pack / "tokenizer" / "vocab.json").write_text("{}")
    (p.pack / "index.txt").write_text("x")
    (p.pack / "experts.bin").write_text("x")
    monkeypatch.setattr(P, "run_tool", lambda *a, **k: pytest.fail("a finished step ran again"))
    P.step_pack(p)
    assert P.marker_ok(tmp_path, "pack-Q2_0", f"q2_avx512:{p.shards[0].name}")
    (p.rt).mkdir(parents=True)
    (p.rt / "experts.bin").write_text("x")
    P.mark_done(tmp_path, "mtp-rt", C.MTP_REVISION)
    P.step_mtp(p)                                                              # nothing to fetch or pack


def test_disk_needed_counts_what_is_already_there(tmp_path):
    s = P.Settings(quant="Q2_0", data_dir=tmp_path, app_dir=APP, context=65536)
    p = P.build_plan(s, hw(avx512=False))
    full = P.disk_needed_gb(p)
    assert 70 < full < 73                                  # 66.4 download + 5.2 MTP
    p.shards[0].parent.mkdir(parents=True)
    with open(p.shards[0], "wb") as f:
        f.truncate(37623740192)                            # sparse: no disk used
    assert abs((full - P.disk_needed_gb(p)) - 37.6) < 0.1


def test_entrypoint_and_healthcheck_scripts_parse():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    for name in ("entrypoint.sh", "healthcheck.sh"):
        script = ROOT / "docker" / name
        if script.exists():
            subprocess.run([bash, "-n", str(script)], check=True)


def test_entrypoint_ram_table_matches_the_catalog():
    text = (ROOT / "docker" / "entrypoint.sh").read_text()
    body = text[text.index("quant_ram_gb()"):text.index("is_wsl()")]
    table = {}
    for pats, num in __import__("re").findall(r"^\s*([A-Za-z0-9_|\-]+)\)\s*echo (\d+)", body, __import__("re").M):
        for name in pats.split("|"):
            table[name] = int(num)
    assert set(table) == set(C.QUANTS), set(table) ^ set(C.QUANTS)
    for name, q in C.QUANTS.items():
        assert table[name] == q.ram_gb, name


def test_every_setting_is_documented_in_env_example():
    import re
    example = (ROOT / ".env.example").read_text()
    used = set(re.findall(r'"(LLMSERVE_[A-Z_]+)"', (ROOT / "llmserve" / "prepare.py").read_text()))
    used |= set(re.findall(r"\b(LLMSERVE_[A-Z_]+)\b", (ROOT / "docker" / "entrypoint.sh").read_text()))
    internal = {"LLMSERVE_APP_DIR", "LLMSERVE_ENGINE_DIR", "LLMSERVE_DATA_DIR", "LLMSERVE_HF_TOKEN_ENV",
                "LLMSERVE_DOWNLOADER", "LLMSERVE_PORT"}      # set by the image / compose, not by the person
    missing = sorted(n for n in used - internal if n not in example)
    assert not missing, f"not in .env.example: {missing}"


def test_full_run_with_fake_tools_then_nothing_to_do(tmp_path, monkeypatch, capsys):
    """main() end to end with the downloads and the packing tools replaced: the steps run in order, write their markers
    and cfg.json, and a second start with the same settings does nothing."""
    app, data = tmp_path / "app", tmp_path / "data"
    (app / "tools").mkdir(parents=True)
    (app / "data").mkdir()
    for n in ("strata_pack.py", "pack_index.py", "strata_tokenizer.py", "iq_pack.py", "mtp_fetch.py", "mtp_pack.py",
              "mtp_rt.py"):
        (app / "tools" / n).write_text("")
    calls = []

    def touch(p: Path, text="x"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def fake_tool(cmd, env, what, cwd=None, check=True):
        name, args = Path(cmd[1]).name, cmd[2:]
        calls.append(name + (" " + args[0] if name == "mtp_fetch.py" else ""))
        out = Path(args[args.index("--out") + 1]) if "--out" in args else None
        if name == "strata_pack.py":
            touch(out / "experts.bin")
        elif name == "pack_index.py":
            touch(Path(args[args.index("--pack") + 1]) / "index.txt")
        elif name == "strata_tokenizer.py":
            touch(out / "tokenizer" / "vocab.json", "{}")
        elif name == "iq_pack.py":
            touch(out / "native_experts.txt")
            touch(out / "tokenizer" / "vocab.json", "{}")
        elif name == "mtp_pack.py":
            touch(out)
        elif name == "mtp_rt.py":
            touch(out / "experts.bin")
        return 0

    def fake_download(plan, backend):
        for sh in plan.shards:
            touch(sh)

    monkeypatch.setattr(P, "run_tool", fake_tool)
    monkeypatch.setattr(P, "step_download", fake_download)
    monkeypatch.setattr(P, "check_environment", lambda plan: None)
    monkeypatch.setattr(P, "free_gb", lambda path: 1000.0)
    argv = ["--data-dir", str(data), "--app-dir", str(app), "--engine-dir", str(app / "bin"), "--ram-gib", "64",
            "--vram-gib", "24", "--avx512", "on", "--wsl", "off", "--disk", "ssd", "--if-changed"]
    assert P.main(argv) == 0
    assert calls == ["strata_pack.py", "pack_index.py", "strata_tokenizer.py", "mtp_fetch.py fetch",
                     "mtp_fetch.py verify", "mtp_pack.py", "mtp_rt.py"]
    cfg = json.loads((data / "cfg.json").read_text())
    assert cfg["exe"] == str(app / "bin" / "strata") and "--mtp" in cfg["args"]
    assert P.marker_ok(data, "pack-Q2_0", f"q2_avx512:{Path(cfg['args'][3]).name}")
    assert P.marker_ok(data, "mtp-rt", C.MTP_REVISION)
    capsys.readouterr()
    calls.clear()
    assert P.main(argv) == 0                                   # same settings: current, nothing runs
    assert calls == [] and "nothing to do" in capsys.readouterr().out
    assert P.main(argv[:-1] + ["--force"]) == 0                # --force remakes the pack and the draft layer
    assert "strata_pack.py" in calls and "mtp_rt.py" in calls
