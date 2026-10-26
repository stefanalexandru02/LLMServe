"""LLMServe model catalog: Qwen3.8-Flash-Next and its quantizations.

Everything here is copied from Strata's setup.py (MIT, Niko1221 and contributors; https://github.com/niko1221/strata)
at the commit this fork was taken from, with the same pins:

  * every Hugging Face file is read at a FIXED revision (HF_REVISIONS), with a fixed size and SHA-256 (the Hub's LFS
    pointers, as setup.py's ISTA_FILES / CODER_FILES / UNSLOTH_* tables);
  * the MTP draft layer's tensors come from Qwen/Qwen3.8-Flash-Next at the revision tools/mtp_fetch.py pins;
  * llama.cpp (ggml, gguf-py, mtmd) is the commit in LLAMA_CPP_COMMIT.

Do not edit a hash or a revision by hand: copy the new values from a newer Strata release.  Nothing in this module
touches the network, the disk or the GPU, so it is safe to import anywhere (tests, --dry-run).

Model licenses: Qwen3.8-Flash-Next is under the Qwen Community License 1.0
(https://huggingface.co/Qwen/Qwen3.8-Flash-Next); the GSQ-RCO files are ISTA-DASLab's quantizations of it; the Coder
release is Apache-2.0 for its own changes; Unsloth's files follow the base model's license.  You accept them by
downloading the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------------- pins
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"          # setup.py:220, third_party/ggml/VERSION.txt
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

# setup.py HF_REVISIONS (only the repositories this fork uses)
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
}

# The MTP draft layer: only its ~5 GB of `mtp.*` tensors are read (HTTP range requests) from the original checkpoint.
# tools/mtp_fetch.py carries this same pin and checks every tensor against SHA-256; this constant is informational.
MTP_REPO = "Qwen/Qwen3.8-Flash-Next"
MTP_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MTP_DOWNLOAD_GB = 5.2

# data/draft_vocab*.bin: the draft head's token subset (setup.py DRAFT_VOCABS)
DRAFT_VOCABS = {'cjk': 'draft_vocab.bin', 'en': 'draft_vocab_en.bin', 'cyrillic': 'draft_vocab_cyrillic.bin', 'fr': 'draft_vocab_fr.bin'}
DEFAULT_DRAFT_VOCAB = "cjk"

# setup.py CONTEXTS: the menu of context lengths (the trained length is 262144; more uses yarn rope scaling)
CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
TRAINED_CONTEXT = 262144

MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"      # the picture reader, the same file for every Flash-Next model

# --------------------------------------------------------------------------------------------------------- files
# name -> (bytes, sha256), the Hub's LFS pointers at the revision above.  The n-gram table (shard 2 of the ISTA
# files) is the same file for every size.
ISTA_FILES = {  # ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S + the picture reader
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
CODER_FILES = {  # ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF: IQ1_M + the picture reader
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf":
        (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
UNSLOTH_FILES = {  # unsloth/Qwen3.8-Flash-Next-GGUF: UD-Q4_K_XL (4), UD-IQ4_XS (3), UD-Q5_K_XL / UD-Q6_K_XL / Q8_0 (6 each)
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf":
        (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf":
        (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf":
        (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf":
        (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf":
        (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf":
        (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf":
        (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf":
        (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf":
        (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf":
        (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf":
        (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf":
        (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
}

# --------------------------------------------------------------------------------------------------------- types
@dataclass(frozen=True)
class RemoteFile:
    """One pinned file on Hugging Face."""
    repo: str
    revision: str
    path: str            # path inside the repository, e.g. "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def url(self, endpoint: str = "https://huggingface.co") -> str:
        return f"{endpoint.rstrip('/')}/{self.repo}/resolve/{self.revision}/{self.path}"


@dataclass(frozen=True)
class Family:
    key: str
    title: str
    model_name: str                  # the name the API lists (cfg "model_name" is "<model_name>-<quant>")
    repo: str
    subdir: bool                     # files live in "<quant>/" inside the repository
    file_pattern: str                # "{q}" quant, "{i}" shard number (1-based), "{n}" shard count
    files: dict                      # name -> (bytes, sha256)
    mmproj_repo: str
    tag_prefix: str = ""             # folder names: models/<tag>, packs/<tag.lower()>
    profile: str = "expert-profile.bin"   # data/<profile>, the expert order the GPU cache starts from
    pack_args: tuple = ()            # extra arguments for tools/iq_pack.py
    license: str = ""

    @property
    def revision(self) -> str:
        return HF_REVISIONS[self.repo]


@dataclass(frozen=True)
class Quant:
    name: str
    family: str
    about: str
    download_gb: float               # everything for the text model (both/all shards)
    ram_gb: int                      # the RAM Strata asks for (setup.py MODELS["ram_gb"])
    arena_gb: float                  # the experts pinned in RAM (setup.py "arena_gb")
    shards: int = 2
    file_pattern: str = ""           # overrides the family's
    budget: bool = False             # Unsloth K-quants: all experts in RAM only with ~16 GiB to spare, else a RAM budget
    vision: bool = True
    ple_shard: int | None = 2        # the shard that holds per_layer_token_embd (None: the engine finds it from shard 1)
    notes: str = ""
    aliases: tuple = field(default=())

    @property
    def tag(self) -> str:
        return FAMILIES[self.family].tag_prefix + self.name


# --------------------------------------------------------------------------------------------------------- families
FAMILIES = {
    "qwen": Family(
        key="qwen", title="Qwen3.8-Flash-Next", model_name="qwen3.8-flash-next",
        repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", subdir=True,
        file_pattern="Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-0000{n}.gguf", files=ISTA_FILES,
        mmproj_repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        license="Qwen Community License 1.0: https://huggingface.co/Qwen/Qwen3.8-Flash-Next"),
    # ISTA-DASLab's expert-pruned release: half of each layer's experts, kept for code, tools and pictures
    "coder": Family(
        key="coder", title="Qwen3.8-Flash-Next Coder", model_name="qwen3.8-flash-next-coder",
        repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", subdir=True,
        file_pattern="Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-0000{n}.gguf", files=CODER_FILES,
        mmproj_repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        tag_prefix="coder-", profile="expert-profile-coder.bin",
        license="Apache-2.0 for the Coder release; the base model is under the Qwen Community License 1.0"),
    # Unsloth's 4- to 8-bit files of the original model.  Their pack needs --compat-bf16 (Q8_0 hyper-connection
    # projections become BF16, the form the engine reads); the picture reader is the ISTA repository's file.
    "unsloth": Family(
        key="unsloth", title="Qwen3.8-Flash-Next (Unsloth)", model_name="qwen3.8-flash-next-unsloth",
        repo="unsloth/Qwen3.8-Flash-Next-GGUF", subdir=True,
        file_pattern="Qwen3.8-Flash-Next-{q}-0000{i}-of-0000{n}.gguf", files=UNSLOTH_FILES,
        mmproj_repo="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        tag_prefix="unsloth-", pack_args=("--compat-bf16",),
        license="Qwen Community License 1.0 (the base model's): https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF"),
}

# --------------------------------------------------------------------------------------------------------- quants
# setup.py MODELS: download_gb / ram_gb / arena_gb are its numbers.  "ram_gb" is the RAM it asks for; the experts
# alone ("arena_gb") are pinned in RAM and the CPU computes the ones the GPU's expert cache does not hold.
QUANTS = {
    "Q2_0": Quant("Q2_0", "qwen", "2-bit, the fastest", 66.4, 48, 34.0),
    "IQ2_XS": Quant("IQ2_XS", "qwen", "2-bit i-quant, a little better quality, close in speed", 68.0, 48, 35.5),
    "IQ3_XXS": Quant("IQ3_XXS", "qwen", "3-bit i-quant, better quality, slower (more CPU work per token)", 75.8, 60, 42.9),
    "IQ3_S": Quant("IQ3_S", "qwen", "3.5-bit i-quant, the best quality (matches the full model), the slowest; "
                   "needs a 64 GB PC with little else running", 83.6, 62, 50.3),
    # the Coder release: 256 of the 512 experts kept, stored like IQ3_S
    "IQ1_M": Quant("IQ1_M", "coder", "the Coder's only size: half the experts (code, tools, images kept), stored like "
                   "IQ3_S (3.5 bits); weaker outside coding", 58.4, 32, 23.4),
    # Unsloth Dynamic quants: GGUF-in-place, a RAM budget of the most-used experts when the RAM is short
    "UD-IQ4_XS": Quant("UD-IQ4_XS", "unsloth", "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in "
                       "quality; on a PC with less than ~80 GB of RAM part of its experts are read from the SSD",
                       93.7, 48, 59.5, shards=3, budget=True, ple_shard=None),
    "UD-Q4_K_XL": Quant("UD-Q4_K_XL", "unsloth", "4-bit (Unsloth Dynamic): K-quant experts; ~94 GB of RAM keeps all of "
                        "them, a 64 GB PC keeps a budget of them and reads the rest from the SSD (single-digit tok/s "
                        "to ~8 tok/s)", 111.3, 48, 77.0, shards=4, budget=True, ple_shard=None),
    "UD-Q5_K_XL": Quant("UD-Q5_K_XL", "unsloth", "5-bit (Unsloth Dynamic), close to Q8_0 in quality; for 128 GB+ PCs",
                        158.3, 108, 98.4, shards=6, budget=True, ple_shard=None),
    "UD-Q6_K_XL": Quant("UD-Q6_K_XL", "unsloth", "6-bit (Unsloth Dynamic); for 128 GB+ PCs",
                        169.2, 118, 109.2, shards=6, budget=True, ple_shard=None),
    "Q8_0": Quant("Q8_0", "unsloth", "8-bit, the reference quality (every expert Q8_0); for a 192 GB PC",
                  188.2, 136, 128.3, shards=6, budget=True, ple_shard=None),
}

#: the quant a plain install picks (the fastest; fits a 64 GB PC with room to spare)
DEFAULT_QUANT = "Q2_0"


def resolve_quant(name: str) -> Quant:
    """A quant by name, case-insensitive ("q2_0", "ud-q4_k_xl"); ValueError lists the choices."""
    key = (name or "").strip().lower().replace("-", "_")
    for q in QUANTS.values():
        if q.name.lower().replace("-", "_") == key or key in tuple(a.lower() for a in q.aliases):
            return q
    raise ValueError(f"unknown quant {name!r}; choose one of: " + ", ".join(QUANTS))


def family_of(q: Quant) -> Family:
    return FAMILIES[q.family]


def shard_names(q: Quant) -> list[str]:
    fam = family_of(q)
    pattern = q.file_pattern or fam.file_pattern
    return [pattern.format(q=q.name, i=i, n=q.shards) for i in range(1, q.shards + 1)]


def _remote(fam: Family, repo: str, name: str, subdir: str, files: dict) -> RemoteFile:
    size, sha = files[name]
    return RemoteFile(repo=repo, revision=HF_REVISIONS[repo], path=(f"{subdir}/" if subdir else "") + name,
                      size=size, sha256=sha)


def shard_files(q: Quant) -> list[RemoteFile]:
    """The model's GGUF shards as pinned remote files (KeyError: a name missing from the tables, a bug)."""
    fam = family_of(q)
    return [_remote(fam, fam.repo, n, q.name if fam.subdir else "", fam.files) for n in shard_names(q)]


def mmproj_file(q: Quant) -> RemoteFile:
    """The picture reader (0.9 GB); the repository root of the family's mmproj repository."""
    fam = family_of(q)
    files = ISTA_FILES if fam.mmproj_repo == FAMILIES["qwen"].repo else CODER_FILES
    return _remote(fam, fam.mmproj_repo, MMPROJ, "", files)


def total_download_bytes(q: Quant, vision: bool = False) -> int:
    n = sum(f.size for f in shard_files(q))
    return n + (mmproj_file(q).size if vision else 0)
