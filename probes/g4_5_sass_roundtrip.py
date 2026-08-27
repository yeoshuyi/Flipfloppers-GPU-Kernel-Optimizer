#!/usr/bin/env python3
"""
G4.5 Phase 0 -- CuAssembler round-trip feasibility on sm_89 / CUDA 13.1.

WHAT THIS IS TESTING. docs/PROGRESS.md step 37 left G4.4's hand-written
mma.sync FP16-accumulate GEMM at 55.1% of its own tier ceiling (181.93 TF of
330.3 TF) while cuBLASLt sits at 91.2% of its (2x lower) tier, and diagnosed
the remaining gap as needing CUTLASS-grade kernel engineering. SASS-level
instruction reordering (CuAsmRL's idea) is a narrower, cheaper lever than that
rewrite -- but it exists at all only if an external SASS assembler can
round-trip this kernel's cubin WITHOUT CHANGING IT. That is the only question
this script answers. It ships no optimisation and asserts no speedup.

THE ROUND TRIP MUST BE A NO-OP. cubin -> .cuasm -> cubin with zero edits must
(a) assemble at all and (b) reproduce the original .text byte for byte.
Anything less and every subsequent edit is uninterpretable: a wrong answer
could be the edit or could be the assembler.

TWO ADAPTATIONS ARE NEEDED, AND NEITHER IS "PATCHING THE ASSEMBLER":

1. ELF e_flags layout (a CUDA-VERSION issue, not an Ada one).
   CuAssembler reads the target arch out of e_flags using the CUDA 11 layout
   (sm in the low byte, virtual arch in bits 16-23; its own comment cites
   0x500556 for sm_86). CUDA 12+ moved the sm number to bits 8-15. This
   container's nvcc 13.1 emits:
       sm_80 -> 0x06005004   sm_86 -> 0x06005604   sm_89 -> 0x06005904
                       ^^                  ^^                  ^^
   so CuAssembler reads sm = e_flags & 0xff = 4 and dies with "Invalid SM
   version 4" -- on an sm_86 cubin (an OFFICIALLY SUPPORTED arch) exactly as
   surely as on an sm_89 one. Verified both ways. The shim below rewrites only
   CuAssembler's in-memory *interpretation* of that field; the bytes on disk
   are left alone so nvdisasm still accepts the file. Four bytes of ELF
   header, no .text, no encoding tables.

2. No prebuilt sm_89 instruction repository.
   CuAssembler ships DefaultInsAsmRepos for sm_60/61/70/75/80/86 only, and
   sm_89 has no alias entry (InsAsmReposAliasDict = {62:61, 72:75, 87:86}).
   Building one is the tool's OWN DOCUMENTED mechanism (UserGuide.md
   "Instruction Assembler Repository": repos.update(feeder) over observed
   SASS, then save2file) -- CuAssembler solves each opcode's encoding from
   real (code, disassembly) pairs. That is auto-probing, not hand-authoring
   encoding tables, and it is exactly how the tool is meant to reach an arch
   whose repo its authors never shipped.

   repos.verify() then re-encodes every instruction it learned and compares
   against the original bytes. THAT is the honest Ada compatibility test: if
   HMMA / LDSM / LDGSTS on sm_89 encode in a way the solver cannot express,
   verify() is where it surfaces.
"""
import argparse
import hashlib
import logging
import os
import shutil
import struct
import sys
import traceback
from collections import Counter
from io import StringIO
from subprocess import check_output

E_FLAGS_OFF = 0x30  # ELF64 file header: e_flags, 4 bytes at offset 0x30

# sm numbers this shim is willing to recognise in the CUDA 12/13 layout.
KNOWN_SM = {60, 61, 70, 72, 75, 80, 86, 87, 89, 90}


def read_flags(path):
    with open(path, "rb") as f:
        f.seek(E_FLAGS_OFF)
        return struct.unpack("<I", f.read(4))[0]


def write_flags(path, value):
    with open(path, "r+b") as f:
        f.seek(E_FLAGS_OFF)
        f.write(struct.pack("<I", value))


def legacy_flags(sm):
    """The CUDA 11 layout CuAssembler parses: virtual arch bits 16-23, sm low byte."""
    return (sm << 16) | sm


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def install_eflags_shim():
    """Make CuAssembler read CUDA 12/13 e_flags correctly; on-disk bytes untouched.

    Wraps ELFFile.__init__ in place rather than rebinding the name in each
    CuAsm module, so it takes effect at every import site (CubinFile,
    CuAsmParser, utils/CubinUtils) no matter which one constructed the object.
    """
    import elftools.elf.elffile as _eff

    real_init = _eff.ELFFile.__init__
    if getattr(real_init, "_cuda13_shim", False):
        return

    def patched_init(self, *a, **kw):
        real_init(self, *a, **kw)
        f = self.header["e_flags"]
        if (f & 0xFF) not in KNOWN_SM and ((f >> 8) & 0xFF) in KNOWN_SM:
            self.header["e_flags"] = legacy_flags((f >> 8) & 0xFF)
        # CUDA 11 cubins stashed the toolkit version in e_version (e.g. 111),
        # which pyelftools left as a raw int; CUDA 13 sets it to 1, which
        # pyelftools resolves to the enum NAME 'EV_CURRENT'. CuAssembler
        # formats that field with %d. Normalise back to the integer -- the
        # value written out is then identical to what was read in.
        v = self.header["e_version"]
        if isinstance(v, str):
            self.header["e_version"] = {"EV_NONE": 0, "EV_CURRENT": 1}.get(v, 1)

    patched_init._cuda13_shim = True
    _eff.ELFFile.__init__ = patched_init


def set_section_type(path, secname, newtype):
    """Rewrite one section header's sh_type in place; return the old raw value.

    Used to hide .note.nv.tkinfo from nvdisasm 13.1, which PRETTY-PRINTS
    SHT_NOTE sections (`.tkinfo` + `.string "ptxas"` + the compiler command
    line) instead of dumping raw `.byte`s. CuAsmParser has no `.tkinfo` or
    `.string` directive and no ELF-note model at all, so that rendering cannot
    be reassembled. Presenting the same bytes as SHT_PROGBITS makes nvdisasm
    emit them as a plain byte dump, which round-trips. The original sh_type is
    restored in the output, so the final cubin is byte-for-byte a normal one.
    """
    import struct as _s
    with open(path, "r+b") as f:
        data = f.read()
        e_shoff, = _s.unpack_from("<Q", data, 0x28)
        e_shentsize, = _s.unpack_from("<H", data, 0x3A)
        e_shnum, = _s.unpack_from("<H", data, 0x3C)
        e_shstrndx, = _s.unpack_from("<H", data, 0x3E)
        strtab_off, = _s.unpack_from(
            "<Q", data, e_shoff + e_shstrndx * e_shentsize + 0x18)
        for i in range(e_shnum):
            base = e_shoff + i * e_shentsize
            sh_name, sh_type = _s.unpack_from("<II", data, base)
            end = data.index(b"\0", strtab_off + sh_name)
            name = data[strtab_off + sh_name:end].decode()
            if name == secname:
                f.seek(base + 4)
                f.write(_s.pack("<I", newtype))
                return sh_type
    return None


def text_section(path):
    from elftools.elf.elffile import ELFFile
    with open(path, "rb") as f:
        ef = ELFFile(f)
        for s in ef.iter_sections():
            if s.name.startswith(".text."):
                return s.name, s.data()
    return None, None


def opcode_histogram(cuasm_text):
    ops = Counter()
    for line in cuasm_text.splitlines():
        s = line.strip()
        if "*/" not in s:
            continue
        body = s.split("*/", 1)[1].strip()
        if not body or body.startswith("/*"):
            continue
        body = body.lstrip("@").lstrip("!")
        toks = body.split()
        if not toks:
            continue
        op = toks[0]
        if len(toks) > 1 and op[:1] in ("P", "U") and op[1:].rstrip("T").isdigit():
            op = toks[1]
        ops[op.split(".")[0]] += 1
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--cuasm-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep-notes", action="store_true",
                    help="do NOT retype .note.nv.tkinfo (shows the raw failure)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    sys.path.insert(0, args.cuasm_root)

    from CuAsm.CuAsmLogger import CuAsmLogger
    CuAsmLogger.initLogger(os.path.join(args.work, "cuasm.log"),
                           file_level=logging.INFO, stdout_level=logging.ERROR)

    install_eflags_shim()

    from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
    from CuAsm.CuInsFeeder import CuInsFeeder
    from CuAsm.CubinFile import CubinFile
    from CuAsm.CuAsmParser import CuAsmParser
    from CuAsm.utils.CubinUtils import (updateReposWithCubin, hackCubinDesc,
                                        transDescFeeder, feedBinFromCubin)
    from CuAsm.config import Config

    orig = os.path.join(args.work, "orig.cubin")
    shutil.copyfile(args.cubin, orig)
    real_flags = read_flags(orig)
    sm = (real_flags >> 8) & 0xFF
    arch = "sm_%d" % sm

    # Shim 3: hide .note.nv.tkinfo's SHT_NOTE type from nvdisasm 13.1 (see
    # set_section_type). Restored on the way out.
    SHT_PROGBITS = 1
    tkinfo_type = None
    if not args.keep_notes:
        tkinfo_type = set_section_type(orig, ".note.nv.tkinfo", SHT_PROGBITS)

    print("=" * 78)
    print("PHASE 0 -- CuAssembler round-trip on %s (CUDA 13.1 cubin)" % arch)
    print("=" * 78)
    print("cubin           : %s" % args.cubin)
    print("e_flags on disk : 0x%08x -> sm_%d (CUDA 12/13 layout, bits 8-15)"
          % (real_flags, sm))
    print("shimmed as      : 0x%08x (the CUDA 11 layout CuAssembler parses)"
          % legacy_flags(sm))
    print("orig sha256[:16]: %s" % sha(orig), flush=True)

    # ---------------------------------------------------------------- step 1
    print("\n" + "-" * 78)
    print("STEP 1: build an sm_%d instruction repository from this cubin" % sm)
    print("        (CuAssembler ships none for sm_89 and has no alias for it)")
    print("-" * 78, flush=True)
    repos_path = Config.getDefaultInsAsmReposFile(sm)
    print("target repo file: %s" % repos_path, flush=True)
    repos = CuInsAssemblerRepos(arch=arch)
    try:
        updateReposWithCubin(repos, orig, savname=repos_path)
    except Exception as e:
        traceback.print_exc()
        print("\nREPOS BUILD RAISED: %s: %s" % (type(e).__name__, e))
        print("RESULT: KILL GATE -- cannot learn sm_%d encodings." % sm)
        return 2
    print("repo entries learned: %d" % len(repos))
    if len(repos) == 0:
        print("RESULT: KILL GATE -- empty repository, nothing was learned.")
        return 2
    print("repo saved        : %s (%s)"
          % (repos_path, "yes" if os.path.isfile(repos_path) else "NO"),
          flush=True)

    # ---------------------------------------------------------------- step 2
    print("\n" + "-" * 78)
    print("STEP 2: verify() -- re-encode every learned instruction and compare")
    print("        against the ORIGINAL bytes. This is the real Ada test.")
    print("-" * 78, flush=True)
    hb = os.path.join(args.work, "hack.orig.cubin")
    hackCubinDesc(orig, hb)
    b_arch = arch.replace("_", "").upper()
    nfail = 0
    for outname in feedBinFromCubin(hb, outname=None, merge_all_kernels=False):
        sass = check_output(["nvdisasm", "-hex", "-c", "-b", b_arch,
                             outname]).decode()
        feeder = transDescFeeder(CuInsFeeder(StringIO(sass)))
        try:
            res = repos.verify(feeder)
            print("  repos.verify() returned: %r" % (res,))
        except Exception as e:
            nfail += 1
            traceback.print_exc()
            print("  VERIFY RAISED: %s: %s" % (type(e).__name__, e))
    try:
        err = repos.showErrRecords()
        print("  error records: %r" % (err,))
    except Exception as e:
        print("  showErrRecords raised: %s" % e)
    if nfail:
        print("\nRESULT: KILL GATE -- verify() failed; sm_%d encodings are not"
              " reproducible by this assembler." % sm)
        return 3

    # ---------------------------------------------------------------- step 3
    print("\n" + "-" * 78)
    print("STEP 3: cubin -> .cuasm")
    print("-" * 78, flush=True)
    cuasm_path = os.path.join(args.work, "rt.cuasm")
    try:
        cf = CubinFile(orig)
        cf.saveAsCuAsm(cuasm_path)
    except Exception:
        traceback.print_exc()
        print("\nRESULT: KILL GATE -- disassembly to .cuasm failed.")
        return 4
    text = open(cuasm_path).read()
    print("  .cuasm: %d bytes, %d lines" % (len(text), text.count("\n")))

    ops = opcode_histogram(text)
    print("\n  SASS opcode histogram (top 20 of %d distinct):" % len(ops))
    for op, n in ops.most_common(20):
        print("    %-16s %d" % (op, n))
    print("\n  instructions this investigation cares about:")
    for key in ("HMMA", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR",
                "STG", "LDG", "STS", "LDS"):
        print("    %-12s %d" % (key, ops.get(key, 0)))
    sys.stdout.flush()

    # ---------------------------------------------------------------- step 4
    print("\n" + "-" * 78)
    print("STEP 4: .cuasm -> cubin, ZERO edits")
    print("-" * 78, flush=True)
    rt = os.path.join(args.work, "roundtrip.cubin")
    try:
        cap = CuAsmParser()
        cap.parse(cuasm_path)
        cap.saveAsCubin(rt)
    except Exception:
        traceback.print_exc()
        print("\nRESULT: KILL GATE -- reassembly failed.")
        return 5

    write_flags(rt, real_flags)  # restore the CUDA 13 layout for the driver
    if tkinfo_type is not None:
        set_section_type(rt, ".note.nv.tkinfo", tkinfo_type)
        set_section_type(orig, ".note.nv.tkinfo", tkinfo_type)
    print("  wrote %s (%d bytes)" % (rt, os.path.getsize(rt)))
    print("  orig      sha256[:16]: %s" % sha(orig))
    print("  roundtrip sha256[:16]: %s" % sha(rt))
    print("  whole-file identical : %s" % (sha(rt) == sha(orig)))

    na, da = text_section(orig)
    nb, db = text_section(rt)
    same_text = (da == db)
    print("  .text identical      : %s  (%s: %d B vs %s: %d B)"
          % (same_text, na, len(da or b""), nb, len(db or b"")))
    if not same_text and da and db and len(da) == len(db):
        diffs = [i for i in range(len(da)) if da[i] != db[i]]
        print("  differing .text bytes: %d / %d (first: %s)"
              % (len(diffs), len(da), [hex(d) for d in diffs[:16]]))
        print("  -> instruction slots affected: %s"
              % sorted({d // 16 for d in diffs})[:20])

    print("\n" + "=" * 78)
    if same_text:
        print("RESULT: ROUND TRIP IS A NO-OP AT THE .text LEVEL. Phase 0's")
        print("        assembler gate is PASSED. Runtime + numerics still")
        print("        have to be confirmed on the GPU (g4_5_sass_run.py).")
    else:
        print("RESULT: reassembly succeeded but .text CHANGED. Every later")
        print("        edit would be confounded by this. Treat as KILL GATE")
        print("        unless the diff is explained.")
    print("=" * 78)
    return 0 if same_text else 6


if __name__ == "__main__":
    sys.exit(main())
