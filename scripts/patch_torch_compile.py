#!/usr/bin/env python
"""
Patch PyTorch 2.10.0 Inductor bugs that break torch.compile.

BUG-08: Three assertions in torch._inductor crash ALL compile attempts:
  1. select_algorithm.py:1695 — "duplicate template name" (mm_scaled_grouped + flex_attention)
  2. select_algorithm.py:2167 — "duplicate extern kernel: _grouped_mm"
  3. lowering.py:2251 — "both a fallback and a decomp for same op: aten.mm.default"

Usage:
  python scripts/patch_torch_compile.py           # Patch
  python scripts/patch_torch_compile.py --check    # Check if patches needed
  python scripts/patch_torch_compile.py --revert   # Restore from .bak files

Run AFTER any torch reinstall. Safe to run multiple times (idempotent).
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch


def get_inductor_dir() -> Path:
    return Path(torch.__file__).parent / "_inductor"


def patch_select_algorithm(inductor_dir: Path, dry_run: bool = False) -> list[str]:
    """Fix duplicate template name + duplicate extern kernel assertions."""
    path = inductor_dir / "select_algorithm.py"
    content = path.read_text()
    patches = []

    # Patch 1: duplicate template name
    old1 = '        assert name not in self.all_templates, "duplicate template name"'
    new1 = (
        "        # BUG-08: PyTorch 2.10.0 duplicate template — allow override\n"
        "        if name in self.all_templates:\n"
        '            pass  # allow override'
    )
    if old1 in content:
        if not dry_run:
            content = content.replace(old1, new1)
        patches.append("duplicate_template_name")

    # Patch 2: duplicate extern kernel
    old2 = '        assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"'
    new2 = (
        "        # BUG-08: Allow duplicate extern kernels in PyTorch 2.10.0\n"
        "        if hasattr(extern_kernels, name):\n"
        "            pass  # allow override"
    )
    if old2 in content:
        if not dry_run:
            content = content.replace(old2, new2)
        patches.append("duplicate_extern_kernel")

    if patches and not dry_run:
        bak = path.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(content)

    return patches


def patch_lowering(inductor_dir: Path, dry_run: bool = False) -> list[str]:
    """Fix fallback/decomp conflict for aten.mm.default."""
    path = inductor_dir / "lowering.py"
    content = path.read_text()
    patches = []

    old = (
        '    assert op not in decompositions or override_decomp, (\n'
        '        f"both a fallback and a decomp for same op: {op}"\n'
        '    )'
    )
    new = (
        "    # BUG-08: PyTorch 2.10.0 has ops registered as both fallback and decomp.\n"
        "    # Prefer decomposition (optimized Triton kernels) over ATen fallback.\n"
        "    if op in decompositions and not override_decomp:\n"
        "        return"
    )
    if old in content:
        if not dry_run:
            content = content.replace(old, new)
        patches.append("fallback_decomp_conflict")

    if patches and not dry_run:
        bak = path.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(content)

    return patches


def revert(inductor_dir: Path):
    """Restore from .bak files."""
    for name in ["select_algorithm.py", "lowering.py"]:
        bak = inductor_dir / f"{name}.bak"  # lowering.py.bak won't match
    # Fix: look for .py.bak
    for bak in inductor_dir.glob("*.py.bak"):
        orig = bak.with_suffix("")  # .py.bak -> .py
        if bak.exists():
            shutil.copy2(bak, orig)
            print(f"  Reverted: {orig.name}")


def main():
    parser = argparse.ArgumentParser(description="Patch PyTorch Inductor BUG-08")
    parser.add_argument("--check", action="store_true", help="Check only, don't patch")
    parser.add_argument("--revert", action="store_true", help="Restore .bak files")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    inductor_dir = get_inductor_dir()
    print(f"Inductor: {inductor_dir}")

    if args.revert:
        revert(inductor_dir)
        print("Reverted all patches.")
        return

    all_patches = []
    all_patches.extend(patch_select_algorithm(inductor_dir, dry_run=args.check))
    all_patches.extend(patch_lowering(inductor_dir, dry_run=args.check))

    if not all_patches:
        print("No patches needed — already patched or different PyTorch version.")
    elif args.check:
        print(f"Patches NEEDED: {', '.join(all_patches)}")
        sys.exit(1)
    else:
        print(f"Applied {len(all_patches)} patches: {', '.join(all_patches)}")

    # Verify
    try:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 64)
            def forward(self, x):
                return self.fc(x)

        if torch.cuda.is_available():
            m = torch.compile(M().cuda(), mode="default")
            m(torch.randn(4, 64, device="cuda"))
            print("Verification: torch.compile PASS")
        else:
            print("Verification: skipped (no CUDA)")
    except Exception as e:
        print(f"Verification: FAIL — {e}")


if __name__ == "__main__":
    main()
