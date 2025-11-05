# FinRL Pro Module Scaffolding Guide

Use the FinRL Pro CLI to generate new modules inside the `finrl_pro` namespace
without touching upstream FinRL Podracer directories.

```bash
python -m finrl_pro scaffold finrl_pro.agents.new_strategy --doc "New strategy agent."
```

The scaffolder will:

- Ensure the module path starts with `finrl_pro.` before writing files.
- Create missing package directories and `__init__.py` files.
- Generate a Python module populated with the supplied docstring.

After scaffolding, register the module in `finrl_pro/configs/modules.yaml` and
update the corresponding Spec-Kit tasks before implementation.
