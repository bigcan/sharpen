"""Module: test_placeholder
Purpose: Verify the pytest discovery pipeline for the FinRL Pro scaffold."""


def test_scaffold_imports() -> None:
    """Ensure the FinRL Pro package can be imported."""
    import finrl_pro

    assert hasattr(finrl_pro, "__all__")
