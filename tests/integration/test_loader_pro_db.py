import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("FINRL_PRO_DB_DSN"), reason="Requires FINRL_PRO_DB_DSN to run DB integration tests",
)

from finrl_pro.data.loader_pro import ProFeatureAssembler


def test_assemble_from_snapshot_smoke():
    # Requires preexisting snapshot in DB; this is a smoke test for CI environments with a seeded DB.
    snapshot_id = os.getenv("FINRL_PRO_TEST_SNAPSHOT_ID")
    if not snapshot_id:
        pytest.skip("FINRL_PRO_TEST_SNAPSHOT_ID not set")
    features_cfg = {
        "families": {"trend": True, "momentum": True, "vol": True, "volume": False},
        "advanced": {"fracdiff": {"enable": False}, "wavelet": {"enable": False}},
    }
    asm = ProFeatureAssembler(dsn=os.getenv("FINRL_PRO_DB_DSN")).assemble_from_snapshot(
        snapshot_id=snapshot_id, features_cfg=features_cfg
    )
    assert asm.price_ary.ndim == 2 and asm.tech_ary.ndim == 2
    assert len(asm.feature_set_id) > 0

