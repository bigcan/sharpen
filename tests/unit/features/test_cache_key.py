from finrl_pro_ds.data.cache import feature_cache_key


def test_feature_cache_key_stable_order():
    ds = "dvc://datasets/demo"
    cfg1 = {"families": {"trend": True, "momentum": False}, "advanced": {"fracdiff": {"enable": True, "d": 0.5}}}
    cfg2 = {"advanced": {"fracdiff": {"d": 0.5, "enable": True}}, "families": {"momentum": False, "trend": True}}
    k1 = feature_cache_key(ds, cfg1)
    k2 = feature_cache_key(ds, cfg2)
    assert k1 == k2

