"""Tests for AlphaSeek HPO runner — search space, window scheduling, smoke HPO."""




# ===========================================================================
# Search Space Tests (no data dependency)
# ===========================================================================

class TestSearchSpace:
    """Test HPO search space definition."""

    def test_search_space_keys(self):
        """Verify all 10 expected HPO dimensions are present."""
        import optuna

        # Create a mock trial
        study = optuna.create_study(direction="maximize")

        from scripts.alphaseek_hpo_runner import define_search_space

        trial = study.ask()
        params = define_search_space(trial)

        expected_keys = {
            "learning_rate", "soft_update_tau", "reward_scale", "gamma",
            "net_dims", "batch_size", "step_gap", "explore_rate",
            "clip_grad_norm", "stop_loss_thresh",
        }
        assert set(params.keys()) == expected_keys

    def test_search_space_ranges(self):
        """Verify sampled values are within expected ranges."""
        import optuna

        study = optuna.create_study(direction="maximize")

        from scripts.alphaseek_hpo_runner import define_search_space

        # Sample multiple trials to check range
        for _ in range(10):
            trial = study.ask()
            params = define_search_space(trial)

            assert 1e-6 <= params["learning_rate"] <= 1e-3
            assert 1e-6 <= params["soft_update_tau"] <= 5e-3
            assert 100 <= params["reward_scale"] <= 10000
            assert 0.95 <= params["gamma"] <= 0.999
            assert params["net_dims"] in ["128,128,128", "256,256", "128,128"]
            assert params["batch_size"] in [256, 512, 1024]
            assert params["step_gap"] in [1, 2, 4, 8]
            assert 0.005 <= params["explore_rate"] <= 0.05
            assert 1.0 <= params["clip_grad_norm"] <= 10.0
            assert 5e-5 <= params["stop_loss_thresh"] <= 5e-3


# ===========================================================================
# Window Schedule Tests (no data dependency)
# ===========================================================================

class TestWindowSchedule:
    """Test walk-forward window construction."""

    def test_default_schedule(self):
        from scripts.alphaseek_hpo_runner import build_window_schedule

        windows = build_window_schedule(n_segments=13)
        # 13 segments, 8+1+1=10 window size, slide by 1 → 4 windows
        assert len(windows) == 4

    def test_window_assignment(self):
        from scripts.alphaseek_hpo_runner import build_window_schedule

        windows = build_window_schedule(
            n_segments=13, train_segments=8, val_segments=1,
            test_segments=1, slide_by=1,
        )

        # W0: train=[0-7], val=[8], test=[9]
        assert windows[0]["train_segs"] == list(range(0, 8))
        assert windows[0]["val_segs"] == [8]
        assert windows[0]["test_segs"] == [9]

        # W3: train=[3-10], val=[11], test=[12]
        assert windows[3]["train_segs"] == list(range(3, 11))
        assert windows[3]["val_segs"] == [11]
        assert windows[3]["test_segs"] == [12]

    def test_no_overlap(self):
        from scripts.alphaseek_hpo_runner import build_window_schedule

        windows = build_window_schedule(n_segments=13)
        for w in windows:
            train = set(w["train_segs"])
            val = set(w["val_segs"])
            test = set(w["test_segs"])
            # No segment appears in multiple roles within same window
            assert len(train & val) == 0
            assert len(train & test) == 0
            assert len(val & test) == 0

    def test_small_dataset(self):
        from scripts.alphaseek_hpo_runner import build_window_schedule

        # Too few segments for any window
        windows = build_window_schedule(n_segments=5, train_segments=8)
        assert len(windows) == 0

    def test_slide_by_2(self):
        from scripts.alphaseek_hpo_runner import build_window_schedule

        windows = build_window_schedule(n_segments=13, slide_by=2)
        assert len(windows) == 2  # start at 0, 2 → windows at 0 and 2


# ===========================================================================
# Agent Map Tests (no data dependency)
# ===========================================================================

class TestAgentMap:
    def test_all_agent_names_resolve(self):
        from finrl_pro_ds.alphaseek.agents import AGENT_MAP

        for name in ["D3QN", "DoubleDQN", "TwinD3QN"]:
            assert name in AGENT_MAP
            cls = AGENT_MAP[name]
            assert cls is not None
