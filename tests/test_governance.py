import os
import shutil
from finrl_pro_ds.governance.research_logger import ResearchLogger
from finrl_pro_ds.governance.leaderboard import Leaderboard

def test_governance():
    # Setup
    if os.path.exists("test_log.md"): os.remove("test_log.md")
    if os.path.exists("test_leaderboard.json"): os.remove("test_leaderboard.json")
    
    # Test Logger
    print("Testing ResearchLogger...")
    logger = ResearchLogger(log_path="test_log.md")
    logger.log_experiment("Test Run 1", {"lr": 0.001}, {"Sharpe": 1.2}, status="PASSED")
    
    with open("test_log.md", "r") as f:
        content = f.read()
        assert "Test Run 1" in content
        assert "PASSED" in content
        print("Logger Passed.")

    # Test Leaderboard
    print("Testing Leaderboard...")
    lb = Leaderboard(path="test_leaderboard.json", primary_metric="Sharpe")
    lb.add_entry("Model_A", {"Sharpe": 1.5})
    lb.add_entry("Model_B", {"Sharpe": 2.0})
    lb.add_entry("Model_C", {"Sharpe": 1.0})
    
    # Check Order
    best = lb.get_best_model()
    assert best["model"] == "Model_B"
    assert best["metrics"]["Sharpe"] == 2.0
    print("Leaderboard Sort Passed.")
    
    # Check Persistence
    lb2 = Leaderboard(path="test_leaderboard.json", primary_metric="Sharpe")
    assert len(lb2.entries) == 3
    assert lb2.get_top_n(1)[0]["model"] == "Model_B"
    print("Leaderboard Persistence Passed.")
    
    # Cleanup
    os.remove("test_log.md")
    os.remove("test_leaderboard.json")
    print("All Governance Tests Passed.")

if __name__ == "__main__":
    test_governance()
