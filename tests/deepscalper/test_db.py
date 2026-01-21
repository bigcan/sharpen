
import unittest
from unittest.mock import MagicMock
from finrl_pro_ds.data.db import DatabaseClient, LOBSnapshot

class TestDatabaseClientLOB(unittest.TestCase):
    def setUp(self):
        self.mock_connect = MagicMock()
        self.client = DatabaseClient(dsn="mock://", connect=self.mock_connect)
        self.mock_conn = self.mock_connect.return_value.__enter__.return_value
        self.mock_cur = self.mock_conn.cursor.return_value.__enter__.return_value

    def test_upsert_lob(self):
        snaps = [
            LOBSnapshot("2023-01-01", "BTC", 1, 10.0, 1.0, 11.0, 1.0, "s")
        ]
        self.client.upsert_lob_snapshots(snaps)
        self.mock_cur.executemany.assert_called()
        args = self.mock_cur.executemany.call_args
        # args[0] is (sql, payloads)
        self.assertIn("INSERT INTO lob_snapshots", args[0][0])
        self.assertEqual(len(args[0][1]), 1)

    def test_fetch_lob(self):
        # Mock rows
        self.mock_cur.fetchall.return_value = [
            ("2023-01-01", "BTC", 1, 10.0, 1.0, 11.0, 1.0, "s")
        ]
        res = self.client.fetch_lob_snapshots(ticker="BTC")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].ticker, "BTC")
        self.mock_cur.execute.assert_called()
        sql = self.mock_cur.execute.call_args[0][0]
        self.assertIn("SELECT", sql)
        self.assertIn("FROM lob_snapshots", sql)

if __name__ == "__main__":
    unittest.main()
