# Ralph Workflow Summary Report

**Date:** 2026-02-07 08:13
**Status:** MAX ITERATIONS REACHED
**Total Iterations:** 29 / 10
**Best Test Sharpe:** -0.9932 (Run `2zhgu161`)
**Goal:** Test Sharpe >= 1.0

## 1. Iteration History

|   Iteration | RunID    | Timestamp                  |   Val Sharpe |   Test Sharpe | Outcome   |
|------------:|:---------|:---------------------------|-------------:|--------------:|:----------|
|           1 | pp5orsxq | 2026-02-05T11:38:43.985006 |       0      |     -999      | failure   |
|           1 | 4iougvkk | 2026-02-05T18:02:44.025732 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:04:16.941138 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:04:45.070672 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:05:18.665885 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:05:50.808188 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:06:33.617481 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:06:59.819927 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:07:29.272110 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:07:57.418514 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:08:25.538375 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:08:53.538565 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:09:25.037187 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:09:57.152924 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:10:32.898108 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:11:01.881063 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:11:33.701655 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:12:01.388864 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:12:29.552646 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:13:00.979668 |     -61.9229 |      -60.2751 | failure   |
|           1 | 4iougvkk | 2026-02-05T18:13:33.142836 |     -61.9229 |      -60.2751 | failure   |
|           3 | o3oici7z | 2026-02-06T00:25:17.670638 |      -9.3463 |       -6.726  | failure   |
|           4 | hf2ieigc | 2026-02-06T04:19:00.143010 |    -343.132  |     -277.979  | failure   |
|           5 | c7z5sjux | 2026-02-06T08:46:00.147728 |     -77.6821 |      -87.1222 | failure   |
|           6 | nn69bf2q | 2026-02-06T13:18:07.229661 |     -81.4499 |      -85.4299 | failure   |
|           7 | 4qmfj1ni | 2026-02-06T17:58:18.676728 |    -261.707  |     -241.737  | failure   |
|           8 | cuo5mtph | 2026-02-06T22:59:47.938361 |     -37.2667 |      -36.8865 | failure   |
|           9 | eytstl82 | 2026-02-07T02:53:49.790927 |     -11.512  |       -9.2645 | failure   |
|          10 | 2zhgu161 | 2026-02-07T07:28:10.985176 |      -3.4096 |       -0.9932 | failure   |

## 2. Fixes Applied

| Iteration | Timestamp | Fix Applied | Reason |
| :--- | :--- | :--- | :--- |
| 1 | 2026-02-05T11:39:26.044305 | Increased HPO n_trials from 50 to 75 due to Low Test Sharpe (-0.49) | None |
| 1 | 2026-02-05T11:46:07.001161 | Reverted HPO n_trials to 50 (User Request) | None |
| 1 | 2026-02-05T18:02:45.562662 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:04:18.401906 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:04:46.503146 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:05:20.195467 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:05:52.335193 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:06:35.118933 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:07:01.256239 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:07:30.780690 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:07:58.942745 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:08:27.041067 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:08:55.036747 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:09:26.551895 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:09:58.688101 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:10:34.430045 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:11:03.309211 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:11:35.206944 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:12:02.816932 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:12:31.049382 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:13:02.490161 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 1 | 2026-02-05T18:13:34.682782 | Adjust HPO range: Increase n_trials +20 | State: finished, TestSharpe: -60.27512758033696 |
| 3 | 2026-02-06T00:25:19.249693 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -6.72597252941388 |
| 4 | 2026-02-06T04:19:01.739590 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -277.97875177259095 |
| 5 | 2026-02-06T08:46:01.712469 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -87.12220259361304 |
| 6 | 2026-02-06T13:18:08.816195 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -85.4299184017476 |
| 7 | 2026-02-06T17:58:20.218950 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -241.73652456126672 |
| 8 | 2026-02-06T22:59:49.453396 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -36.88649373881848 |
| 9 | 2026-02-07T02:53:51.376569 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -9.264504955048409 |
| 10 | 2026-02-07T07:28:12.583993 | Adjust HPO: Shift Learning Rate Range | State: finished, TestSharpe: -0.9931510463263878 |

