# Ralph Mission Control: User Guide

This guide explains how to monitor and control the DeepScalper Ralph Autonomous Workflow directly from Notion.

## 📊 Dashboard Overview

The **Ralph Mission Control** database provides real-time visibility into the training pipeline.

| Column | Description |
|--------|-------------|
| **Name** | Title of the log entry (e.g., "Ralph Status"). |
| **Status** | Current state of the workflow: <br>🟢 `Running`: Active <br>🟡 `Paused`: Waiting for user command <br>🔴 `Stopped`: Terminated <br>🔴 `Error`: Failed |
| **Sharpe** | Latest **Validation Sharpe Ratio** from the backtest. |
| **Command** | **USER INPUT**: Control the workflow here (see below). |
| **Logs** | Recent log snippets (e.g., "Run ID: ... State: running"). |
| **Timestamp**| Time of last update. |

---

## 🎮 Remote Control

You can control the running Ralph workflow by changing the **Command** property in Notion. The agent checks this command every **60 seconds** during the monitoring phase.

### Available Commands

1. **Pause** (🟠)
    - **Action**: Select `Pause` in the Command column.
    - **Effect**: Ralph will effectively "sleep" and stop progressing to the next iteration or phase. It will verify the pause by updating the Status to `Paused`.
    - **Use Case**: Determine if you want to inspect a run before it finishes, or hold off on a new deployment.

2. **Continue** (🔵)
    - **Action**: Select `Continue` (or clear the field) in the Command column.
    - **Effect**: If Paused, Ralph will resume normal operations.
    - **Use Case**: Resuming after a Pause.

3. **Stop** (🔴)
    - **Action**: Select `Stop` in the Command column.
    - **Effect**: Ralph will **terminate** the entire workflow process on the next check.
    - **Use Case**: Emergency stop or deciding enough iterations have passed.

---

## 🛠 Troubleshooting

- **Updates stopped?**: Check if the physical process on the machine is still running (or if the remote machine is online).
- **Commands not working?**: Ralph only checks commands during the **Monitoring Phase** (while waiting for a run). If it is in the middle of "Deploying" or "Generating Reports", it may not react immediately.

---

## 🔗 Quick Links

- [Ralph Mission Control Database](https://www.notion.so/2fedf15f7333800dbcb2000b885189da) (Direct Link)
