# gemini.md — FinRL Pro Project Overview

Last updated: 2025-11-19

This document provides a high-level technical overview of the FinRL_Podracer project for developers. It describes the project's architecture, core components, and key functionalities.

## Project Goal

The FinRL_Podracer project provides a modular, institutional-grade platform for developing, training, and managing financial reinforcement learning (DRL) trading strategies. It is designed to enforce reproducibility, manage risk, and streamline the research-to-production workflow.

## Core Architecture

The system is built on a three-layer architecture, ensuring a clear separation of concerns between the underlying RL engine, the professional extension layer, and the ensemble execution service.

1.  **Core Engine (`FinRLPodracer/`)**
    *   **Description:** This layer contains the upstream, open-source reinforcement learning libraries: `elegantrl` and `finrl`. It provides the fundamental DRL algorithms, environments, and core financial data processing capabilities.
    *   **Usage:** This layer is treated as a foundational dependency and should not be modified. This separation allows the core engine to be updated independently of the custom business logic.

2.  **Extension Layer (`finrl_pro/`)**
    *   **Description:** This is the primary development and research interface. It is a comprehensive Python package that wraps the Core Engine, adding a suite of professional-grade features for robust strategy development.
    *   **Usage:** All new research, agent development, and workflow customization occurs in this layer. It is designed to be the main entry point for users of the platform.

3.  **Ensemble Service (`Podracer/`)**
    *   **Description:** A separate, containerized application designed to run ensemble strategies. It uses Docker and Docker Compose to orchestrate parallel training and evaluation of multiple RL models.
    *   **Usage:** This service is used for advanced, scaled-up experiments and for combining multiple strategies to improve performance and robustness.

## Key Functionality (`finrl_pro`)

The `finrl_pro` extension layer provides the following key features:

-   **Reproducibility:** A "fingerprinting" system captures all parameters, configurations, and data hashes of an experiment. This allows any experiment to be perfectly replicated with a single command.
-   **Data Management:** A robust data system utilizing a TimescaleDB/PostgreSQL database for versioned data snapshots and a feature store. This ensures point-in-time correctness and abstracts data sources away from the training logic.
-   **MLOps & Risk Management:**
    *   **Experiment Tracking:** Deep integration with MLflow for logging parameters, metrics, and artifacts.
    *   **Structured Logging:** Centralized logging for better observability.
    *   **Risk Guardrails:** A configurable risk management policy engine (e.g., max drawdown, leverage limits) that actively enforces constraints during training and evaluation.
-   **Evaluation & Reporting:**
    *   **Advanced Evaluation:** Built-in support for walk-forward testing and other sophisticated evaluation schemes.
    *   **Automated Reporting:** Tools to generate compliance-ready reports based on experiment results.

## Developer Workflow Overview

A typical developer workflow involves interacting primarily with the `finrl_pro` package:

1.  **Data:** Ingest and manage financial data using the database snapshot system.
2.  **Configuration:** Define experiment parameters, agent hyperparameters, and risk profiles using YAML configuration files.
3.  **Training:** Run training jobs using the `finrl_pro` command-line tools, which handle experiment tracking, fingerprinting, and risk enforcement.
4.  **Evaluation:** Analyze results using the provided evaluation tools and generate reports.
5.  **Iteration:** Use the tracked experiments and reproducible results to iterate on and improve trading strategies.
