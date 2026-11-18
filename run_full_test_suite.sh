#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=~/disaggregated-kv-transport-engine
BUILD_DIR="$REPO_DIR/build"

echo "================================================================================"
echo "  RUNNING FULL DIAGNOSTIC & BENCHMARK SUITE: disaggregated-kv-transport-engine"
echo "  Hardware: NVIDIA GeForce RTX 5060 (8GB GDDR7, SM 12.0) | AMD Ryzen 5 5600X"
echo "================================================================================"

cd "$BUILD_DIR"

# 1. Executar os Testes Unitários de Diagnóstico (3/3)
echo ""
echo "[STEP 1/3] Running Unit & Numerical Verification Tests..."
./disagg_kv_test

# 2. Executar o Benchmark de Alta Performance (1,000 steps)
echo ""
echo "[STEP 2/3] Running Empirical Benchmark (Compression, Ring Buffer & Transfer Latency)..."
./disagg_kv_benchmark

# 3. Gerar os Gráficos Dark-Mode a partir dos CSVs Empíricos
echo ""
echo "[STEP 3/3] Generating Telemetry Plots from CSV Outputs..."
python3 ../scripts/generate_plots.py

echo ""
echo "================================================================================"
echo "  TEST & BENCHMARK SUITE COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Generated CSV files:"
ls -lh transfer_latencies.csv benchmark_summary.csv

echo ""
echo "Generated Plot Assets in assets/:"
ls -lh ../assets/*.png
echo "================================================================================"
