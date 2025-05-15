#!/bin/bash
#
# Run script for Chess-PINNs-Forecasting MVP
#
# This script provides convenient shortcuts for running different analysis scenarios
# with the Chess-PINNs-Forecasting MVP.

# Ensure the script runs from its directory
cd "$(dirname "$0")"

# Set default parameters
EPOCHS=20
BATCH_SIZE=16
FORECAST_YEARS=5
GM_LIST="GM001 GM002 GM003"  # Default to top 3 GMs
MILESTONE_START=2400
MILESTONE_END=2900
MILESTONE_STEP=100

# Help function
print_help() {
    echo "Chess-PINNs-Forecasting Analysis Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --fast             Run with minimal epochs for quick testing (5 epochs)"
    echo "  --full             Run with all 7 grandmasters (default is top 3)"
    echo "  --linear-only      Train only linear models (faster)"
    echo "  --nonlinear-only   Train only nonlinear models"
    echo "  --load             Load saved models instead of training new ones"
    echo "  --epochs N         Set number of training epochs (default: $EPOCHS)"
    echo "  --batch-size N     Set batch size (default: $BATCH_SIZE)"
    echo "  --forecast N       Set forecast years (default: $FORECAST_YEARS)" 
    echo "  --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --fast          Run quick test with minimal epochs"
    echo "  $0 --full --load   Run with all GMs using saved models"
}

# Process command-line arguments
ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --help)
            print_help
            exit 0
            ;;
        --fast)
            EPOCHS=5
            ;;
        --full)
            GM_LIST="GM001 GM002 GM003 GM004 GM005 GM006 GM007"
            ;;
        --linear-only)
            ARGS="$ARGS --only-linear"
            ;;
        --nonlinear-only)
            ARGS="$ARGS --only-nonlinear"
            ;;
        --load)
            ARGS="$ARGS --no-train"
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            ;;
        --forecast)
            FORECAST_YEARS="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
    shift
done

# Print banner
echo "=================================================="
echo "    Chess-PINNs-Forecasting MVP Analysis Tool     "
echo "=================================================="
echo ""
echo "Running with parameters:"
echo "  GMs: $GM_LIST"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Forecast years: $FORECAST_YEARS"
echo "  Additional args: $ARGS"
echo ""

# Run the program with the specified parameters
# Use a direct path approach to avoid module path issues
python3 main.py \
    --gms $GM_LIST \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --forecast-years $FORECAST_YEARS \
    --milestone-start $MILESTONE_START \
    --milestone-end $MILESTONE_END \
    --milestone-step $MILESTONE_STEP \
    $ARGS

echo ""
echo "Analysis complete! Results stored in the output directory."
