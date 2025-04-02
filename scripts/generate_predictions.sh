#!/bin/bash

# Editable parameters
GENERATION_MODEL="LenguajeNaturalAI/supernova-fns"
RETRIEVAL_TECHNIQUES_MODEL="LenguajeNaturalAI/leniachat-qwen2-1.5B-v0"
GENERATION_MODEL_BASE_URL="http://localhost:8000"
RETRIEVAL_TECHNIQUES_MODEL_BASE_URL="http://localhost:8001"

# Do not edit
API_COMMAND="uv run python src/llm_api.py"
STOP_COMMAND="uv run pkill -f src/llm_api.py"



wait_for_api() {
    local API_CHECK_URL="$1"
    echo "Waiting for API at $API_CHECK_URL to be ready (Max 15 min)..."
    SECONDS_WAITED=0
    MAX_WAIT=900  

    while [[ $SECONDS_WAITED -lt $MAX_WAIT ]]; do
        RESPONSE=$(curl --silent --fail "$API_CHECK_URL")
        if [[ "$RESPONSE" == *"true"* ]]; then
            echo "API at $API_CHECK_URL is ready!"
            return 0
        fi
        sleep 10
        ((SECONDS_WAITED++))
    done

    echo "ERROR: API at $API_CHECK_URL did not start within 15 minutes."
    exit 1
}

# Start generation model API
echo "Starting generation model API..."
$API_COMMAND --model_name="$GENERATION_MODEL" --port=8000 --gpu_memory_utilization=0.6 &
wait_for_api "$GENERATION_MODEL_BASE_URL/api_active/"

# Start retrieval techniques model API
echo "Starting retrieval techniques model API..."
$API_COMMAND --model_name="$RETRIEVAL_TECHNIQUES_MODEL" --port=8001 --gpu_memory_utilization=0.2 &
wait_for_api "$RETRIEVAL_TECHNIQUES_MODEL_BASE_URL/api_active/"

uv run python src/generate_responses.py 


# Ensure both APIs are stopped properly
echo "Stopping both APIs..."
$STOP_COMMAND 
wait
echo "Both APIs stopped successfully."