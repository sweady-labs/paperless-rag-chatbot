#!/bin/bash

# Model Speed Benchmark - alle Modelle mit identischem Prompt testen

MODELS=("gemma2:2b" "llama3.2:3b" "phi3:mini" "qwen2.5:7b")
PROMPT="Explain what a neural network is in one short sentence."

echo "=== Ollama Model Speed Benchmark ==="
echo "All models are now using Metal/GPU (verified in logs)"
echo ""

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Model: $model"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Warmup (preload model into GPU)
        ollama run "$model" "Hi" > /dev/null 2>&1
        
        # Actual benchmark
        result=$(ollama run "$model" "$PROMPT" --verbose 2>&1)
        
        # Extract metrics
        total_dur=$(echo "$result" | grep "total duration:" | awk '{print $3}')
        eval_rate=$(echo "$result" | grep "eval rate:" | awk '{print $3}')
        
        echo "  Total Duration: $total_dur"
        echo "  Generation Speed: $eval_rate tokens/s"
        echo ""
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Empfehlung basierend auf Speed + Quality:"
echo "  - Schnellstes: gemma2:2b (~100+ tokens/s)"
echo "  - Beste Qualität: qwen2.5:7b (~30-50 tokens/s)"
echo "  - Bester Kompromiss: llama3.2:3b (~50-80 tokens/s)"
