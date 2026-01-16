#!/bin/bash

# GPU Usage Test für Ollama Modelle auf Mac
# Testet verschiedene Modelle und überwacht GPU-Nutzung

MODELS=("qwen2.5:7b" "gemma2:2b" "llama3.2:3b" "phi3:mini")
TEST_PROMPT="Explain what a neural network is in one sentence."

echo "=== Ollama GPU Usage Test für Mac M5 ==="
echo "Datum: $(date)"
echo ""

# Funktion zum Testen eines Modells
test_model() {
    local model=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Starte powermetrics im Hintergrund
    echo "Starting GPU monitoring..."
    sudo powermetrics --samplers gpu_power -i 1000 -n 30 > "/tmp/powermetrics_${model//[:\/]/_}.log" 2>&1 &
    POWERMETRICS_PID=$!
    
    sleep 2
    
    # Führe Ollama-Query aus
    echo "Running inference..."
    START_TIME=$(date +%s)
    ollama run "$model" "$TEST_PROMPT" > /dev/null 2>&1
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Warte bis powermetrics fertig ist
    sleep 3
    kill $POWERMETRICS_PID 2>/dev/null
    
    # Analysiere GPU-Nutzung
    echo ""
    echo "Results:"
    echo "  Duration: ${DURATION}s"
    
    if [ -f "/tmp/powermetrics_${model//[:\/]/_}.log" ]; then
        # Extrahiere GPU Active Residency (Prozentsatz wie oft GPU aktiv war)
        GPU_ACTIVE=$(grep -o "GPU Active residency: [0-9.]*%" "/tmp/powermetrics_${model//[:\/]/_}.log" | \
                     awk -F': ' '{print $2}' | awk -F'%' '{sum+=$1; count++} END {if(count>0) print sum/count"%"; else print "N/A"}')
        
        # Extrahiere GPU Wattage
        GPU_POWER=$(grep -o "GPU Power: [0-9]* mW" "/tmp/powermetrics_${model//[:\/]/_}.log" | \
                    awk '{sum+=$3; count++} END {if(count>0) printf "%.0f mW", sum/count; else print "N/A"}')
        
        echo "  GPU Active Residency: $GPU_ACTIVE"
        echo "  Average GPU Power: $GPU_POWER"
        
        # Bewertung
        AVG_VALUE=$(echo "$GPU_ACTIVE" | sed 's/%//')
        if [ -n "$AVG_VALUE" ] && (( $(echo "$AVG_VALUE > 5" | bc -l) )); then
            echo "  ✅ Modell nutzt GPU (Metal)"
        else
            echo "  ❌ Modell läuft vermutlich auf CPU"
        fi
    else
        echo "  ⚠️  Keine powermetrics Daten verfügbar"
    fi
    
    echo ""
}

# Prüfe ob Ollama läuft
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama Server läuft nicht. Starte..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Teste jedes Modell
for model in "${MODELS[@]}"; do
    # Prüfe ob Modell existiert
    if ollama list | grep -q "$model"; then
        test_model "$model"
    else
        echo "⚠️  Model $model nicht gefunden - überspringe"
        echo ""
    fi
done

# Cleanup
rm -f /tmp/powermetrics_*.log

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test abgeschlossen!"
echo ""
echo "Empfehlungen:"
echo "  - Modelle mit GPU Residency > 5% nutzen Metal effektiv"
echo "  - Bei CPU-Fallback: Modell neu herunterladen oder Ollama neu starten"
echo "  - Für beste Performance: Wähle Modelle mit höchster GPU-Auslastung"
