#!/bin/bash
# Convenience wrapper to start services with real models (llm_service_small_model + t2vid).
# Adjust N1/N2/ports as needed before running on a GPU-capable host.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid AUTO_OPTIMIZE_ENABLED=true "$SCRIPT_DIR/start_all_services.sh" "$@"
