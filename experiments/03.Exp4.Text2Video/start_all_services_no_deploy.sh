#!/bin/bash
# Start local services with planner in passive mode (no auto deploy).
AUTO_OPTIMIZE_ENABLED=false "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_all_services.sh" "$@"
