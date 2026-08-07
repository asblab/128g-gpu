#!/bin/bash
# consistent bench: ~600-tok prompt, 128 gen tokens; prints pp/gen tok/s
P=$(python3 -c "print(('The quick brown fox jumps over the lazy dog near the riverbank while autumn leaves drift across the meadow. '*60).strip())")
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Summarize in one sentence: $P\"}],\"max_tokens\":128}" \
| python3 -c "import json,sys; t=json.load(sys.stdin)[\"timings\"]; print(f\"pp {t[\"prompt_per_second\"]:.1f} tok/s ({t[\"prompt_n\"]} tok) | gen {t[\"predicted_per_second\"]:.1f} tok/s ({t[\"predicted_n\"]} tok)\")"
