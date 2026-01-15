# Leaderboard Specification

## Purpose
Rank agent performance based on benchmark results.

## Ranking Entities
- Agent
- Agent version
- Prompt version
- Model version
- Pipeline run

## Rules
- Rankings derived only from persisted artifacts
- Compliance is a hard gate
- Deterministic aggregation only

## Outputs
- leaderboard.json
- leaderboard.md
