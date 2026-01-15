# Agent Specification — Financial Intelligence Governance Platform

## Overview

This system operates as a **multi-agent financial intelligence governance platform**.
Agents are strictly separated by responsibility to ensure auditability, neutrality, and reproducibility.

Agents fall into three categories:
- Analysis Agents
- Judge Agents
- Governance (Purple) Agents

No agent may assume responsibilities outside its category.

---

## 1. Analysis Agents

### Purpose
Generate financial analysis strictly from provided SEC filing evidence.

### Characteristics
- Stateless
- Deterministic
- Evidence-bound
- No policy enforcement
- No scoring authority

### Analysis Agent Types
- RiskAnalysisAgent
- ComplianceAnalysisAgent
- ValuationAnalysisAgent
- LiquidityAnalysisAgent
- EarningsQualityAgent

### Inputs
- Normalized SEC evidence artifacts
- Agent-specific analysis prompt
- Run metadata (agent version, model version)

### Outputs
- Structured analysis report
- Explicit references to evidence identifiers

### Constraints
- Must not infer beyond evidence
- Must not provide investment advice
- Must not evaluate itself or other agents

---

## 2. Judge Agents

### Purpose
Evaluate analysis outputs using predefined rubrics.

### Characteristics
- Independent from analysis agents
- Rubric-driven
- Explainable
- Deterministic

### Judge Agent Types
- DeterministicCorrectnessJudge
- ComplianceJudge
- FinancialReasoningJudge
- MaterialityJudge
- ConsistencyJudge

### Inputs
- Analysis agent output
- Rubric definitions
- Evidence references

### Outputs
- Rubric scores
- Pass / Partial / Fail judgments
- Written rationales

### Constraints
- No modification of agent output
- No cross-judge communication
- No aggregation responsibility

---

## 3. Governance (Purple) Agents

### Purpose
Oversee judge behavior and enforce governance policies.

### Characteristics
- No direct scoring authority
- Observational and enforcement role
- Fully auditable

### Governance Agent Types
- PurpleOversightAgent
- PurplePolicyEnforcementAgent

### Responsibilities
- Detect inconsistent or conflicting judge scores
- Enforce compliance gating rules
- Disqualify non-compliant runs from leaderboard
- Trigger governance alerts

### Outputs
- Governance decisions
- Audit artifacts
- Compliance flags

### Constraints
- Cannot modify judge scores
- Cannot generate financial analysis
- Decisions must be explainable

---

## Agent Interaction Rules

- Analysis Agents → Judges → Purple Agents
- No upward or circular dependencies
- All interactions are traceable and logged

---

## Compliance & Audit

All agents must:
- Produce replayable outputs
- Record inputs and versions
- Support offline audit

Failure to comply disqualifies the run from leaderboard inclusion.
