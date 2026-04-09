---
name: ml-rag-project-workflow
description: 'Design and execute an end-to-end SOTA ML project for SLM fine-tuning for RAG in a domain corpus. Use for planning, implementing, and validating data pipeline, QLoRA training, retrieval, evaluation, robustness analysis, and report-ready outputs.'
argument-hint: 'Task goal, dataset location, runtime environment (local/Colab/Kaggle), and expected deliverables'
---

# ML RAG Project Workflow

## When to Use
- Build or refine an academic ML project around SLM fine-tuning for RAG.
- Adapt project workflows across environments (local GPU, Colab, Kaggle).
- Enforce evaluation and robustness requirements before reporting results.
- Prepare publishable-quality artifacts for paper, poster, or presentation.

## Inputs to Collect First
- Objective: algorithmic novelty, application target, or hybrid scope.
- Domain and data: source files, language, legal/privacy constraints.
- Runtime constraints: GPU, VRAM, storage, execution time limits.
- Existing code state: available modules, missing modules, current outputs.
- Required outputs: trained adapter/model, metrics, plots, reproducible commands.

## Step-by-Step Procedure
1. Define problem framing and hypotheses.
2. Specify baseline and proposed variant.
3. Validate data availability and preprocessing assumptions.
4. Build retrieval assets (chunks, embeddings, vector index).
5. Prepare training set (instruction format, split strategy, quality checks).
6. Configure fine-tuning (QLoRA, quantization, batch strategy by VRAM).
7. Train with logging and checkpoint strategy.
8. Evaluate base vs fine-tuned with fixed protocol.
9. Run robustness checks and failure analysis.
10. Produce reproducible artifacts and communication-ready summaries.

## Decision Points and Branching Logic
- If preprocessing artifacts already exist:
  Reuse cached outputs and skip heavy extraction/generation stages.
- If data modules are missing in the codebase:
  Stop preprocessing attempts, flag the missing dependency, and continue from available artifacts.
- If VRAM is limited:
  Reduce per-device batch size, increase gradient accumulation, keep 4-bit quantization enabled.
- If RAGAS execution fails:
  Fall back to deterministic proxy metrics and clearly label them as fallback metrics.
- If evaluation data is too small or imbalanced:
  Expand/stratify samples before drawing conclusions.

## Quality Gates (Completion Checks)
- Data checks:
  Source files detected, chunk count plausible, QA dataset exists, train/validation splits valid.
- Training checks:
  Non-empty trainable parameter set, no OOM crashes, metrics/logs written to output directory.
- Retrieval checks:
  FAISS index exists and similarity search returns expected domain chunks.
- Evaluation checks:
  Base and fine-tuned runs both completed with aggregate metrics saved.
- Comparison checks:
  Side-by-side metric summary and plots generated.
- Reproducibility checks:
  Single-command pipeline path documented, environment dependencies pinned.

## Robustness and Originality Review
- Verify that choices are motivated both theoretically and empirically.
- Document at least one non-trivial design decision beyond a template baseline.
- Include failure cases, error patterns, and limits of generalization.
- Compare alternatives considered and explain rejected options.

## Deliverables Checklist
- Executable pipeline commands for each stage.
- Saved artifacts in outputs directory (model, index, evaluation, plots).
- Concise experiment log: configuration, runtime, notable issues, fixes.
- Final summary with technical quality, originality, communication clarity, and robustness discussion.

## Prompt Examples
- "Apply ml-rag-project-workflow to adapt this notebook from Colab to Kaggle with dataset path /kaggle/input/..."
- "Use ml-rag-project-workflow to run a full base vs fine-tuned evaluation and generate plots."
- "Use ml-rag-project-workflow to diagnose weak retrieval quality and propose chunking/retrieval changes."
