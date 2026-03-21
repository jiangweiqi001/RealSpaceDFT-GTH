# AGENTS.md

## Project goal
This repo is a real-space DFT package.
Keep the existing uniform-grid implementation intact.
Add new functionality in an adaptive-grid backend later.

## Working rules
- Never delete the current uniform-grid path.
- Prefer additive refactors over destructive rewrites.
- Before coding, identify which files implement grid, Laplacian, Poisson, SCF, and projector overlap.
- For large tasks, plan first, then implement in small steps.
- Run the smallest available verification script after each change.
- Report exactly what was changed, what was tested, and what remains unverified.

## Review rules
- Keep diffs minimal and reviewable.
- Do not claim adaptive-grid support is complete unless grid, Laplacian, weighted integration, Poisson, and SCF all work.