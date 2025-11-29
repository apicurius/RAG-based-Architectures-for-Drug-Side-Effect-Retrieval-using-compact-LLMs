---
description: Gain a general understanding of the codebase
---

# Prime

Execute the `Workflow` and `Report` sections to understand the codebase then summarize your understanding.

## Instructions

- We're building custom agents in a concise instructional way. Make sure your code is organized and easy to understand. Use extra variables with clear names to make your code easier to understand.
- IMPORTANT: Make sure all the code you write is spaced out and easy to read.
- Before you do work, make sure you move back to the root of the application.
- As you build out new apps/* for our custom agents, be sure to document them in the README.md file under the 'Custom Agent Capabilities' section following the existing format.

## Codebase Structure

- `apps/<name of specific experiment>/*`: 
  - Here we store individual custom agent examples. 
  - These represent their own micro codebases. 
  - Here we'll use 'bun' for typescript and 'uv' for python.
- `apps/custom_1_pong_agent/*` and `apps/custom_4_social_hype/*`: 
  - This is the gold standard for how to structure and build our custom agents. 
  - Take note of how we're using uv for python. 
  - SKIP TYPESCRIPT + BUN: We have a typescript version in here, but the typescript SDK has major issues, so we're not using it.
  - Focus on only python for now.
  - Think Hard about the code spacing and organization and clear step by step instructions. 
  - Organize all your code this way.
  - Notice how we're using rich logging and panels for rich output. Let's stick to this pattern for clear logging and communicating. Also use tables to showcase data.

## Workflow

- Run `git ls-files` to list all files in the repository.
- Read these files: 
  - `README.md`
  - `ai_docs/claude-code-sdk-python.md`
  - `specs/custom-agent-ideation-init.md`
  - `specs/cc_sdk_custom_tools.md`

## Report

Summarize your understanding of the codebase.