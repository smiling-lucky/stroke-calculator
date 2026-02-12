---
name: "find-skills"
description: "Helps discover and install skills from the open agent skills ecosystem. Invoke when user asks 'how do I do X' or 'find a skill for X'."
---

# Find Skills

This skill helps you discover and install skills from the open agent skills ecosystem.

## When to Use

Use this skill when the user:
- Asks "how do I do X" where X might be a common task with an existing skill
- Says "find a skill for X" or "is there a skill for X"
- Asks "can you do X" where X is a specialized capability
- Expresses interest in extending agent capabilities

## Key Commands

- `npx skills find [query]` - Search for skills interactively or by keyword
- `npx skills add <package>` - Install a skill from GitHub or other sources
- `npx skills check` - Check for skill updates
- `npx skills update` - Update all installed skills

## Example Usage

If a user asks for React performance optimization:
`npx skills find react performance`

If a user wants to install a specific skill:
`npx skills add vercel-labs/agent-skills@vercel-react-best-practices -g -y`
