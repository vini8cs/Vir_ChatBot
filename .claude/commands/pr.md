# Create Pull Request

Before creating the PR, run these checks in order and **abort if any fails**.

## 1. Sensitive file check

Look at every file that will be included in the PR (changed vs main branch).
Refuse to proceed and warn the user if any of the following are staged or changed:
- `.env`, `.env.*`, `*.env`
- Files containing patterns like `GEMINI_API_KEY`, `GCP_CREDENTIALS`, private keys (`-----BEGIN`), tokens, passwords or secrets hardcoded as string literals
- `db_data/` or any `.sqlite` file
- `*.pem`, `*.key`, `*.p12`, `*.pfx`

If a sensitive file is found, list it explicitly and ask the user to remove it before continuing.

## 2. Run the test suite

```bash
uv run pytest -q
```

If any test fails, show the failure output and **do not create the PR**. Ask the user to fix the failures first.

## 3. Create the PR

Only if both checks above pass:

- Gather `git log main..HEAD`, `git diff main...HEAD --stat`, and the full diff of non-test files to understand the changes.
- Push the branch if not already pushed (`git push -u origin <branch>`).
- Draft a PR with:
  - **Title**: short (≤70 chars), imperative mood, in the same language the commit messages are written in.
  - **Body** in this exact structure:

```
## Summary
<2-5 bullet points describing what changed and why>

## Test plan
<bulleted checklist of what to verify manually>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

- Use `gh pr create` with the drafted title and body.
- Return the PR URL to the user.
