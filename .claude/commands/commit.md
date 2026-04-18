# Smart Commit & Push

Commit only the currently staged files, after running pre-commit and checking for secrets.

## 1. Verify there are staged files

```bash
git diff --cached --name-only
```

If the output is empty, stop and tell the user there is nothing staged.

## 2. Sensitive file check

Inspect every staged file. Refuse to proceed and warn the user if any of the following are staged:

- `.env`, `.env.*`, `*.env`
- Files containing patterns like `GEMINI_API_KEY`, `GCP_CREDENTIALS`, private keys (`-----BEGIN`), tokens, passwords or secrets hardcoded as string literals
- `db_data/` or any `.sqlite` file
- `*.pem`, `*.key`, `*.p12`, `*.pfx`

If a sensitive file is found, list it explicitly and **do not commit**. Ask the user to unstage it (`git restore --staged <file>`) before continuing.

## 3. Run pre-commit on staged files only

```bash
uv run pre-commit run --files $(git diff --cached --name-only | tr '\n' ' ')
```

If pre-commit makes automatic fixes (exit code non-zero but files were modified):

- Show which files were auto-fixed.
- Re-stage the fixed files with `git add <fixed-files>`.
- Re-run `git diff --cached --name-only` to confirm the corrected content is staged.
- Tell the user what was fixed.

If pre-commit fails with errors that it could **not** fix automatically (e.g. lint errors requiring manual intervention), show the output and **do not commit**. Ask the user to fix the issues and re-run `/commit`.

## 4. Create the commit

- Run `git log -5 --oneline` to match the existing commit message style.
- Draft a concise commit message (imperative mood, same language as existing commits) that explains *why* the change was made, not just what.
- Commit only the staged files (do **not** run `git add -A` or `git add .`):

```bash
git commit -m "$(cat <<'EOF'
<your commit message here>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

## 5. Push

Push the current branch to its upstream:

```bash
git push
```

If no upstream is set, push and set it:

```bash
git push -u origin $(git branch --show-current)
```

Report the result — commit SHA and push status — to the user.
