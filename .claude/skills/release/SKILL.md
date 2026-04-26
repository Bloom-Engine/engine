---
name: release
description: Cut a new Bloom Engine release — bump package.json if requested, commit, tag, push, create GitHub Release, and watch the gated release pipeline
disable-model-invocation: true
argument-hint: [optional: "patch" | "minor" | "major" to bump the version first, or free-text release highlights]
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# New Bloom Release

## Model (important — read before acting)

Bloom is a library, not a CLI. It's consumed by games as `"bloom": "file:../../"` or (eventually) via npm. The release event therefore is:

1. The version in `package.json` is the version to ship.
2. Tag HEAD (or the commit that bumped the version) with `vX.Y.Z`.
3. Push the tag. This fires `.github/workflows/release.yml`, which gates on the `Tests` workflow passing for the same SHA and then creates the GitHub Release.
4. Watch both workflows. Report success or failure.

Unlike Perry, Bloom does **not** bump the patch version on every commit. The version in `package.json` is bumped explicitly at release time. `/release` can do the bump for you if you pass `patch` / `minor` / `major`, or you can bump it in a regular commit first and just run `/release` to tag HEAD.

## Steps

### 1. Sanity checks

- `git status` — must be clean (unless we're about to do a `patch`/`minor`/`major` bump, in which case a dirty `package.json` alone is acceptable only if *you* are the one about to edit it).
- `git rev-parse --abbrev-ref HEAD` — must be `main`. If not, STOP and ask.
- `git fetch origin && git log HEAD..origin/main --oneline` — must be empty. If origin is ahead, pull first.

### 2. Determine the target version

- Read `package.json` `version` field.
- If `$ARGUMENTS` is `patch` / `minor` / `major`:
  - Compute the new version.
  - Edit `package.json`.
  - `git add package.json && git commit -m "chore(release): vX.Y.Z"`.
- Otherwise the current `package.json` version is the release version. Do **not** bump.

### 3. Verify the tag doesn't exist

```bash
git rev-parse "vX.Y.Z" 2>/dev/null && echo "tag exists — aborting" && exit 1
git ls-remote --tags origin "vX.Y.Z" | grep -q "vX.Y.Z" && echo "tag exists on origin — aborting" && exit 1
```

If the tag already exists locally or on origin, STOP. Either the release already shipped or someone started it and didn't finish — don't silently duplicate. Retagging a published version is a cardinal sin.

### 4. Survey commits since the last tag

```bash
last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [ -n "$last_tag" ]; then
  git log "$last_tag"..HEAD --oneline
else
  git log --oneline -20   # first release — no previous tag
fi
```

Group the subjects mentally by `feat:` / `fix:` / `perf:` / `docs:` / `chore:`. These become the "Highlights / Features / Fixes" sections in the release body.

### 5. Push the bump commit (if any) + tag

```bash
# If step 2 created a bump commit:
git push origin main

git tag "vX.Y.Z"
git push origin "vX.Y.Z"
```

The tag push fires two workflows in parallel:

- `test.yml` — full CI matrix (macOS / Linux / Windows builds, shared-crate tests, WASM build)
- `release.yml` — its `await-tests` job polls for the test.yml run on the same SHA and gates the release on it

### 6. Create the GitHub Release

```bash
gh release create "vX.Y.Z" \
  --title "vX.Y.Z" \
  --notes "$(cat <<'EOF'
## Highlights
- ...

## Features
- ...

## Fixes
- ...

## Infrastructure
- ...
EOF
)"
```

If `$ARGUMENTS` contained free-text highlights (anything other than the `patch`/`minor`/`major` keywords), seed the "Highlights" section from it.

If you skip the `gh release create` step, `release.yml`'s `github-release` job will auto-create a release with `--generate-notes` once the gate passes. The explicit-body path is preferred when there's narrative worth writing; the auto-notes fallback keeps the release from just not existing if something goes sideways.

### 7. Watch the pipeline

```bash
gh run watch $(gh run list --workflow="Release" --limit 1 --json databaseId --jq '.[0].databaseId')
```

Expected timeline:
- `Tests` finishes in ~15-25 min (Linux + Windows Jolt cmake build dominates cold runs)
- `Release` `await-tests` unblocks, then `github-release` creates the release if needed

If `Tests` failed, `await-tests` will fail loudly and the release will not be created. Do **not** re-tag `vX.Y.Z` — fix on main, bump to `vX.Y.(Z+1)`, and `/release` that. Retagging a published tag breaks npm/git consumers that already resolved the old SHA.

### 8. Report back

- GitHub release URL
- Whether `Tests` + `Release` both went green
- The new `package.json` version + tag

## Failure modes

- **Dirty worktree (unrelated files)**: STOP. Commit or stash first — don't sweep random changes into a release commit.
- **Tag already exists**: STOP. Investigate before deciding whether to bump further or resume a partial release.
- **Tests failed on the tag SHA**: `await-tests` blocks release creation. Fix on main, bump patch, `/release` again. The stale tag is harmless noise — no GH release body, no assets.
- **Not on main / origin ahead**: STOP. Releases come off main. Pull and reconcile before tagging.

## What NOT to do

- Do not bump the version *and* add unrelated changes in the same commit. Keep the bump commit mechanical: `chore(release): vX.Y.Z`, one file.
- Do not force-push tags.
- Do not use `git add -A` anywhere in this skill — scoped `git add package.json` only.
- Do not try to amend a release commit after the tag is pushed. If the body is wrong, `gh release edit vX.Y.Z --notes "..."`.
- Do not skip the test gate. Manual `workflow_dispatch` of release.yml exists for emergencies only.
