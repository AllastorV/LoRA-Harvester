# Release Process for LoRA-Harvester

## Creating a New Release Tag

To create and push a new version tag (e.g., LoRA-HarvesterV2), follow these steps:

### Steps:

1. **Checkout main branch**
   ```bash
   git checkout main
   ```

2. **Pull latest changes**
   ```bash
   git pull
   ```

3. **Create the version tag**
   ```bash
   git tag LoRA-HarvesterV2
   ```

4. **Push the tag to origin**
   ```bash
   git push origin LoRA-HarvesterV2
   ```

### Important Notes:

- The tag name should follow the format: `LoRA-HarvesterV[version]`
- Make sure to push the tag with a **space** between `origin` and the tag name
- ❌ **INCORRECT**: `git push originLoRA-HarvesterV2` (missing space)
- ✅ **CORRECT**: `git push origin LoRA-HarvesterV2`

### Current Version

- **v2.0** - LoRA-HarvesterV2 (Latest)
  - Added BLIP + WD14 Captioning
  - Advanced Tag Settings
  - Quality Analysis
  - Async I/O
  - Checkpoint/Resume functionality

## Viewing Tags

To list all tags:
```bash
git tag -l
```

To view tag details:
```bash
git show LoRA-HarvesterV2
```

## Deleting a Tag (if needed)

Local:
```bash
git tag -d LoRA-HarvesterV2
```

Remote:
```bash
git push origin --delete LoRA-HarvesterV2
```
