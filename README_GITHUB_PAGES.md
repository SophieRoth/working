# Publish Interactive Plot to Notion via GitHub Pages

## 1) Generate files
From this folder:

```bash
python3 work_hours_dashboard.py
```

This writes:
- `work_hours_2025_vacation_named_fixed.html` (local copy)
- `docs/index.html` (GitHub Pages copy)

## 2) Push to GitHub

```bash
git add work_hours_dashboard.py docs/index.html README_GITHUB_PAGES.md
git commit -m "Add GitHub Pages output for work-hours dashboard"
git push
```

## 3) Enable GitHub Pages
In your GitHub repo:
1. Open `Settings` -> `Pages`
2. Under `Build and deployment`, set:
   - `Source`: `Deploy from a branch`
   - `Branch`: `main` (or your default branch)
   - `Folder`: `/docs`
3. Save

Your public page URL will be:

`https://<github-username>.github.io/<repo-name>/`

## 4) Embed in Notion
In Notion:
1. Type `/embed`
2. Paste the GitHub Pages URL above
3. Resize the embed block as needed

## Notes
- After each data/script update, rerun `python3 work_hours_dashboard.py` and push `docs/index.html`.
- If Notion still shows an old version, hard refresh the page (or wait ~1-2 minutes for cache to refresh).
