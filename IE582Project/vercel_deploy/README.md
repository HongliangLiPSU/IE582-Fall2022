# Vercel Static Deploy Folder

This directory is intentionally static-only for Vercel deployment.

## Files

- `index.html`: Published map page (copied from `outputs/maps/county_index_map.html`)
- `vercel.json`: Static deployment config

## Update and deploy

From project root:

```bash
.venv/bin/python build_food_insecurity_map.py
./sync_vercel_deploy.sh
vercel link --cwd vercel_deploy
vercel --cwd vercel_deploy --prod
```

If asked during first deployment, keep defaults and do not enable Python settings.
