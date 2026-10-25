# Streamlit redeployment handoff for Jordan

Both entry points move in this reorganization. Jordan (`mushroomyyy`) owns the
Streamlit apps and will create replacements after the team approves the local
changes and they are merged into remote `main`. Do not deploy the local
`codex/repo-reorg` branch or push that branch. Coordinate the merge with Jordan:
the old apps lose their entry points once `main` changes.

## Delivery Watch

1. In Streamlit Community Cloud, create a new app from
   `Team8_IT5006_Ecommerce_Analytics_AY2627Sem1`, branch `main`.
2. Set main file to `deployment/app/app.py`. Match the current app's Python
   version (the repository's tested runtime is Python 3.13.5). Dependencies are
   in the adjacent `deployment/app/requirements.txt`.
3. Choose a readable available subdomain, such as `team8-delivery-watch`.
4. In Advanced settings / Secrets, set:

   ```toml
   DELIVERY_API_URL = "https://team8-delivery-api.onrender.com"
   ```

5. Deploy after the existing Render API has successfully redeployed and passed
   HTTP verification. A sleeping free API may take time to wake.
6. Verify all seven pages. Check these specific values and interactions:

   | Check | Expected |
   |---|---|
   | Start here | Loads without errors |
   | Daily simulation, 2 July 2018, defaults | 129 scored orders, 39 selected |
   | Daily simulation | Previous/next date, download, reveal/reset outcomes work |
   | Month-end review | July results appear |
   | Explore the data, default filters | 98,958 approved orders; filters/charts work |
   | Single-order API demo | Produces classification and regression results |
   | Batch API demo | Upload `docs/examples/batch_example.csv`; 10/10 rows succeed |
   | Models and guidance | Both model versions and guidance render |

7. Record the new URL and delete the old Delivery Watch app after the replacement
   passes these checks. No compatibility entry point remains at the old path.

## Phase 1 dashboard

1. Create another app from the same repository and branch `main`.
2. Set main file to `deployment/phase1_dashboard/streamlit_app.py`.
   Dependencies are in `deployment/phase1_dashboard/requirements.txt`; use the
   existing Phase 1 app's Python version. No secrets are needed.
3. Choose an available subdomain and deploy. The wrapper sets
   `OLIST_HOSTED_RELEASE=1` and extracts repository-root `data/olist_csv.zip`
   into a temporary cache. First load can take a minute.
4. Check the dashboard loads, sidebar filters respond and charts render. If
   memory kills the app, record the deployment logs: its previous local peak
   was about 1.3 GB because it loads the full raw tables.
5. Record the new URL and delete the old Phase 1 app after the replacement passes.

## Return information and update links

Send Andrew the two new URLs, selected Python versions, deployed commit and any
failed checks. Update all active repository links in one follow-up commit:
`README.md`, `docs/APPLICATION.md`, `docs/DEPLOYMENT.md`, `docs/VERIFICATION.md`
and any remaining old `streamlit.app` link found by a repository search. Use
`[skip render]` for that link-only commit. Historical evidence files should remain
explicitly labelled as historical if they retain old URLs.

The team also updates the slides/PDF, final report and previously shared links
outside the repository. Retire the old URLs only after replacements are verified.
