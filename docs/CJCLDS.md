# CJCLDS Categorization (NAS)

Purpose
- Auto-tag General Conference talks from churchofjesuschrist.org with a CJCLDS category and a subcategory for the speaker (one of the 14 current Apostles) or "Other".

How it works
- Detection:
  - All `churchofjesuschrist.org` links are considered CJCLDS.
  - General Conference talks: URLs under `/study/general-conference/...`.
  - Non‑conference church content: any other path on the same domain.
- Speaker / subcategory assignment:
  - General Conference → parse trailing slug (e.g., `41holland` → `holland`) and map to canonical Apostle.
    - If unknown/non‑Apostle → subcategory `Other`.
  - Non‑conference → subcategory `Non GC`.
- Mapping lives in `modules/cjclds.py` (`APOSTLES_LASTNAME_TO_CANONICAL`).
- When matched:
  - `subcategories_json` merged with `{ category: "CJCLDS", subcategories: ["<Speaker>"] }`
  - `analysis_json` gets `{ speaker: "<Speaker>", speaker_role: "apostle" }` (or `"other"`)

Integration
- Applied during summary processing in `modules/services/summary_service.py` before the JSON report is saved.
- The dashboard will automatically surface CJCLDS + subcategories under filters.

Backfill existing content

- Run a quick backfill from NAS:
  - Dry run: `python3 tools/backfill_cjclds.py --dry-run`
  - Apply (limit first N items): `python3 tools/backfill_cjclds.py --limit 50`
  - Restrict by host: `python3 tools/backfill_cjclds.py --only-host churchofjesuschrist.org`

This updates JSON under `/app/data/reports` and upserts to the dashboard using the NAS dual-sync path.

Current Apostles (canonical subcategories)
- Dallin H. Oaks
- Henry B. Eyring
- Jeffrey R. Holland
- Dieter F. Uchtdorf
- David A. Bednar
- Quentin L. Cook
- D. Todd Christofferson
- Neil L. Andersen
- Ronald A. Rasband
- Gary E. Stevenson
- Dale G. Renlund
- Gerrit W. Gong
- Ulisses Soares
- Patrick Kearon

Notes
- Unknown or non‑Apostle General Conference talks are labeled `Other` under `CJCLDS`.
- Non‑conference church content is labeled `Non GC` under `CJCLDS`.
- Multiple speakers (rare) can be supported by extending the mapping logic.
