# CJCLDS Categorization (NAS)

Purpose
- Auto-tag General Conference talks from churchofjesuschrist.org with a CJCLDS category and a subcategory for the speaker (one of the 14 current Apostles) or "Other".

How it works
- Detection: URLs under `https://www.churchofjesuschrist.org/study/general-conference/...` are considered.
- Speaker extraction: parse the trailing slug from the URL (e.g., `41holland` → `holland`) and map to a canonical speaker name.
- Mapping lives in `modules/cjclds.py` (`APOSTLES_LASTNAME_TO_CANONICAL`).
- When matched:
  - `subcategories_json` merged with `{ category: "CJCLDS", subcategories: ["<Speaker>"] }`
  - `analysis_json` gets `{ speaker: "<Speaker>", speaker_role: "apostle" }` (or `"other"`)

Integration
- Applied during summary processing in `modules/services/summary_service.py` before the JSON report is saved.
- The dashboard will automatically surface CJCLDS + subcategories under filters.

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
- Unknown or non-Apostle talks are labeled `Other` under `CJCLDS`.
- Multiple speakers (rare) can be supported by extending the mapping logic.

