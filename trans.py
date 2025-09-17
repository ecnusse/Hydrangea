import os
import re
import pandas as pd
import yaml

# Input Excel file
INPUT_FILE = "defect.xlsx"
SHEET_NAME = "defect"
OUTPUT_DIR = "db"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read Excel, note that header=1 is the actual header row
try:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=1)
except Exception as e:
    print(f"Error: Failed to read Excel file {INPUT_FILE}: {e}")
    exit(1)

last_type_display = ""

for _, row in df.iterrows():
    app_full = str(row.get("APP", "")).strip()  # e.g. cpacker/MemGPT
    repo_url_raw = str(row.get("commit url", "")).strip()
    # Read and forward-fill type: if empty, reuse the last non-empty type
    raw_type_value = row.get("types")
    if pd.notna(raw_type_value) and str(raw_type_value).strip():
        defect_type_display = str(raw_type_value).strip()
        last_type_display = defect_type_display
    else:
        defect_type_display = last_type_display
    case = str(row.get("cases", "")).strip()

    if not app_full or not repo_url_raw:
        continue  # Skip invalid rows

    # Extract app name (remove author prefix)
    app_name = app_full.split("/")[-1]

    # Process defect_type (for defect_id/filename: lowercase+underscore; display field keeps original/forward-fill)
    # Normalize: compress any consecutive whitespace to single underscore, deduplicate underscores, trim ends
    if defect_type_display:
        tmp = defect_type_display.strip().lower()
        tmp = re.sub(r"\s+", "_", tmp)
        tmp = re.sub(r"_+", "_", tmp).strip("_")
        defect_type_norm = tmp if tmp else "unknown"
    else:
        defect_type_norm = "unknown"

    # Normalize case: prioritize digitization, use "/" if empty
    case_value = row.get("cases")
    if pd.notna(case_value):
        if isinstance(case_value, (int, float)):
            try:
                case_int_like = int(case_value)
                case = str(case_int_like)
            except Exception:
                case = str(case_value).strip() or "/"
        else:
            raw_case = str(case_value).strip()
            # Support formats like "case1" / "Case 1" / "CASE-1"
            m = re.match(r"(?i)^case\s*[-_]?\s*(\d+)$", raw_case)
            if m:
                case = m.group(1)
            else:
                case = raw_case or "/"
    else:
        case = "/"

    # Construct defect_id (including author name)
    author_name = app_full.split("/")[0] if "/" in app_full else app_name
    defect_id = (
        f"{author_name}-{app_name}-{defect_type_norm}-case{case}"
        if case != "/"
        else f"{author_name}-{app_name}-{defect_type_norm}-/"
    )

    # Construct output filename: include author/project, omit case segment when case is "/"
    app_full_for_name = app_full  # e.g. imartinez/privateGPT
    case_segment = f"-case{case}" if case != "/" else ""
    file_stem = f"{app_full_for_name}-{defect_type_norm}{case_segment}"
    # Windows compatibility: replace "/" with "-", use hyphens consistently
    sanitized_stem = file_stem.replace("/", "-")
    file_name = sanitized_stem + ".yaml"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    # If file with same name exists, append numeric suffix to avoid overwriting
    if os.path.exists(file_path):
        suffix = 2
        while True:
            candidate_name = f"{sanitized_stem}-{suffix}.yaml"
            candidate_path = os.path.join(OUTPUT_DIR, candidate_name)
            if not os.path.exists(candidate_path):
                file_name = candidate_name
                file_path = candidate_path
                break
            suffix += 1

    # Split repo and commit from commit url (format: .../tree/<commit>)
    if "/tree/" in repo_url_raw:
        parts = repo_url_raw.split("/tree/")
        repo = parts[0]
        commit = parts[1].split("/")[0]
    else:
        repo = repo_url_raw
        commit = ""

    # consequence / locations / trigger_tests
    consequences = (
        [c.strip() for c in str(row.get("consequences", "")).split(",") if c.strip()]
        if pd.notna(row.get("consequences"))
        else []
    )
    locations = []
    if pd.notna(row.get("source-code locations")):
        for loc in str(row.get("source-code locations", "")).split("\n"):
            loc_clean = loc.strip()
            if loc_clean:
                locations.append(loc_clean)
    trigger_tests = (
        [str(row.get("defect-triggering tests", "")).strip()]
        if pd.notna(row.get("defect-triggering tests"))
        else []
    )

    # YAML content
    data = {
        "app": app_name,
        "repo": repo,
        "commit": commit,
        "defect_id": defect_id,
        "type": defect_type_display if defect_type_display else "unknown",
        "case": case,
        "consequence": consequences,
        "locations": locations,
        "trigger_tests": trigger_tests,
    }

    # Write YAML
    with open(file_path, "w", encoding="utf-8") as out:
        yaml.dump(data, out, allow_unicode=True, sort_keys=False)

    print(f"Wrote {file_path}")
