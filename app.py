from io import StringIO

# Synonyms are already defined above:
# DATE_SYNS, ITEM_SYNS, QTY_SYNS

def _read_csv_any(file):
    """
    Read a ThriveMetrics CSV that may contain a report preamble above the table.
    - Detects delimiter (comma/semicolon/tab/pipe)
    - Scans first ~200 lines to find the *true* header row that contains at least
      two of the three groups: Date / Item / Quantity (by synonyms)
    - Reads with pandas starting at that header row
    """
    # Load entire file once (UploadedFile / file-like supported)
    try:
        file.seek(0)
        raw_bytes = file.read()
    except Exception:
        # If it's a raw path-like, read directly
        with open(file, "rb") as f:
            raw_bytes = f.read()

    text = raw_bytes.decode("utf-8-sig", errors="ignore")
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]

    # Helper to score a header candidate
    def score_tokens(tokens):
        cols = [t.strip().lower() for t in tokens if t.strip() != ""]
        has_date = any(any(s in c for s in DATE_SYNS) for c in cols)
        has_item = any(any(s in c for s in ITEM_SYNS) for c in cols)
        has_qty  = any(any(s in c for s in QTY_SYNS)  for c in cols)
        score = int(has_date) + int(has_item) + int(has_qty)
        return score, len(cols)

    candidates_delims = [",", ";", "\t", "|"]

    best_header_idx = None
    best_delim = ","
    best_score = (-1, -1)  # (score, num_cols)

    # Scan first N lines for a likely table header
    scan_limit = min(200, len(lines))
    for i in range(scan_limit):
        line = lines[i]
        for d in candidates_delims:
            toks = line.split(d)
            s = score_tokens(toks)
            # pick the row that hits max score; tiebreaker: more columns
            if (s[0] > best_score[0]) or (s[0] == best_score[0] and s[1] > best_score[1]):
                best_score = s
                best_header_idx = i
                best_delim = d

    # Fallback if nothing looks like a header (use first line & try sniffer)
    if best_header_idx is None or best_score[0] < 2:
        import csv
        try:
            best_delim = csv.Sniffer().sniff(text[:2048]).delimiter
        except Exception:
            best_delim = ","
        best_header_idx = 0

    # Read with pandas from the detected header row
    sio = StringIO(text)
    df = pd.read_csv(
        sio,
        delimiter=best_delim,
        header=best_header_idx,
        engine="python",
        on_bad_lines="skip",
    )

    # Normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df
