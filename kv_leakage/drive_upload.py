"""
drive_upload.py — Google Drive uploader for experiment results.

Setup (one-time):
  1. Enable Google Drive API at console.cloud.google.com
  2. Create OAuth 2.0 credentials (Desktop App) and download as credentials.json
  3. Place credentials.json next to this file (or set GDRIVE_CREDENTIALS env var)
  4. First run opens a browser for OAuth consent — token.json is saved automatically

Usage:
  from kv_leakage.drive_upload import upload_results
  upload_results([path1, path2, ...], folder_id="<optional Drive folder ID>")
"""

import os
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    pass

# ── Credential paths ──────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
CREDENTIALS_FILE = Path(os.environ.get("GDRIVE_CREDENTIALS", _HERE / "credentials.json"))
TOKEN_FILE        = Path(os.environ.get("GDRIVE_TOKEN",       _HERE / "token.json"))

SCOPES = ["https://www.googleapis.com/auth/drive"]

# MIME type map
_MIME = {
    ".csv":  "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".json": "application/json",
    ".txt":  "text/plain",
    ".png":  "image/png",
}


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_service():
    """Authenticate and return a Drive v3 service object."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google client libraries not installed.\n"
            "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    creds = None

    if TOKEN_FILE.exists():
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_FILE}\n"
                    "Download it from console.cloud.google.com → "
                    "APIs & Services → Credentials → OAuth 2.0 Client IDs"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"[AUTH] Token saved → {TOKEN_FILE}")

    return build("drive", "v3", credentials=creds)


# ── Upload a single file ──────────────────────────────────────────────────────

def upload_file(local_path: str | Path, folder_id: str = None) -> str | None:
    """
    Upload one file to Google Drive.

    Args:
        local_path:  Path to the local file.
        folder_id:   Google Drive folder ID (from the URL).
                     None → uploads to the root of My Drive.
    Returns:
        Drive file ID string, or None on failure.
    """
    from googleapiclient.http import MediaFileUpload

    path = Path(local_path)
    if not path.exists():
        print(f"[SKIP] File not found: {path}")
        return None

    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    metadata = {"name": path.name}
    if folder_id:
        metadata["parents"] = [folder_id]

    try:
        service = _get_service()
        media   = MediaFileUpload(str(path), mimetype=mime, resumable=True)
        result  = service.files().create(
            body=metadata, media_body=media, fields="id,name"
        ).execute()
        print(f"[DRIVE] ✓ {result['name']}  (id={result['id']})")
        return result["id"]
    except Exception as e:
        print(f"[DRIVE] ✗ Failed to upload {path.name}: {e}")
        return None


# ── Upload a list of files ────────────────────────────────────────────────────

def upload_results(paths: list, folder_id: str = None) -> dict:
    """
    Upload multiple result files to Google Drive.

    Args:
        paths:      List of file paths (str or Path).
        folder_id:  Drive folder ID, or None for root.
    Returns:
        Dict mapping filename → Drive file ID (None if upload failed).
    """
    if not paths:
        print("[DRIVE] No files to upload.")
        return {}

    print(f"\n{'='*60}")
    print(f"UPLOADING {len(paths)} FILE(S) TO GOOGLE DRIVE")
    if folder_id:
        print(f"Folder ID : {folder_id}")
    print(f"{'='*60}")

    results = {}
    for p in paths:
        file_id = upload_file(p, folder_id=folder_id)
        results[Path(p).name] = file_id

    succeeded = sum(1 for v in results.values() if v)
    print(f"\n[DRIVE] Upload complete: {succeeded}/{len(paths)} succeeded")
    return results


# ── Convenience: upload an entire directory ───────────────────────────────────

def upload_directory(directory: str | Path, pattern: str = "*",
                     folder_id: str = None) -> dict:
    """
    Upload all files matching `pattern` in `directory`.

    Example:
        upload_directory(config.WITHIN_DIR, "*.csv", folder_id="...")
    """
    files = sorted(Path(directory).glob(pattern))
    return upload_results(files, folder_id=folder_id)
