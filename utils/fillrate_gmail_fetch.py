# utils/fillrate_gmail_fetch.py
from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import re
import sys
from datetime import datetime
from email.header import decode_header
from email.message import Message
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993


def load_env_file(env_path: str = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def decode_mime_words(value: Optional[str]) -> str:
    if not value:
        return ""

    parts = decode_header(value)
    decoded = []

    for part, encoding in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded.append(part)

    return "".join(decoded)


def safe_filename(name: str) -> str:
    name = name.strip().replace("\x00", "")
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta variable de entorno requerida: {name}")
    return value


def connect_gmail(email_addr: str, app_password: str) -> imaplib.IMAP4_SSL:
    client = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    client.login(email_addr, app_password)
    return client


def search_candidate_message_ids(
    client: imaplib.IMAP4_SSL,
    mailbox: str,
    sender: str,
) -> List[bytes]:
    status, _ = client.select(mailbox)
    if status != "OK":
        raise RuntimeError(f"No se pudo abrir el buzón: {mailbox}")

    # Traemos por remitente; luego filtramos por adjunto/nombre al leer mensajes.
    status, data = client.search(None, "FROM", f'"{sender}"')
    if status != "OK":
        raise RuntimeError("Falló la búsqueda IMAP por remitente")

    raw_ids = data[0].split()
    return raw_ids


def fetch_message(client: imaplib.IMAP4_SSL, msg_id: bytes) -> Message:
    status, data = client.fetch(msg_id, "(RFC822)")
    if status != "OK" or not data or data[0] is None:
        raise RuntimeError(f"No se pudo leer el mensaje IMAP id={msg_id!r}")

    raw_email = data[0][1]
    return email.message_from_bytes(raw_email)


def extract_message_meta(msg: Message) -> Dict[str, Any]:
    subject = decode_mime_words(msg.get("Subject"))
    from_ = decode_mime_words(msg.get("From"))
    date_ = decode_mime_words(msg.get("Date"))
    message_id = decode_mime_words(msg.get("Message-ID"))

    return {
        "subject": subject,
        "from": from_,
        "date": date_,
        "message_id": message_id,
    }


def iter_attachments(msg: Message):
    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition", "")).lower()
        if "attachment" not in content_disposition:
            continue

        filename = decode_mime_words(part.get_filename())
        if not filename:
            continue

        yield part, filename


def attachment_matches(
    filename: str,
    contains_1: str,
    contains_2: str,
) -> bool:
    lower_name = filename.lower()

    if not lower_name.endswith(".xlsx"):
        return False

    if contains_1 and contains_1.lower() not in lower_name:
        return False

    if contains_2 and contains_2.lower() not in lower_name:
        return False

    return True


def save_attachment(
    part: Message,
    filename: str,
    save_dir: Path,
    archive_dir: Optional[Path] = None,
) -> Dict[str, str]:
    payload = part.get_payload(decode=True)
    if payload is None:
        raise RuntimeError(f"No se pudo decodificar adjunto: {filename}")

    clean_name = safe_filename(filename)

    latest_path = save_dir / "fillrate_latest.xlsx"
    latest_path.write_bytes(payload)

    saved = {
        "latest_path": str(latest_path),
        "latest_filename": latest_path.name,
    }

    if archive_dir is not None:
        archive_path = archive_dir / clean_name
        archive_path.write_bytes(payload)
        saved["archive_path"] = str(archive_path)
        saved["archive_filename"] = archive_path.name

    return saved


def find_and_download_latest_fillrate(
    mailbox: str = "INBOX",
    env_file: str = ".env",
    max_messages_to_scan: int = 50,
) -> Dict[str, Any]:
    load_env_file(env_file)

    gmail_email = get_required_env("GMAIL_EMAIL")
    gmail_app_password = get_required_env("GMAIL_APP_PASSWORD")
    fillrate_from = get_required_env("FILLRATE_FROM")
    contains_1 = os.getenv("FILLRATE_NAME_CONTAINS", "").strip()
    contains_2 = os.getenv("FILLRATE_NAME_CONTAINS_2", "").strip()
    save_dir = ensure_dir(get_required_env("FILLRATE_SAVE_DIR"))
    archive_dir = ensure_dir(get_required_env("FILLRATE_ARCHIVE_DIR"))

    client = connect_gmail(gmail_email, gmail_app_password)

    try:
        ids = search_candidate_message_ids(
            client=client,
            mailbox=mailbox,
            sender=fillrate_from,
        )

        if not ids:
            return {
                "status": "not_found",
                "reason": "No se encontraron mensajes del remitente objetivo",
                "mailbox": mailbox,
                "sender": fillrate_from,
            }

        # Recorremos del más nuevo al más antiguo
        ids_to_scan = list(reversed(ids))[:max_messages_to_scan]

        for msg_id in ids_to_scan:
            msg = fetch_message(client, msg_id)
            meta = extract_message_meta(msg)

            for part, filename in iter_attachments(msg):
                if not attachment_matches(filename, contains_1, contains_2):
                    continue

                saved = save_attachment(
                    part=part,
                    filename=filename,
                    save_dir=save_dir,
                    archive_dir=archive_dir,
                )

                return {
                    "status": "success",
                    "processed_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mailbox": mailbox,
                    "sender_expected": fillrate_from,
                    "message": meta,
                    "attachment": {
                        "original_filename": filename,
                        **saved,
                    },
                }

        return {
            "status": "not_found",
            "reason": "Se encontraron correos del remitente, pero no adjuntos .xlsx válidos",
            "mailbox": mailbox,
            "sender": fillrate_from,
            "scanned_messages": len(ids_to_scan),
        }

    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            client.logout()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga automática del último FillRate desde Gmail."
    )
    parser.add_argument("--env-file", default=".env", help="Ruta al archivo .env")
    parser.add_argument("--mailbox", default="INBOX", help="Buzón IMAP, default INBOX")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=50,
        help="Cantidad máxima de mensajes recientes a revisar",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Ruta opcional para guardar metadata JSON del rescate",
    )

    args = parser.parse_args()

    try:
        result = find_and_download_latest_fillrate(
            mailbox=args.mailbox,
            env_file=args.env_file,
            max_messages_to_scan=int(args.max_messages),
        )

        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[OK] Metadata guardada en: {out_path}")

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1

    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())