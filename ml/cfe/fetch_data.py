import hashlib
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile

PUBLIC_KEY = "https://disk.yandex.ru/d/LE74B6p4wfEapg"
FOLDER = "/3. МТС: MWS: CompactFactExtraction"
ARCHIVE = "Данные кейса.zip"
FILES = ["train.jsonl", "val.jsonl", "baseline_pred.jsonl", "case_metrics.py",
         "catalog.json", "schema.md", "README.md", "baseline_metrics.json", "CHECKSUMS.txt"]
NESTED = ("CompactFactExtraction_participants.zip", "01_public_participants/ANSWERS_v2.md")
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def download():
    q = urllib.parse.urlencode({"public_key": PUBLIC_KEY, "path": f"{FOLDER}/{ARCHIVE}"})
    with urllib.request.urlopen(f"{API}?{q}") as r:
        href = json.load(r)["href"]
    with urllib.request.urlopen(href) as r:
        return r.read()


def main():
    os.makedirs("data/raw", exist_ok=True)
    zf = zipfile.ZipFile(io.BytesIO(download()))
    out = {name: zf.read(name) for name in FILES}
    nested = zipfile.ZipFile(io.BytesIO(zf.read(NESTED[0])))
    out["ANSWERS_v2.md"] = nested.read(NESTED[1])
    for name, data in out.items():
        with open(f"data/raw/{name}", "wb") as f:
            f.write(data)
        print(name, len(data))
    sums = {}
    for line in out["CHECKSUMS.txt"].decode("utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip().lstrip("*")] = parts[0]
    for name, data in out.items():
        if name == "CHECKSUMS.txt":
            continue
        if name not in sums:
            print("no checksum:", name)
        elif hashlib.sha256(data).hexdigest() != sums[name]:
            raise SystemExit(f"checksum mismatch: {name}")


if __name__ == "__main__":
    main()
