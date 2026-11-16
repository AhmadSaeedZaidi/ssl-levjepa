"""Fetch a public OCI simulator image into project-local storage, without Docker."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import tarfile

import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="roboticserlangen/simulatorcli")
    ap.add_argument("--reference", default="latest")
    ap.add_argument("--output", type=Path, default=Path(".sim/erforce-image"))
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = session.get("https://auth.docker.io/token", params={"service": "registry.docker.io", "scope": f"repository:{args.image}:pull"}, timeout=30)
    token.raise_for_status()
    session.headers["Authorization"] = "Bearer " + token.json()["token"]
    session.headers["Accept"] = ",".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])
    base = f"https://registry-1.docker.io/v2/{args.image}"
    response = session.get(f"{base}/manifests/{args.reference}", timeout=30)
    response.raise_for_status()
    index_digest = response.headers.get("Docker-Content-Digest")
    manifest = response.json()
    if "manifests" in manifest:
        entry = next(m for m in manifest["manifests"] if m["platform"]["os"] == "linux" and m["platform"]["architecture"] == "amd64")
        response = session.get(f"{base}/manifests/{entry['digest']}", timeout=30)
        response.raise_for_status()
        manifest = response.json()
    digest = response.headers.get("Docker-Content-Digest")
    config_response = session.get(f"{base}/blobs/{manifest['config']['digest']}", timeout=30)
    config_response.raise_for_status()
    evidence = {"image": args.image, "reference": args.reference, "index_digest": index_digest,
                "manifest_digest": digest, "manifest": manifest, "config": config_response.json()}
    (args.output / "image.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"image": args.image, "digest": digest, "compressed_bytes": sum(m["size"] for m in manifest["layers"]),
                      "labels": evidence["config"].get("config", {}).get("Labels")}), flush=True)
    if args.inspect_only:
        return
    root = args.output / "rootfs"
    root.mkdir(exist_ok=True)
    # Only runtime libraries and the simulator/configuration are needed. No global installs.
    prefixes = ("usr/lib/", "usr/local/", "lib/", "lib64/", "home/default/")
    def safe_member(member, destination):
        member.name = member.name.lstrip("./")
        if not member.name.startswith(prefixes) or "/.wh." in member.name:
            return None
        if member.issym() and member.linkname.startswith("/"):
            member.linkname = posixpath.relpath(member.linkname.lstrip("/"), posixpath.dirname(member.name))
        return tarfile.data_filter(member, destination)
    for layer in manifest["layers"]:
        filename = args.output / (layer["digest"].replace(":", "-")+".tar.gz")
        if not filename.exists():
            with session.get(f"{base}/blobs/{layer['digest']}", stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                temporary = filename.with_suffix(".partial")
                with temporary.open("wb") as output:
                    for chunk in stream.iter_content(1024*1024):
                        output.write(chunk)
                temporary.rename(filename)
        with filename.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if "sha256:"+actual != layer["digest"]:
            raise RuntimeError("OCI layer checksum mismatch")
        with tarfile.open(filename) as archive:
            archive.extractall(root, filter=safe_member)
        print(f"verified/extracted {layer['digest']} ({layer['size']} bytes)", flush=True)
    (args.output / "READY").write_text(digest+"\n")


if __name__ == "__main__":
    main()
