"""Setup and drive the pinned SSL simulator without shell wrappers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

from .layout import CACHE, ROOT

SUMATRA_COMMIT = "0bb4c653ef8611f9e71a880165f101209be97b3c"
SUMATRA_REPOSITORY = "https://github.com/TIGERs-Mannheim/Sumatra.git"
SUMATRA_ZIP_URL = "https://github.com/TIGERs-Mannheim/Sumatra/releases/download/2025/sumatra.zip"
SUMATRA_ZIP_SHA256 = "262a8355112464c70ebcf5de6bf895039cef73706b3383c9b7d5aa7b13023b37"
JAVA_SOURCE = ROOT / "simulation/sumatra/java/SumatraRecorder.java"
SUMATRA_SOURCE = CACHE / "Sumatra"
SUMATRA_ARCHIVE = CACHE / "sumatra.zip"
SUMATRA_RUNTIME = CACHE / "sumatra-dist/sumatra"
JAVA_CLASSES = CACHE / "classes"


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True,
         capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a child while preserving its return code and diagnostic output."""
    return subprocess.run(command, cwd=cwd, check=check, text=True,
                          stdout=subprocess.PIPE if capture_output else None,
                          stderr=subprocess.STDOUT if capture_output else None)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_source() -> None:
    if not SUMATRA_SOURCE.exists():
        _run(["git", "clone", "--no-checkout", SUMATRA_REPOSITORY, str(SUMATRA_SOURCE)])
        _run(["git", "fetch", "--depth", "1", "origin", SUMATRA_COMMIT], cwd=SUMATRA_SOURCE)
        _run(["git", "checkout", "--detach", SUMATRA_COMMIT], cwd=SUMATRA_SOURCE)
        return
    if not (SUMATRA_SOURCE / ".git").exists():
        raise RuntimeError(f"Sumatra source exists without Git metadata; preserving it: {SUMATRA_SOURCE}")
    current = _run(["git", "rev-parse", "HEAD"], cwd=SUMATRA_SOURCE,
                   capture_output=True).stdout.strip()
    if current == SUMATRA_COMMIT:
        return
    dirty = _run(["git", "status", "--porcelain"], cwd=SUMATRA_SOURCE,
                 capture_output=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"Sumatra has local changes; preserving them (HEAD {current}): {SUMATRA_SOURCE}")
    _run(["git", "fetch", "--depth", "1", "origin", SUMATRA_COMMIT], cwd=SUMATRA_SOURCE)
    _run(["git", "switch", "--detach", SUMATRA_COMMIT], cwd=SUMATRA_SOURCE)


def _ensure_archive() -> None:
    if not SUMATRA_ARCHIVE.exists():
        _download(SUMATRA_ZIP_URL, SUMATRA_ARCHIVE)
    with SUMATRA_ARCHIVE.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    if digest != SUMATRA_ZIP_SHA256:
        raise RuntimeError(f"Sumatra archive checksum mismatch: {SUMATRA_ARCHIVE} ({digest})")
    destination = CACHE / "sumatra-dist"
    runtime = destination / "sumatra"
    if runtime.is_dir():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="sumatra-dist-", dir=CACHE))
    try:
        with zipfile.ZipFile(SUMATRA_ARCHIVE) as archive:
            for member in archive.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise RuntimeError(f"Unsafe path in Sumatra archive: {member.filename}")
            archive.extractall(staging)
        if not (staging / "sumatra").is_dir():
            raise RuntimeError("Sumatra archive did not contain the expected sumatra/ runtime")
        if destination.exists():
            if any(destination.iterdir()):
                raise RuntimeError(f"Incomplete Sumatra runtime exists; preserving it: {destination}")
            destination.rmdir()
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def setup() -> None:
    """Ensure the pinned Sumatra source and checksum-verified Java runtime exist."""
    CACHE.mkdir(parents=True, exist_ok=True)
    _ensure_source()
    _ensure_archive()
    _run(["java", "-version"])
    _run(["java", "-m", "jdk.compiler/com.sun.tools.javac.Main", "-version"])


def _prepare_runtime() -> None:
    config = SUMATRA_RUNTIME / "config"
    source_config = SUMATRA_SOURCE / "config"
    if not config.exists():
        shutil.copytree(source_config, config)
    source_xml = source_config / "moduli/sim.xml"
    record_xml = config / "moduli/ssl_record.xml"
    xml = source_xml.read_text(encoding="utf-8")
    xml = xml.replace("<geometry>DIV_A</geometry>", "<geometry>DIV_B</geometry>")
    record_xml.parent.mkdir(parents=True, exist_ok=True)
    record_xml.write_text(xml, encoding="utf-8")


def _compile() -> None:
    _prepare_runtime()
    jars = sorted((SUMATRA_RUNTIME / "lib").glob("*.jar"))
    if not jars:
        raise FileNotFoundError(f"No Sumatra library jars found under {SUMATRA_RUNTIME / 'lib'}")
    JAVA_CLASSES.mkdir(parents=True, exist_ok=True)
    classpath = os.pathsep.join(str(jar) for jar in jars)
    _run(["java", "-m", "jdk.compiler/com.sun.tools.javac.Main", "-proc:none",
          "-cp", classpath, "-d", str(JAVA_CLASSES), str(JAVA_SOURCE)])


def _record_one(output: Path, match_index: int, episodes: int, samples: int,
                seed: int, timeout: int = 180, log_stream=None, expanded=False, fps=15, compile=True) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if compile:
        _compile()
    classpath = os.pathsep.join([str(JAVA_CLASSES), str(SUMATRA_RUNTIME / "lib" / "*")])
    command = ["java", "-Xmx3g", "-cp", classpath, "SumatraRecorder", str(output),
               str(match_index), str(episodes), str(samples), str(seed)]
    if expanded:
        command.extend(["v2", str(fps)])
    process = subprocess.Popen(command, cwd=SUMATRA_RUNTIME, stdout=log_stream,
                               stderr=subprocess.STDOUT if log_stream is not None else None)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_child(process)
        raise TimeoutError(f"Sumatra recording timed out after {timeout}s: {output}")
    except BaseException:
        _stop_child(process)
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def record(output: Path, match_index: int = 0, episodes: int = 6,
           samples: int = 128, seed: int = 20261116) -> None:
    """Record one Sumatra session JSONL, compiling the unchanged Java harness."""
    _record_one(Path(output), match_index, episodes, samples, seed)


def generate(dataset: Path) -> None:
    """Create 12 six-episode sessions and a one-episode replay check."""
    dataset = Path(dataset).expanduser().resolve()
    if dataset.exists():
        raise FileExistsError(f"Refusing to overwrite {dataset}")
    trajectories = dataset / "trajectories"
    trajectories.mkdir(parents=True)
    for session in range(12):
        name = f"session_{session:03d}"
        with (trajectories / f"{name}.log").open("w", encoding="utf-8") as output_log:
            _record_one(trajectories / f"{name}.jsonl", session, 6, 128,
                        20261116, timeout=195, log_stream=output_log)
    replay_dir = dataset / "replay_check"
    replay_dir.mkdir()
    with (dataset / "replay_check.log").open("w", encoding="utf-8") as output_log:
        _record_one(replay_dir / "session_000.jsonl", 0, 1, 128,
                    20261116, timeout=195, log_stream=output_log)


def fetch_image(image: str = "roboticserlangen/simulatorcli", reference: str = "latest",
                output: Path | None = None, inspect_only: bool = False) -> dict:
    """Fetch and safely extract the ERForce OCI image (optional tooling)."""
    import requests

    output = Path(output) if output is not None else CACHE / "erforce-image"
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = session.get("https://auth.docker.io/token",
                        params={"service": "registry.docker.io", "scope": f"repository:{image}:pull"}, timeout=30)
    token.raise_for_status()
    session.headers["Authorization"] = "Bearer " + token.json()["token"]
    session.headers["Accept"] = ",".join([
        "application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])
    base = f"https://registry-1.docker.io/v2/{image}"
    response = session.get(f"{base}/manifests/{reference}", timeout=30)
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
    evidence = {"image": image, "reference": reference, "index_digest": index_digest,
                "manifest_digest": digest, "manifest": manifest, "config": config_response.json()}
    (output / "image.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if inspect_only:
        return evidence
    root = output / "rootfs"
    root.mkdir(exist_ok=True)
    prefixes = ("usr/lib/", "usr/local/", "lib/", "lib64/", "home/default/")

    def safe_member(member, destination):
        member.name = member.name.lstrip("./")
        if not member.name.startswith(prefixes) or "/.wh." in member.name:
            return None
        if member.issym() and member.linkname.startswith("/"):
            member.linkname = posixpath.relpath(member.linkname.lstrip("/"), posixpath.dirname(member.name))
        return tarfile.data_filter(member, destination)

    for layer in manifest["layers"]:
        filename = output / (layer["digest"].replace(":", "-") + ".tar.gz")
        if not filename.exists():
            temporary = filename.with_suffix(".partial")
            with session.get(f"{base}/blobs/{layer['digest']}", stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                with temporary.open("wb") as target:
                    for chunk in stream.iter_content(1024 * 1024):
                        target.write(chunk)
            temporary.replace(filename)
        with filename.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if "sha256:" + actual != layer["digest"]:
            raise RuntimeError("OCI layer checksum mismatch")
        with tarfile.open(filename) as archive:
            archive.extractall(root, filter=safe_member)
    (output / "READY").write_text(digest + "\n", encoding="utf-8")
    return evidence


def run_erforce(*args: str) -> None:
    """Run the optional cached ERForce simulator image without Docker."""
    root = CACHE / "erforce-image/rootfs"
    candidates = list(root.rglob("ld-*.so"))
    if not candidates:
        raise FileNotFoundError(f"No dynamic loader found under {root}")
    loader = candidates[0]
    library_path = ":".join([str(root / "usr/lib/x86_64-linux-gnu"), str(root / "lib/x86_64-linux-gnu"), str(root / "usr/local/lib")])
    command = [str(loader), "--library-path", library_path,
               str(root / "home/default/build/bin/simulator-cli"), *args]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(command, cwd=root / "home/default", env=env, check=True)


def generate_sim_protocol() -> None:
    """Generate Python protobuf modules from the cached ERForce protocol sources."""
    source = CACHE / "erforce/src/protobuf"
    output = CACHE / "generated"
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("protobuf/ssl_sim/*.proto"))
    files += sorted(source.glob("protobuf/ssl_vision/*.proto"))
    files += [source / "protobuf/ssl_gc/state/ssl_gc_common.proto"]
    _run([sys.executable, "-m", "grpc_tools.protoc", "-I", str(source),
          f"--python_out={output}", *(str(path) for path in files)], cwd=ROOT)
