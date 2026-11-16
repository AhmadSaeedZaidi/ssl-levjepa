#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [ ! -d upstream/LeVJEPA ]; then
  mkdir -p upstream
  git clone https://github.com/MLO-lab/LeVJEPA.git upstream/LeVJEPA
fi
{
  date -Is
  uname -a
  cat /etc/os-release
  df -h . /home
  free -h
  nvidia-smi
  python3 --version
  git --version
  command -v uv tmux curl || true
  git -C upstream/LeVJEPA remote -v
  git -C upstream/LeVJEPA rev-parse HEAD
  git -C upstream/LeVJEPA status --short
} | tee logs/environment.txt
find upstream/LeVJEPA -name AGENTS.md -print
