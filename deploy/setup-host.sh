#!/usr/bin/env bash
# One-time preparation of a fresh Ubuntu 22.04/24.04 server (for example an AWS EC2 instance).
#   cd ATTS-RAG/deploy && bash setup-host.sh          # CPU server
#   cd ATTS-RAG/deploy && bash setup-host.sh --gpu    # server with an NVIDIA GPU (reboots are needed, see below)
# Safe to run again: it never overwrites an existing deploy/.env.
set -euo pipefail

GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1
cd "$(dirname "$0")"

echo "== Docker =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER" || true
docker compose version >/dev/null 2>&1 || sudo apt-get install -y docker-compose-plugin

if [[ $GPU -eq 1 ]]; then
  echo "== NVIDIA driver and container toolkit =="
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers install
    echo ">> The NVIDIA driver was installed. Reboot (sudo reboot), log in again and run this script again."
    exit 0
  fi
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  nvidia-smi | head -5
fi

echo "== deploy/.env =="
if [[ -f .env ]]; then
  echo ".env exists, left unchanged."
else
  cp .env.production.example .env
  chmod 600 .env
  JWT=$(openssl rand -hex 32)
  # A Fernet key is 32 random bytes in URL-safe base64.
  TOTP=$(openssl rand -base64 32 | tr '+/' '-_')
  sed -i "s|^JWT_SECRET=.*|JWT_SECRET=${JWT}|; s|^TOTP_KEY=.*|TOTP_KEY=${TOTP}|" .env
  if [[ $GPU -eq 1 ]]; then sed -i "s|^OLLAMA_MAX_CONCURRENCY=.*|OLLAMA_MAX_CONCURRENCY=2|" .env; fi
  echo "Created .env with fresh JWT_SECRET and TOTP_KEY. Edit DOMAIN in it before the first start."
  echo "Copy JWT_SECRET and TOTP_KEY to a safe place (password manager): without TOTP_KEY the 2FA secrets are lost."
fi

echo
echo "Next: log out and in again once (so the docker group applies), then follow DEPLOY_AWS.md, step 5."
