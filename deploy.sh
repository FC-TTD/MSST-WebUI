#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s verify|build|formal\n' "$0"
}

case "${1:-}" in
  verify)
    ansible-playbook -i 'ttd-edge,' ansible/site.yml --syntax-check
    ;;
  build)
    ansible-playbook -i 'ttd-edge,' ansible/site.yml --tags sync,ci
    ;;
  formal)
    ansible-playbook -i 'ttd-edge,' ansible/site.yml --tags sync,ci,cd
    ;;
  *)
    usage
    exit 2
    ;;
esac
