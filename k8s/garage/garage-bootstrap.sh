#!/usr/bin/env bash
# Bootstraps a Helm-deployed Garage (single node): layout, key, buckets, permissions.
# Usage:
#   ./garage-bootstrap.sh
#   NS=mlops POD=garage-0 CAPACITY=25G ./garage-bootstrap.sh
#   KEY_ID=GK... KEY_SECRET=... ./garage-bootstrap.sh   # use predefined credentials
set -euo pipefail

NS="${NS:-mlops}"
POD="${POD:-garage-0}"
ZONE="${ZONE:-dc1}"
CAPACITY="${CAPACITY:-25G}"          # keep below the data PVC size (30000Mi)
KEY_NAME="${KEY_NAME:-mlops-key}"
BUCKETS="${BUCKETS:-churn-mlops mlflow-artifacts}"
KEY_ID="${KEY_ID:-}"                 # optional: "GK" + 24 hex chars
KEY_SECRET="${KEY_SECRET:-}"         # optional: 64 hex chars

g() { kubectl -n "$NS" exec "$POD" -- /garage "$@"; }

echo "==> Waiting for pod $NS/$POD to be Running"
kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Running "pod/$POD" --timeout=300s

echo "==> Waiting for Garage daemon"
until g status >/dev/null 2>&1; do sleep 2; done

# 1. Layout
if g status 2>&1 | grep -q "NO ROLE ASSIGNED"; then
  NODE_ID="$(g node id -q | head -n1 | cut -d@ -f1)"
  CUR_VER="$(g layout show 2>/dev/null | sed -n 's/.*layout version: *\([0-9][0-9]*\).*/\1/p' | head -n1)"
  CUR_VER="${CUR_VER:-0}"
  NEXT_VER=$((CUR_VER + 1))
  echo "==> Assigning layout (node ${NODE_ID:0:8}, zone $ZONE, capacity $CAPACITY, version $NEXT_VER)"
  g layout assign -z "$ZONE" -c "$CAPACITY" "$NODE_ID"
  g layout apply --version "$NEXT_VER"
else
  echo "==> Layout already assigned, skipping"
fi

# 2. Key
if g key info "$KEY_NAME" >/dev/null 2>&1; then
  echo "==> Key '$KEY_NAME' already exists, skipping"
elif [ -n "$KEY_ID" ] && [ -n "$KEY_SECRET" ]; then
  echo "==> Importing predefined key '$KEY_NAME'"
  g key import --yes -n "$KEY_NAME" "$KEY_ID" "$KEY_SECRET"
else
  echo "==> Creating key '$KEY_NAME' (save the credentials below)"
  g key create "$KEY_NAME"
fi

# 3. Buckets + permissions
for b in $BUCKETS; do
  if g bucket info "$b" >/dev/null 2>&1; then
    echo "==> Bucket '$b' already exists, skipping create"
  else
    echo "==> Creating bucket '$b'"
    g bucket create "$b"
  fi
  g bucket allow --read --write "$b" --key "$KEY_NAME"
done

echo
echo "==> Result"
g status
g bucket list
g key info "$KEY_NAME"
echo
echo "Done. If the key was generated, copy Key ID / Secret key into mlops-secrets"
echo "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY). To reprint the secret later:"
echo "  kubectl -n $NS exec $POD -- /garage key info $KEY_NAME --show-secret"
