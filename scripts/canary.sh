#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-pw2048}
INGRESS=${INGRESS:-pw2048-canary}
DEPLOYMENT=${DEPLOYMENT:-pw2048-canary}
STABLE_DEPLOYMENT=${STABLE_DEPLOYMENT:-pw2048-stable}
CANARY_SERVICE=${CANARY_SERVICE:-pw2048-canary}
OBSERVE_SECONDS=${OBSERVE_SECONDS:-60}
MAX_ERROR_RATE=${MAX_ERROR_RATE:-0.01}
MAX_AVG_LATENCY=${MAX_AVG_LATENCY:-0.25}
MIN_REQUESTS=${MIN_REQUESTS:-10}

usage() {
  echo "Usage: $0 bootstrap IMAGE [MODEL] [VERSION] | deploy IMAGE [MODEL] [VERSION] | promote | rollback | status"
}

weight() {
  kubectl -n "$NAMESPACE" annotate ingress "$INGRESS" \
    nginx.ingress.kubernetes.io/canary-weight="$1" --overwrite >/dev/null
  echo "canary weight -> $1%"
}

metrics() {
  kubectl -n "$NAMESPACE" run "metrics-$RANDOM" --rm -i --restart=Never \
    --image=curlimages/curl:8.10.1 --command -- \
    curl -fsS "http://$CANARY_SERVICE/metrics" 2>/dev/null
}

gate() {
  local raw requests errors sum error_rate avg
  raw=$(metrics)
  requests=$(awk '/^pw2048_inference_requests_total/ {print $2}' <<<"$raw")
  errors=$(awk '/^pw2048_inference_errors_total/ {print $2}' <<<"$raw")
  sum=$(awk '/^pw2048_inference_duration_seconds_sum/ {print $2}' <<<"$raw")
  requests=${requests:-0}; errors=${errors:-0}; sum=${sum:-0}
  if (( requests < MIN_REQUESTS )); then
    echo "gate failed: only $requests requests, need $MIN_REQUESTS"
    return 1
  fi
  error_rate=$(awk -v e="$errors" -v r="$requests" 'BEGIN {print e/r}')
  avg=$(awk -v s="$sum" -v r="$requests" 'BEGIN {print s/r}')
  echo "gate: requests=$requests error_rate=$error_rate avg_latency=${avg}s"
  awk -v x="$error_rate" -v m="$MAX_ERROR_RATE" 'BEGIN {exit !(x <= m)}' &&
    awk -v x="$avg" -v m="$MAX_AVG_LATENCY" 'BEGIN {exit !(x <= m)}'
}

rollback() {
  weight 0
  kubectl -n "$NAMESPACE" scale deployment "$DEPLOYMENT" --replicas=0 >/dev/null
  echo "rollback complete: all traffic remains on stable"
}

case "${1:-}" in
  bootstrap)
    [[ $# -ge 2 ]] || { usage; exit 2; }
    kubectl apply -f deploy/k8s/all.yaml
    weight 0
    kubectl -n "$NAMESPACE" set image deployment/$STABLE_DEPLOYMENT api="$2"
    kubectl -n "$NAMESPACE" set env deployment/$STABLE_DEPLOYMENT \
      MODEL_NAME="${3:-greedy}" MODEL_VERSION="${4:-v1}" RELEASE_TRACK=stable
    kubectl -n "$NAMESPACE" rollout status deployment/pw2048-stable --timeout=180s
    kubectl -n "$NAMESPACE" scale deployment "$DEPLOYMENT" --replicas=0 >/dev/null
    echo "stable baseline ready: $2 (${3:-greedy}:${4:-v1})"
    ;;
  deploy)
    [[ $# -ge 2 ]] || { usage; exit 2; }
    if ! kubectl -n "$NAMESPACE" get deployment "$STABLE_DEPLOYMENT" >/dev/null 2>&1; then
      echo "stable baseline missing; run '$0 bootstrap STABLE_IMAGE' first" >&2
      exit 1
    fi
    kubectl -n "$NAMESPACE" scale deployment "$DEPLOYMENT" --replicas=1
    kubectl -n "$NAMESPACE" set image deployment/$DEPLOYMENT api="$2"
    kubectl -n "$NAMESPACE" set env deployment/$DEPLOYMENT \
      MODEL_NAME="${3:-heuristic}" MODEL_VERSION="${4:-v2}" RELEASE_TRACK=canary
    kubectl -n "$NAMESPACE" rollout status deployment/$DEPLOYMENT --timeout=180s
    for w in 5 20 50; do
      weight "$w"
      echo "observing for ${OBSERVE_SECONDS}s"
      sleep "$OBSERVE_SECONDS"
      if ! gate; then rollback; exit 1; fi
    done
    echo "canary passed all gates; run '$0 promote' after business-metric approval"
    ;;
  promote)
    image=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o jsonpath='{.spec.template.spec.containers[0].image}')
    model=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODEL_NAME")].value}')
    version=$(kubectl -n "$NAMESPACE" get deployment "$DEPLOYMENT" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODEL_VERSION")].value}')
    kubectl -n "$NAMESPACE" set image deployment/$STABLE_DEPLOYMENT api="$image"
    kubectl -n "$NAMESPACE" set env deployment/$STABLE_DEPLOYMENT MODEL_NAME="$model" MODEL_VERSION="$version" RELEASE_TRACK=stable
    kubectl -n "$NAMESPACE" rollout status deployment/$STABLE_DEPLOYMENT --timeout=180s
    weight 0
    kubectl -n "$NAMESPACE" scale deployment "$DEPLOYMENT" --replicas=0
    echo "promotion complete: $image ($model:$version) is stable"
    ;;
  rollback) rollback ;;
  status)
    kubectl -n "$NAMESPACE" get deploy,pod,svc,ingress -l app=pw2048
    kubectl -n "$NAMESPACE" get ingress "$INGRESS" -o jsonpath='{.metadata.annotations.nginx\.ingress\.kubernetes\.io/canary-weight}'; echo
    ;;
  *) usage; exit 2 ;;
esac
