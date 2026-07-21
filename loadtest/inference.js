import http from "k6/http";
import { check } from "k6";
import { Rate } from "k6/metrics";

const inferenceErrors = new Rate("inference_errors");
const target = __ENV.TARGET_URL || "http://localhost:8080";

export const options = {
  scenarios: {
    inference: {
      executor: "ramping-arrival-rate",
      startRate: Number(__ENV.START_RPS || 5),
      timeUnit: "1s",
      preAllocatedVUs: Number(__ENV.PRE_ALLOCATED_VUS || 20),
      maxVUs: Number(__ENV.MAX_VUS || 200),
      stages: [
        { target: Number(__ENV.TARGET_RPS || 25), duration: __ENV.RAMP_DURATION || "30s" },
        { target: Number(__ENV.TARGET_RPS || 25), duration: __ENV.HOLD_DURATION || "60s" },
      ],
    },
  },
  thresholds: {
    http_req_failed: [`rate<${__ENV.MAX_ERROR_RATE || "0.01"}`],
    http_req_duration: [`p(95)<${__ENV.MAX_P95_MS || "250"}`],
    inference_errors: [`rate<${__ENV.MAX_ERROR_RATE || "0.01"}`],
  },
};

const body = JSON.stringify({
  board: [[2, 0, 0, 0], [2, 4, 0, 0], [0, 4, 8, 0], [0, 0, 8, 16]],
});

export default function () {
  const response = http.post(`${target}/v1/move`, body, {
    headers: { "Content-Type": "application/json" },
    tags: { endpoint: "move" },
    timeout: __ENV.REQUEST_TIMEOUT || "2s",
  });
  const ok = check(response, {
    "move returns 200": (r) => r.status === 200,
    "move response has a direction": (r) => {
      try {
        return ["up", "down", "left", "right"].includes(r.json("move"));
      } catch (_) {
        return false;
      }
    },
  });
  inferenceErrors.add(!ok);
}

export function handleSummary(data) {
  return {
    [__ENV.SUMMARY_PATH || "loadtest-summary.json"]: JSON.stringify(data, null, 2),
  };
}
