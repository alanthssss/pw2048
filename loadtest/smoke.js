import http from "k6/http";
import { check } from "k6";

const target = __ENV.TARGET_URL || "http://localhost:8080";

export const options = { vus: 1, iterations: 3 };

export default function () {
  const response = http.get(`${target}/readyz`);
  check(response, { "service is ready": (r) => r.status === 200 });
}
