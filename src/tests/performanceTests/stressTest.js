// tests/performanceTests/stressTest.js

import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '30s', target: 100 }, // Ramp up to 100 users
        { duration: '1m', target: 100 },  // Stay at 100 users
        { duration: '30s', target: 0 },    // Ramp down to 0 users
    ],
};

export default function () {
    const res = http.get('http://localhost:3000/api/users');
    check(res, { 'status was 200': (r) => r.status === 200 });
    sleep(1);
}
