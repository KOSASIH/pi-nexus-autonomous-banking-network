export function getCurrentTime() {
  return new Date().getTime();
}

export function formatTime(timestamp) {
  return new Date(timestamp).toLocaleString();
}
