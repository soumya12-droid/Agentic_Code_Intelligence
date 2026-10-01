// Configuration loading.
const defaults = () => ({ verbose: false, limit: 100 });

function parseConfig(text) {
  const base = defaults();
  return Object.assign(base, JSON.parse(text));
}

function loadConfig() {
  const raw = readFileSync('config.json');
  return parseConfig(raw);
}

module.exports = { loadConfig, parseConfig };
