// The main function: load, read, preprocess, validate, render.
const { loadConfig } = require('./config');
const { readInput } = require('./io');
const { normalize } = require('./preprocess');
const { validate } = require('./validate');
const { render } = require('./render');

function main() {
  const config = loadConfig();
  const raw = readInput();
  const clean = normalize(raw);
  if (validate(clean)) {
    render(clean);
  }
  return config;
}

module.exports = { main };
