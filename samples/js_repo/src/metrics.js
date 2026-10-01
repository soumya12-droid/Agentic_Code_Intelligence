// Timing helpers.
const { log } = require('./io');

function perf(fn) {
  const start = Date.now();
  fn();
  return Date.now() - start;
}

function report(fn) {
  const ms = perf(fn);
  log('took ' + ms);
}

module.exports = { perf, report };
