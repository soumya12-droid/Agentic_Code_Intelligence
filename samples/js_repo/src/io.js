// Input / output helpers.
function log(message) {
  console.error(message);
}

function readInput() {
  const text = readFileSync('input.txt');
  log('read input');
  return text;
}

function writeOutput(data) {
  writeFileSync('out.txt', data);
  log('wrote output');
}

module.exports = { log, readInput, writeOutput };
