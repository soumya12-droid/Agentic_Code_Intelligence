// Output formatting.
function pad(x) {
  return ' ' + x + ' ';
}

function format(x) {
  return pad(String(x));
}

module.exports = { pad, format };
