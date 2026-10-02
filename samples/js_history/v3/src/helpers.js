// Small string helpers shared by the other modules.
function collapseSpaces(s) {
  return s.replace(/\s+/g, ' ');
}

function toLower(s) {
  return s.toLowerCase();
}

function checkType(x) {
  if (typeof x !== 'string') {
    throw new TypeError('bad input');
  }
}

function checkLength(x) {
  return x.length > 0 && x.length < 1000;
}

module.exports = { collapseSpaces, toLower, checkType, checkLength };

function pad(x) {
  return ' ' + x + ' ';
}
