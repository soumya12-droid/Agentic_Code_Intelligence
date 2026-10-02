// Input preprocessing: runs before the main function.
const { collapseSpaces, toLower } = require('./helpers');

function normalize(s) {
  const trimmed = s.trim();
  const spaced = collapseSpaces(trimmed);
  const lowered = toLower(spaced);
  return lowered;
}

function tokenize(s) {
  return normalize(s).split(' ');
}

module.exports = { normalize, tokenize };
