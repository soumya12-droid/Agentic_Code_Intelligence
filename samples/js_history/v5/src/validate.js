// Input validation.
const { checkType, checkLength } = require('./helpers');
const { log } = require('./io');

function validate(x) {
  log('validating input');
  checkType(x);
  if (x.trim() === '') {
    return false;
  }
  if (!checkLength(x)) {
    log('input is too long');
    return false;
  }
  return true;
}

module.exports = { validate };
