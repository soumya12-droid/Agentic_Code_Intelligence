// Input validation.
const { checkType, checkLength } = require('./helpers');
const { log } = require('./io');

function validate(x) {
  checkType(x);
  if (!checkLength(x)) {
    log('input too long');
    return false;
  }
  return true;
}

module.exports = { validate };
