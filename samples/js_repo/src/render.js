// Rendering of results.
const { format } = require('./format');
const { writeOutput } = require('./io');

class Renderer {
  constructor() {
    this.reset();
  }

  reset() {
    this.buffer = [];
  }

  draw(items) {
    this.clear();
    items.forEach((item) => this.buffer.push(format(item)));
    return this.buffer.join('\n');
  }

  clear() {
    this.buffer.length = 0;
  }
}

function render(data) {
  const r = new Renderer();
  const text = r.draw(data);
  writeOutput(text);
  return text;
}

module.exports = { Renderer, render };
