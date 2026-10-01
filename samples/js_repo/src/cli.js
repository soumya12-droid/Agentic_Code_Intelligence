// Command line entry point (uses an aliased import).
import { main } from './index';
import { normalize as norm } from './preprocess';
import { report } from './metrics';

function parseArgs(argv) {
  return argv.slice(2).map((a) => norm(a));
}

function run() {
  const args = parseArgs(process.argv);
  report(() => main());
  return args;
}

run();
