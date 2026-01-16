import path from "node:path";
import { pathToFileURL } from "node:url";

import { startWorker } from "./worker.js";

type WorkerArgs = {
  bridge?: string;
  workerId?: number;
  userModules: string[];
};

function parseArgs(argv: string[]): WorkerArgs {
  const args: WorkerArgs = { userModules: [] };
  let idx = 0;
  while (idx < argv.length) {
    const value = argv[idx];
    if (value === "--bridge") {
      args.bridge = argv[idx + 1];
      idx += 2;
      continue;
    }
    if (value === "--worker-id") {
      const parsed = Number(argv[idx + 1]);
      if (Number.isFinite(parsed)) {
        args.workerId = parsed;
      }
      idx += 2;
      continue;
    }
    if (value === "--user-module") {
      const moduleName = argv[idx + 1];
      if (moduleName) {
        args.userModules.push(moduleName);
      }
      idx += 2;
      continue;
    }
    idx += 1;
  }
  return args;
}

async function loadModule(moduleName: string) {
  if (moduleName.startsWith(".") || path.isAbsolute(moduleName)) {
    const resolved = path.resolve(moduleName);
    await import(pathToFileURL(resolved).href);
    return;
  }
  await import(moduleName);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.bridge) {
    console.error("missing --bridge");
    process.exit(1);
  }
  if (typeof args.workerId !== "number") {
    console.error("missing --worker-id");
    process.exit(1);
  }

  for (const moduleName of args.userModules) {
    await loadModule(moduleName);
  }

  startWorker({ workerId: args.workerId, target: args.bridge });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
