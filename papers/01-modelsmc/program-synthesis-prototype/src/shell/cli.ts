#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import { main } from "./cli/index.js";

export { CliArgumentError, parseCliArgs, USAGE, type CliOptions } from "./cli/index.js";
export { main } from "./cli/index.js";

const entryPath = process.argv[1];
if (entryPath !== undefined && import.meta.url === pathToFileURL(entryPath).href) {
  process.exitCode = await main();
}
