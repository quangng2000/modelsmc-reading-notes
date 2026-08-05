import { mkdirSync, writeFileSync, appendFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { jsonStringify } from "../ast/render.js";

export interface TraceOptions {
  readonly enabled?: boolean;
  readonly logFile?: string;
}

export class TraceSink {
  private sequence = 0;
  readonly enabled: boolean;
  readonly logFile: string | undefined;

  constructor(options: TraceOptions = {}) {
    this.enabled = options.enabled ?? false;
    this.logFile = options.logFile === undefined ? undefined : resolve(options.logFile);
    if (this.logFile !== undefined) {
      mkdirSync(dirname(this.logFile), { recursive: true });
      writeFileSync(this.logFile, "", "utf8");
    }
  }

  emit(kind: string, message: string, data: Record<string, unknown> = {}): void {
    this.sequence += 1;
    if (this.enabled) console.log(`[trace] ${message}`);
    if (this.logFile !== undefined) {
      const event = { sequence: this.sequence, kind, message, ...data };
      appendFileSync(this.logFile, `${jsonStringify(event)}\n`, "utf8");
    }
  }
}

export class NullTraceSink extends TraceSink {
  constructor() {
    super();
  }
}
