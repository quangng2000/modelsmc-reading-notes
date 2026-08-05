export class CoreInvariantError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CoreInvariantError";
  }
}
