export type IntelFn<T> = (project: Project) => Promise<T> | T;

export interface IntelRunResult<T> {
  readonly project: Project;
  readonly result: T;
}

export async function runIntel<T>(fn: IntelFn<T>): Promise<IntelRunResult<T>> {
  const project = await loadProject();
  const result = await fn(project);
  return { project, result };
}
