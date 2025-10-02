import type { Demo } from './types';

// Demo registry - add new demos here
const demoModules: Record<string, Demo> = {};

// Dynamically import demos
// For now, we'll add them manually as we create them

export function registerDemo(demo: Demo) {
  demoModules[demo.metadata.id] = demo;
}

export function getDemo(id: string): Demo | undefined {
  return demoModules[id];
}

export function getAllDemos(): Demo[] {
  return Object.values(demoModules);
}

export function getDemosByTag(tag: string): Demo[] {
  return getAllDemos().filter(demo => demo.metadata.tags.includes(tag));
}

export function getDemosByDifficulty(difficulty: 'beginner' | 'intermediate' | 'advanced'): Demo[] {
  return getAllDemos().filter(demo => demo.metadata.difficulty === difficulty);
}
