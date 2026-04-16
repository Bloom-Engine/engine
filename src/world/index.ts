// Bloom shared world module.
//
// Defines the on-disk world/prefab schema that the Bloom world editor writes
// and every Bloom game reads. Games and the editor both import from this
// module via `import { ... } from 'bloom/world'`.
//
// Files:
//   - types.ts      schema interfaces (WorldData, PrefabData, ...)
//   - version.ts    schema version + migration
//   - validate.ts   runtime schema validation
//   - loader.ts     loadWorld / instantiateWorld (reads JSON, spawns scene nodes)
//   - saver.ts      saveWorld / savePrefab (writes JSON)
//   - prefab.ts     loadPrefab / expandPrefab / cycle detection
//   - terrain.ts    buildHeightmapMesh / sampleHeight / raycastTerrain / defaultTerrain

export * from './types';
export * from './version';
export * from './validate';
export * from './loader';
export * from './saver';
export * from './prefab';
export * from './terrain';
