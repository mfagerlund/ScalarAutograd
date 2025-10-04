/**
 * Legacy export for backward compatibility.
 * New code should use GraphCanonicalizer directly.
 * @deprecated Use canonicalizeGraph from GraphCanonicalizer instead
 * @internal
 */
export { canonicalizeGraph, type CanonicalResult as GraphSignature } from './GraphCanonicalizer';
