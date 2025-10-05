/**
 * Legacy export for backward compatibility.
 * New code should use GraphCanonicalizerNoSort directly.
 * @deprecated Use canonicalizeGraphNoSort from GraphCanonicalizerNoSort instead
 * @internal
 */
export { canonicalizeGraphNoSort as canonicalizeGraph, type CanonicalResult as GraphSignature } from './GraphCanonicalizerNoSort';
