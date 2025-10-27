import './EigenvalueHelpers';
export { CompiledFunctions } from './CompiledFunctions';
export { CompiledResiduals } from './CompiledResiduals';
export { Geometry } from './Geometry';
export { GraphBuilder, type GraphSignature } from './GraphBuilder';
export { lbfgs, type LBFGSOptions, type LBFGSResult } from './LBFGS';
export { Losses } from './Losses';
export { Matrix3x3 } from './Matrix3x3';
export { nonlinearLeastSquares, type NonlinearLeastSquaresOptions, type NonlinearLeastSquaresResult } from './NonlinearLeastSquares';
export { Adam, AdamW, Optimizer, SGD } from './Optimizers';
export { V } from './V';
export { Value } from './Value';
export { Vec2 } from './Vec2';
export { Vec3 } from './Vec3';
export { Vec4 } from './Vec4';

// Symbolic gradient generation
export { parse, Parser } from './symbolic/Parser';
export { differentiate, computeGradients } from './symbolic/SymbolicDiff';
export { simplify } from './symbolic/Simplify';
export { generateCode, generateMathNotation, generateGradientCode, generateGradientFunction } from './symbolic/CodeGen';
export type {
  ASTNode,
  NumberNode,
  VariableNode,
  BinaryOpNode,
  UnaryOpNode,
  FunctionCallNode,
  VectorAccessNode,
  VectorConstructorNode,
  Assignment,
  Program
} from './symbolic/AST';
