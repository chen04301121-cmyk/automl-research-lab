/** Diagram config for the teaching “simple research loop” (live React Flow). */

export type SimpleLoopNodeId =
  | 'dataset'
  | 'data_scientist'
  | 'data_quality_auditor'
  | 'hypothesis_generator'
  | 'ml_engineer'
  | 'threshold_tuner'
  | 'judge'
  | 'router'
  | 'memory_writer'
  | 'research_memory'
  | 'context_builder'
  | 'holdout_eval'
  | 'best_candidate'

export type SimpleLoopNodeKind = 'file' | 'llm' | 'code' | 'code_llm' | 'store'

export interface SimpleLoopNode {
  id: SimpleLoopNodeId
  title: string
  role: string
  kind: SimpleLoopNodeKind
  service: string
}

export interface SimpleLoopEdge {
  id: string
  source: SimpleLoopNodeId
  target: SimpleLoopNodeId
  kind: 'control' | 'data'
  label?: string
  sourceHandle?: string
  targetHandle?: string
}

export const SIMPLE_LOOP_NODES: SimpleLoopNode[] = [
  {
    id: 'dataset',
    title: 'dataset.csv',
    role: 'Input table · choose target',
    kind: 'file',
    service: 'dataset file',
  },
  {
    id: 'data_scientist',
    title: 'Data Scientist',
    role: 'Profile + leakage only',
    kind: 'llm',
    service: 'node script',
  },
  {
    // NEW agent (IDS706 Week 11): measured leakage / imbalance checks
    id: 'data_quality_auditor',
    title: 'Data Quality Auditor',
    role: 'Leakage AUC scan · imbalance · ids',
    kind: 'code',
    service: 'node script',
  },
  {
    id: 'hypothesis_generator',
    title: 'Hypothesis Generator',
    role: 'Conceptual next step (no HPT)',
    kind: 'llm',
    service: 'node script',
  },
  {
    id: 'ml_engineer',
    title: 'ML Engineer',
    role: 'HPT config · train · repair',
    kind: 'code_llm',
    service: 'node script',
  },
  {
    // NEW agent (IDS706 Week 11): only runs when the auditor flags imbalance
    id: 'threshold_tuner',
    title: 'Threshold Tuner',
    role: 'F1-optimal cut-off (imbalanced only)',
    kind: 'code',
    service: 'node script',
  },
  {
    id: 'judge',
    title: 'Judge',
    role: 'Verdict from val evidence',
    kind: 'llm',
    service: 'node script',
  },
  {
    id: 'router',
    title: 'Router',
    role: 'accept · revise · stop',
    kind: 'code',
    service: 'langgraph edge',
  },
  {
    id: 'memory_writer',
    title: 'Memory writer',
    role: 'File verdict + lesson',
    kind: 'code',
    service: 'node script',
  },
  {
    id: 'research_memory',
    title: 'Research memory',
    role: 'history · findings · lessons',
    kind: 'store',
    service: 'json store',
  },
  {
    id: 'context_builder',
    title: 'Context builder',
    role: 'Brief for next hypothesis',
    kind: 'code',
    service: 'node script',
  },
  {
    id: 'best_candidate',
    title: 'Best candidate',
    role: 'Accepted model + val evidence',
    kind: 'store',
    service: 'state',
  },
  {
    id: 'holdout_eval',
    title: 'Final holdout',
    role: 'Open sealed holdout once',
    kind: 'code',
    service: 'node script',
  },
]

/**
 * Grid: x step 260, y step 210 — keeps default edges straight (L/R or T/B).
 */
export const SIMPLE_LOOP_EDGES: SimpleLoopEdge[] = [
  // Entry — CSV into Data Scientist
  {
    id: 'e-csv-ds',
    source: 'dataset',
    target: 'data_scientist',
    kind: 'data',
    label: 'csv',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  // Main spine — horizontal
  {
    id: 'e-ds-audit',
    source: 'data_scientist',
    target: 'data_quality_auditor',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-audit-hyp',
    source: 'data_quality_auditor',
    target: 'hypothesis_generator',
    kind: 'control',
    label: 'drops',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-hyp-ml',
    source: 'hypothesis_generator',
    target: 'ml_engineer',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  // Conditional: imbalanced -> threshold tuner (detour above), balanced -> judge
  {
    id: 'e-ml-judge',
    source: 'ml_engineer',
    target: 'judge',
    kind: 'control',
    label: 'balanced',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-ml-tuner',
    source: 'ml_engineer',
    target: 'threshold_tuner',
    kind: 'control',
    label: 'imbalanced',
    sourceHandle: 'top-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-tuner-judge',
    source: 'threshold_tuner',
    target: 'judge',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'top-in',
  },
  {
    id: 'e-judge-router',
    source: 'judge',
    target: 'router',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  // Revise loop — down from router, left into memory writer, left, up into hyp
  {
    id: 'e-router-mem',
    source: 'router',
    target: 'memory_writer',
    kind: 'control',
    label: 'revise',
    sourceHandle: 'bottom-out',
    targetHandle: 'right-in',
  },
  {
    id: 'e-mem-store',
    source: 'memory_writer',
    target: 'research_memory',
    kind: 'data',
    sourceHandle: 'left-out',
    targetHandle: 'right-in',
  },
  {
    id: 'e-store-ctx',
    source: 'research_memory',
    target: 'context_builder',
    kind: 'data',
    sourceHandle: 'left-out',
    targetHandle: 'right-in',
  },
  {
    id: 'e-ctx-hyp',
    source: 'context_builder',
    target: 'hypothesis_generator',
    kind: 'control',
    label: 'brief',
    sourceHandle: 'top-out',
    targetHandle: 'bottom-in',
  },
  // Accept / stop — straight down under router, then right to holdout
  {
    id: 'e-router-best',
    source: 'router',
    target: 'best_candidate',
    kind: 'control',
    label: 'accept/stop',
    sourceHandle: 'bottom-out',
    targetHandle: 'top-in',
  },
  {
    id: 'e-best-hold',
    source: 'best_candidate',
    target: 'holdout_eval',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
]

/**
 * Classroom snap layout (x step 260, y: 40 / 260 / 480).
 * Columns align so control edges stay axis-aligned (straight step paths):
 *   upper:                              tuner (imbalanced detour)
 *   top:    dataset · ds · auditor · hyp · ml · judge · router
 *   mid:               ctx · mem · writer
 *   bottom:                         best · holdout
 */
export const SIMPLE_LOOP_POSITIONS: Record<SimpleLoopNodeId, { x: number; y: number }> = {
  // Upper detour row: threshold tuner sits between ML engineer and judge
  threshold_tuner: { x: 1300, y: -180 },
  // Main row
  dataset: { x: 0, y: 40 },
  data_scientist: { x: 260, y: 40 },
  data_quality_auditor: { x: 520, y: 40 },
  hypothesis_generator: { x: 780, y: 40 },
  ml_engineer: { x: 1040, y: 40 },
  judge: { x: 1560, y: 40 },
  router: { x: 1820, y: 40 },
  // Revise loop row (context builder directly under hypothesis)
  context_builder: { x: 780, y: 260 },
  research_memory: { x: 1300, y: 260 },
  memory_writer: { x: 1560, y: 260 },
  // Finalize row
  best_candidate: { x: 1820, y: 480 },
  holdout_eval: { x: 2080, y: 480 },
}

export const SIMPLE_LOOP_FLOW = [
  'data_scientist',
  'data_quality_auditor',
  'hypothesis_generator',
  'ml_engineer',
  'threshold_tuner',
  'judge',
  'router',
] as const

export const GRAPH_NODE_TO_VISUAL: Record<string, SimpleLoopNodeId> = {
  data_scientist: 'data_scientist',
  data_quality_auditor: 'data_quality_auditor',
  hypothesis_generator: 'hypothesis_generator',
  ml_engineer: 'ml_engineer',
  threshold_tuner: 'threshold_tuner',
  judge: 'judge',
  router: 'router',
  memory_writer: 'memory_writer',
  research_memory: 'research_memory',
  context_builder: 'context_builder',
  holdout_eval: 'holdout_eval',
}
