/* What the reader is currently looking at, shared by every module that can change it. */

export const state = {
  paper: null, version: 1, hits: [], pendingHits: null,
  selection: null, candidate: null, models: [], busy: false,
  breadth: 'balanced', abort: null, lastAsk: null, lastAnswerChunk: null,
  /* What the citation-graph heat is measured against: the most recent search or question,
   * whichever happened last. Opening a paper deliberately does NOT overwrite it — you
   * arrived at that paper *from* a query, and that query is what you still want the
   * neighbourhood shaded by. */
  heatRef: null,
};
