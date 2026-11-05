const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, Header, Footer, TableOfContents, PageBreak, VerticalAlign,
  TabStopType, TabStopPosition, NumberFormat, SectionType, LeaderType
} = require("docx");

// ---------- constants (ABNT / ITA) ----------
const FONT = "Times New Roman";
const BLACK = "000000";
const SZ = 24;            // 12 pt body
const SZ_SMALL = 20;      // 10 pt (captions, long quotes)
const LINE = 360;         // 1.5 line spacing
const CONTENT_W = 9071;   // A4 (11906) - left 1701(3cm) - right 1134(2cm)

// ---------- helpers ----------
const T = (text, o = {}) => new TextRun({ text, font: FONT, size: SZ, ...o });
const B = (text) => new TextRun({ text, font: FONT, size: SZ, bold: true });
const I = (text) => new TextRun({ text, font: FONT, size: SZ, italics: true });

// body paragraph (justified, 1.5, first-line indent 1.25cm per ABNT)
const P = (runs, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 0, line: LINE },
  indent: { firstLine: 709 },
  children: Array.isArray(runs) ? runs : [T(runs)],
  ...opts,
});

// numbered chapter / section / subsection headings (left, black)
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 240, line: LINE }, children: [new TextRun({ text: t, font: FONT, size: 36, bold: true, color: BLACK })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 160, line: LINE }, children: [new TextRun({ text: t, font: FONT, size: 28, bold: true, color: BLACK })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 120, line: LINE }, children: [new TextRun({ text: t, font: FONT, size: SZ, bold: true, color: BLACK })] });

// unnumbered centered title (18pt bold) — NOT in TOC
const CTITLE = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360, line: LINE }, children: [new TextRun({ text: t, font: FONT, size: 36, bold: true, color: BLACK })] });
// centered title that IS in TOC (References, Appendix)
const CTITLE_TOC = (t, lvl = HeadingLevel.HEADING_1) => new Paragraph({ heading: lvl, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 360, line: LINE }, children: [new TextRun({ text: t, font: FONT, size: 36, bold: true, color: BLACK })] });

// numbered equation: centered expression + right-aligned (n)
const EQ = (expr, n) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 160, line: LINE },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
  children: [new TextRun({ text: expr, font: FONT, size: SZ, italics: true }), new TextRun({ text: "\t(" + n + ")", font: FONT, size: SZ })],
});

const figCap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 240 }, children: [new TextRun({ text: t, font: FONT, size: SZ_SMALL })] });
const tabCap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 60, line: 240 }, children: [new TextRun({ text: t, font: FONT, size: SZ_SMALL })] });
const img = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: w, height: h }, altText: { title: "figure", description: "figure", name: "figure" } })] });

// ABNT-style table (three-line: top rule, header rule, bottom rule; no verticals, no fill)
const noB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = { style: BorderStyle.SINGLE, size: 6, color: BLACK };
function tcell(content, { w, header = false, last = false, align = AlignmentType.LEFT } = {}) {
  const runs = Array.isArray(content) ? content : [new TextRun({ text: String(content), font: FONT, size: SZ_SMALL, bold: header })];
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: header ? rule : noB, bottom: (header || last) ? rule : noB, left: noB, right: noB },
    margins: { top: 40, bottom: 40, left: 80, right: 80 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 240 }, children: runs })],
  });
}

// ---------- COVER (capa) ----------
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 480 }, children: [new TextRun({ text: "INSTITUTO TECNOLÓGICO DE AERONÁUTICA", font: FONT, size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new ImageRun({ type: "png", data: fs.readFileSync("logos/ita_logo.png"), transformation: { width: 170, height: 64 }, altText: { title: "ITA", description: "ITA", name: "ITA" } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 360 }, children: [new TextRun({ text: "João Matheus Del Vecchio", font: FONT, size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 480 }, children: [new TextRun({ text: "HAWKES PROCESS ESTIMATION ON HIGH-FREQUENCY CRYPTOCURRENCY DATA VIA NEURAL NETWORKS", font: FONT, size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, after: 0 }, children: [new TextRun({ text: "Trabalho de Graduação", font: FONT, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 720 }, children: [new TextRun({ text: "2026", font: FONT, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480 }, children: [new TextRun({ text: "Curso de Engenharia de Computação", font: FONT, size: 52, bold: true })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- TITLE PAGE (folha de rosto) ----------
const titlePage = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, after: 480 }, children: [new TextRun({ text: "João Matheus Del Vecchio", font: FONT, size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 600 }, children: [new TextRun({ text: "HAWKES PROCESS ESTIMATION ON HIGH-FREQUENCY CRYPTOCURRENCY DATA VIA NEURAL NETWORKS", font: FONT, size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 80, line: LINE }, children: [T("Advisor")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600, line: LINE }, children: [T("Prof. Dr. Mauri Aparecido de Oliveira (ITA)")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 600 }, children: [new TextRun({ text: "ENGENHARIA DE COMPUTAÇÃO", font: FONT, size: 32, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, line: 240 }, children: [T("São José dos Campos")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 240 }, children: [T("Instituto Tecnológico de Aeronáutica")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 240 }, children: [T("2026")] }),
];

// ---------- APPROVAL SHEET (folha de aprovação) ----------
const approval = [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 480 }, children: [new TextRun({ text: "HAWKES PROCESS ESTIMATION ON HIGH-FREQUENCY CRYPTOCURRENCY DATA VIA NEURAL NETWORKS", font: FONT, size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 720, line: LINE }, children: [T("This work was accepted as the Final Report of an Undergraduate Thesis (TG-1).")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 0 }, children: [T("_______________________________________________")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40, line: 240 }, children: [T("João Matheus Del Vecchio")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600, line: 240 }, children: [T("Author")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 360, after: 0 }, children: [T("_______________________________________________")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40, line: 240 }, children: [T("Prof. Dr. Mauri Aparecido de Oliveira (ITA)")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 720, line: 240 }, children: [T("Advisor")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480 }, children: [T("São José dos Campos, ____ de __________ de 2026")] }),
];

// ---------- RESUMO ----------
const resumo = [
  new Paragraph({ pageBreakBefore: true, children: [] }),
  CTITLE("Resumo"),
  P([T("Este trabalho propõe replicar e estender o método de estimação de parâmetros de processos de Hawkes por redes neurais recorrentes proposto por Lee (2023), aplicando-o a dados de alta frequência de criptomoedas e investigando arquiteturas alternativas, com destaque para mecanismos de atenção. O processo de Hawkes é um processo pontual auto-excitante e mutuamente excitante, amplamente usado em finanças de alta frequência para descrever o agrupamento temporal de eventos de preço; a partir dele obtém-se, de forma analítica, uma medida de volatilidade aplicável em tempo real. A estimação tradicional por máxima verossimilhança (MLE) é precisa, porém iterativa e custosa. Lee (2023) mostra que uma rede LSTM treinada em trajetórias simuladas estima os parâmetros com acurácia comparável ao MLE e tempo de inferência inferior a 1% do exigido pela otimização numérica. Este trabalho transporta esse método para o mercado de criptomoedas — usando dados públicos e gratuitos de melhor bid/ask (bookTicker) e negociações (aggTrades) dos contratos futuros perpétuos BTCUSDT e ETHUSDT — e compara a LSTM de referência com arquiteturas GRU, LSTM profunda e codificadores baseados em atenção (Transformer), avaliando acurácia, custo computacional e robustez, além da aplicação à volatilidade de Hawkes intradiária. Como proposta de TG-1, o relatório detalha a hipótese, os objetivos, a revisão bibliográfica, a metodologia, os critérios de sucesso, a análise de riscos e o cronograma de atividades para o TG-2.")]),
  new Paragraph({ spacing: { before: 240, line: LINE }, children: [B("Palavras-chave: "), T("Processo de Hawkes. Redes neurais. Atenção. Criptomoedas. Volatilidade de alta frequência.")] }),
];

// ---------- ABSTRACT ----------
const abstract = [
  new Paragraph({ pageBreakBefore: true, children: [] }),
  CTITLE("Abstract"),
  P([T("This work proposes to replicate and extend the recurrent-neural-network method for estimating the parameters of Hawkes processes introduced by Lee (2023), applying it to high-frequency cryptocurrency data and investigating alternative architectures, notably attention mechanisms. The Hawkes process is a self-exciting and mutually exciting point process, widely used in high-frequency finance to describe the temporal clustering of price events; from it one can obtain, analytically, a volatility measure suitable for real-time use. Traditional maximum-likelihood estimation (MLE) is accurate but iterative and costly. Lee (2023) shows that an LSTM network trained on simulated paths estimates the parameters with accuracy comparable to MLE and inference time below 1% of that required by numerical optimisation. This work carries the method to the cryptocurrency market — using free, public best-bid/ask (bookTicker) and trade (aggTrades) data for the BTCUSDT and ETHUSDT perpetual futures — and compares the reference LSTM with GRU, deep-LSTM and attention-based (Transformer) encoders, assessing accuracy, computational cost and robustness, together with the application to intraday Hawkes volatility. As a TG-1 proposal, the report details the hypothesis, objectives, literature review, methodology, success criteria, risk analysis, and the activity schedule for TG-2.")]),
  new Paragraph({ spacing: { before: 240, line: LINE }, children: [B("Keywords: "), T("Hawkes process. Neural networks. Attention. Cryptocurrencies. High-frequency volatility.")] }),
];

// ---------- LIST OF ABBREVIATIONS ----------
const abbr = [
  ["ABNT", "Associação Brasileira de Normas Técnicas"],
  ["Adam", "Adaptive Moment Estimation (optimiser)"],
  ["GRU", "Gated Recurrent Unit"],
  ["LSTM", "Long Short-Term Memory"],
  ["MLE", "Maximum Likelihood Estimation"],
  ["MSE", "Mean Squared Error"],
  ["NBBO", "National Best Bid and Offer"],
  ["RNN", "Recurrent Neural Network"],
  ["TG", "Trabalho de Graduação (Undergraduate Thesis)"],
];
const abbrW = [1900, 7171];
const listAbbr = [
  new Paragraph({ pageBreakBefore: true, children: [] }),
  CTITLE("List of Abbreviations and Acronyms"),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: abbrW, borders: {
    top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB },
    rows: abbr.map(([a, d]) => new TableRow({ children: [
      new TableCell({ width: { size: abbrW[0], type: WidthType.DXA }, borders: { top: noB, bottom: noB, left: noB, right: noB }, margins: { top: 30, bottom: 30, left: 0, right: 80 }, children: [new Paragraph({ spacing: { line: 240 }, children: [B(a)] })] }),
      new TableCell({ width: { size: abbrW[1], type: WidthType.DXA }, borders: { top: noB, bottom: noB, left: noB, right: noB }, margins: { top: 30, bottom: 30, left: 0, right: 0 }, children: [new Paragraph({ spacing: { line: 240 }, children: [T(d)] })] }),
    ] })) }),
];

// ---------- LIST OF SYMBOLS ----------
const syms = [
  ["λ(t)", "conditional intensity (instantaneous event rate)"],
  ["μ", "baseline (exogenous) intensity"],
  ["α₁", "self-excitation amplitude"],
  ["α₂", "mutual (cross) excitation amplitude"],
  ["β", "exponential decay rate of the kernel"],
  ["Φ(t)", "Hawkes influence kernel"],
  ["g", "branching ratio, g = ∫₀^∞ Φ(u) du"],
  ["ϑ", "minimum tick size"],
  ["N(t)", "counting process of events"],
  ["Z", "mark (price-jump size)"],
  ["σ²", "price volatility (variance rate)"],
];
const listSyms = [
  new Paragraph({ pageBreakBefore: true, children: [] }),
  CTITLE("List of Symbols"),
  new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: abbrW, borders: {
    top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB },
    rows: syms.map(([a, d]) => new TableRow({ children: [
      new TableCell({ width: { size: abbrW[0], type: WidthType.DXA }, borders: { top: noB, bottom: noB, left: noB, right: noB }, margins: { top: 30, bottom: 30, left: 0, right: 80 }, children: [new Paragraph({ spacing: { line: 240 }, children: [I(a)] })] }),
      new TableCell({ width: { size: abbrW[1], type: WidthType.DXA }, borders: { top: noB, bottom: noB, left: noB, right: noB }, margins: { top: 30, bottom: 30, left: 0, right: 0 }, children: [new Paragraph({ spacing: { line: 240 }, children: [T(d)] })] }),
    ] })) }),
];

// ---------- CONTENTS (static, with dot leaders) ----------
const tocLine = (label, page, { bold = false, left = 0 } = {}) => new Paragraph({
  spacing: { after: 30, line: 240 }, indent: { left },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: LeaderType.DOT }],
  children: [new TextRun({ text: label, font: FONT, size: SZ, bold }), new TextRun({ text: "\t" + page, font: FONT, size: SZ, bold })],
});
const contents = [
  new Paragraph({ pageBreakBefore: true, children: [] }),
  CTITLE("Contents"),
  tocLine("1  INTRODUCTION", "9", { bold: true }),
  tocLine("1.1  Objectives and hypothesis", "9", { left: 360 }),
  tocLine("1.2  Scope and structure", "10", { left: 360 }),
  tocLine("2  THEORETICAL BACKGROUND", "10", { bold: true }),
  tocLine("2.1  Volatility as activity", "10", { left: 360 }),
  tocLine("2.2  The Hawkes process", "10", { left: 360 }),
  tocLine("2.3  Marked Hawkes price model and volatility", "11", { left: 360 }),
  tocLine("2.4  Estimation: MLE and neural networks", "12", { left: 360 }),
  tocLine("3  RELATED WORK", "12", { bold: true }),
  tocLine("4  DATASET", "13", { bold: true }),
  tocLine("5  METHODOLOGY", "13", { bold: true }),
  tocLine("6  SUCCESS CRITERIA AND RISK ANALYSIS", "14", { bold: true }),
  tocLine("6.1  Success criteria", "14", { left: 360 }),
  tocLine("6.2  Risk analysis", "14", { left: 360 }),
  tocLine("7  ACTIVITY SCHEDULE", "15", { bold: true }),
  tocLine("8  CONCLUSION", "15", { bold: true }),
  tocLine("REFERENCES", "16", { bold: true }),
];

// ---------- comparative table (Related Work) ----------
const cW = [2380, 1500, 2200, 1430, 1561]; // sum 9071
function compRow(a, b, c, d, e, header = false, last = false) {
  return new TableRow({ children: [
    tcell(a, { w: cW[0], header, last }), tcell(b, { w: cW[1], header, last, align: AlignmentType.CENTER }),
    tcell(c, { w: cW[2], header, last, align: AlignmentType.CENTER }), tcell(d, { w: cW[3], header, last, align: AlignmentType.CENTER }),
    tcell(e, { w: cW[4], header, last, align: AlignmentType.CENTER }) ], tableHeader: header });
}
const compTable = new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cW,
  borders: { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB }, rows: [
  compRow("Work", "Estimation", "Architecture", "Market", "Crypto", true),
  compRow("Hawkes (1971)", "—", "—", "—", "No"),
  compRow("Lee and Seo (2017)", "MLE", "—", "Equities", "No"),
  compRow("Bacry et al. (2013)", "MLE", "—", "Equities", "No"),
  compRow("Mei and Eisner (2017)", "Network", "RNN", "General", "No"),
  compRow("Taddeo et al. (2024)", "MLE", "—", "Crypto", "Yes"),
  compRow("Primo (2018)", "MLE", "—", "Equities (BR)", "No"),
  compRow("Lee (2023)", "Network", "LSTM", "Equities (US)", "No"),
  compRow("This work", "Network", "LSTM + attention", "Crypto", "Yes", false, true),
]});

// ---------- data products table (Dataset) ----------
const dW = [2400, 6671];
function dRow(a, b, header = false, last = false) {
  return new TableRow({ children: [ tcell(a, { w: dW[0], header, last }), tcell(b, { w: dW[1], header, last }) ], tableHeader: header });
}
const dataTable = new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: dW,
  borders: { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB }, rows: [
  dRow("Product", "Content and role", true),
  new TableRow({ children: [ tcell("bookTicker", { w: dW[0] }), tcell("Best bid/ask updates (prices, quantities, millisecond timestamps); used to reconstruct the mid-price — the direct analogue of the NBBO used by Lee (2023).", { w: dW[1] }) ] }),
  new TableRow({ children: [ tcell("aggTrades", { w: dW[0], last: true }), tcell("Aggregated trades (price, quantity, time); used to measure market activity and validate the event series.", { w: dW[1], last: true }) ] }),
]});

// ---------- risk table ----------
const rW = [3100, 1100, 4871];
function riskRow(r, pi, m, header = false, last = false) {
  return new TableRow({ children: [ tcell(r, { w: rW[0], header, last }), tcell(pi, { w: rW[1], header, last, align: AlignmentType.CENTER }), tcell(m, { w: rW[2], header, last }) ], tableHeader: header });
}
const riskTable = new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: rW,
  borders: { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB }, rows: [
  riskRow("Risk", "P × I", "Mitigation", true),
  riskRow("Microstructure noise and crypto data quality.", "M × H", "Filtering the mid-price at 0.1 s; public bookTicker/aggTrades dumps; cleaning and validation routines."),
  riskRow("Model mis-specification (subdue effect, negative α estimates).", "M × M", "Softplus output and prediction of β − α₁ − α₂, ensuring positivity and branching ratio below 1; hybrid training (simulation + MLE)."),
  riskRow("Non-stationarity of the crypto market (24/7 operation, regimes).", "H × M", "Window/segment estimation with overlap; per-regime analysis."),
  riskRow("Computational cost and instability of attention-based training.", "M × M", "Start with simple models; regularisation and hyperparameter tuning; the LSTM baseline guarantees a minimum delivery."),
  riskRow("Scope creep for the TG-2 timeline.", "M × H", "Core (C1 + C2) prioritised; attention architectures and volatility as increments.", false, true),
]});

// ---------- references (ABNT, alphabetical) ----------
const REF = (runs) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 240, line: 240 }, children: runs });
const refs = [
  REF([T("ANDERSEN, T. G.; BOLLERSLEV, T.; LABYS, P. Modeling and forecasting realized volatility. "), B("Econometrica"), T(", v. 71, p. 579–625, 2003.")]),
  REF([T("BACRY, E.; DELATTRE, S.; MUZY, J.-F. Modelling microstructure noise with mutually exciting point processes. "), B("Quantitative Finance"), T(", v. 13, p. 65–77, 2013.")]),
  REF([T("BINANCE. "), B("Binance Public Data"), T(": bookTicker and aggTrades datasets. [S. l.], 2024. Disponível em: https://data.binance.vision. Acesso em: 22 jun. 2026.")]),
  REF([T("BOUCHAUD, J.-P.; BONART, J.; DONIER, J.; GOULD, M. "), B("Trades, Quotes and Prices"), T(": financial markets under the microscope. Cambridge: Cambridge University Press, 2018.")]),
  REF([T("HARDIMAN, S. J.; BERCOT, N.; BOUCHAUD, J.-P. Critical reflexivity in financial markets: a Hawkes process analysis. "), B("The European Physical Journal B"), T(", v. 86, n. 442, 2013.")]),
  REF([T("HAWKES, A. G. Spectra of some self-exciting and mutually exciting point processes. "), B("Biometrika"), T(", v. 58, n. 1, p. 83–90, 1971.")]),
  REF([T("HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. "), B("Neural Computation"), T(", v. 9, n. 8, p. 1735–1780, 1997.")]),
  REF([T("KINGMA, D. P.; BA, J. "), B("Adam"), T(": a method for stochastic optimization. [S. l.], 2014. arXiv:1412.6980.")]),
  REF([T("LEE, K. Recurrent neural network based parameter estimation of Hawkes model on high-frequency financial data. "), B("Finance Research Letters"), T(", v. 55, art. 103922, 2023.")]),
  REF([T("LEE, K. Application of Hawkes volatility in the observation of filtered high-frequency price process in tick structures. "), B("Applied Stochastic Models in Business and Industry"), T(", 2024.")]),
  REF([T("LEE, K.; SEO, B. K. Marked Hawkes process modeling of price dynamics and volatility estimation. "), B("Journal of Empirical Finance"), T(", v. 40, p. 174–200, 2017.")]),
  REF([T("MEI, H.; EISNER, J. M. The neural Hawkes process: a neurally self-modulating multivariate point process. "), B("Advances in Neural Information Processing Systems"), T(", v. 30, 2017.")]),
  REF([T("MORETTIN, P. A. "), B("Análise de dados de alta frequência"), T(". São José dos Campos: IME-USP, 2007. Notas de aula.")]),
  REF([T("MUNI TOKE, I. "), B("An introduction to Hawkes processes with applications to finance"), T(". [S. l.]: École Centrale Paris, 2011.")]),
  REF([T("PACURAR, M. Autoregressive conditional duration models in finance: a survey of the theoretical and empirical literature. [S. l.]: Dalhousie University, 2006.")]),
  REF([T("PRIMO, G. O. "), B("Intraday clustering pattern of the Brazilian stock market"), T(": an exploratory analysis with Hawkes processes. 2018. Dissertação (Mestrado em Economia) – Universidade Federal de Pernambuco, Recife, 2018.")]),
  REF([T("TADDEO, M. M.; SALVI, L.; RIVERA-CASTRO, M. A. On the use of Hawkes processes to assess the jump frequency in cryptocurrency log-returns. "), B("Brazilian Review of Finance"), T(", v. 22, n. 2, 2024.")]),
  REF([T("VASWANI, A. et al. Attention is all you need. "), B("Advances in Neural Information Processing Systems"), T(", v. 30, 2017.")]),
];

// ---------- BODY (textual) ----------
const body = [
  H1("1 Introduction"),
  P([T("On very short time scales (high frequency), asset prices do not move independently in time: a trade or a price change tends to raise the probability of new changes in the instants that immediately follow. This temporal grouping of events — known as clustering — is poorly described by Poisson processes, which assume independence between occurrences. The Hawkes process (Hawkes, 1971) is a self-exciting and mutually exciting point process that captures this memory naturally, and is widely used in high-frequency finance to describe the arrival of trades and the tick-level fluctuations of price (Bacry et al., 2013; Lee and Seo, 2017).")]),
  P([T("From a Hawkes model fitted to the up and down movements of the price, one obtains, analytically, a measure of volatility, paving the way for real-time risk monitoring (Lee and Seo, 2017; Lee, 2024). The practical obstacle is the estimation of the model parameters, traditionally carried out by maximum likelihood (MLE): an accurate but iterative and costly procedure, sensitive to the initial value and ill-suited to continuous, real-time applications.")]),
  P([T("Lee (2023) proposes an alternative: training a recurrent neural network (LSTM) to estimate the Hawkes parameters directly from the observed event sequence. Trained on a large set of simulated paths with known parameters, the network learns to map the sequence onto the parameters, achieving accuracy comparable to MLE with inference time below 1% of that required by numerical optimisation. The study, however, is restricted to US equities (2018–2019) and to a single LSTM architecture.")]),
  H2("1.1 Objectives and hypothesis"),
  P([B("Hypothesis. "), T("It is possible to estimate the parameters of a marked Hawkes process from high-frequency cryptocurrency data using a neural network, achieving accuracy comparable to maximum likelihood at orders-of-magnitude lower computational cost; moreover, attention-based architectures can match or surpass the reference LSTM in estimation quality, especially under the more clustered, heavy-tailed jump dynamics of the cryptocurrency market.")]),
  P([B("General objective. "), T("Replicate and extend Lee (2023), applying the method to cryptocurrency data and investigating alternative architectures — notably attention/Transformer — for parameter estimation and real-time volatility computation.")]),
  P([B("Specific objectives. "), T("(i) implement the 2D marked Hawkes model and the MLE baseline; (ii) build the simulation engine (Ogata thinning) and the training-set generator; (iii) reproduce Lee's (2023) three-layer LSTM; (iv) adapt the pipeline to real crypto data; (v) design and train GRU, deep-LSTM and attention/Transformer architectures; (vi) compare accuracy, cost and robustness; and (vii) apply the estimator to intraday Hawkes volatility.")]),
  H2("1.2 Scope and structure"),
  P([T("The work considers a 2D symmetric Hawkes process with exponential decay and marks (jumps), in the high-frequency regime (mid-price filtered at 0.1 s), not ultra-high frequency. It focuses on parameter estimation and volatility measurement; full limit-order-book modelling, price-direction forecasting and order execution are out of scope. The remainder of this report is organised as follows: Chapter 2 presents the theoretical background; Chapter 3 reviews related work; Chapter 4 describes the dataset; Chapter 5 details the methodology; Chapter 6 states the success criteria and the risk analysis; Chapter 7 presents the activity schedule; and Chapter 8 concludes.")]),

  H1("2 Theoretical Background"),
  H2("2.1 Volatility as activity"),
  P([T("Following a minimal microstructure model (Bouchaud et al., 2018), suppose the price moves by ±ϑ (one tick) at a rate ϕ per unit time — the market activity — with up and down moves equally likely. The price volatility is then given by Equation 2.1:")]),
  EQ("σ² = ϕ · ϑ²", "2.1"),
  P([T("To first order, volatility is the rate ϕ of price-change events; modelling volatility therefore amounts to modelling the intensity of a point process. Since activity is empirically intermittent and clustered, with long memory (volatility clustering), a constant-rate Poisson process is inadequate: the intensity itself, ϕ = ϕ(t), must be stochastic. Duration models such as the ACD family (Pacurar, 2006) and the treatment of intraday data (Morettin, 2007) frame this high-frequency setting.")]),
  H2("2.2 The Hawkes process"),
  P([T("The Hawkes process (Hawkes, 1971; Muni Toke, 2011) realises this idea by letting the intensity depend on the process's own past, as in Equation 2.2:")]),
  EQ("ϕ(t) = ϕ₀ + Σ Φ(t − tᵢ),   tᵢ < t", "2.2"),
  P([T("where ϕ₀ is an exogenous baseline intensity and Φ ≥ 0 is an influence kernel that decays in time (the endogenous, self-exciting term). Because ϕ(t) depends on its own history, the intensity is itself a stochastic process; combined with σ²(t) = ϑ²ϕ(t), this yields a volatility that is random and clustered in time, as observed (Bouchaud et al., 2018). The feedback strength is measured by the branching ratio of Equation 2.3:")]),
  EQ("g = ∫₀^∞ Φ(u) du < 1,   with mean intensity μ/(1 − g)", "2.3"),
  P([T("In the branching interpretation, g is the expected number of child events triggered by each event; as g → 1 the market becomes near-critical, with activity dominated by endogenous dynamics (reflexivity), as documented by Hardiman, Bercot and Bouchaud (2013). The full derivations are given in Appendix A.")]),
  H2("2.3 Marked Hawkes price model and volatility"),
  P([T("For tick-level prices, a 2D marked Hawkes model is adopted, with an exponential kernel and a symmetric excitation matrix, as in Equation 2.4:")]),
  EQ("λᵢ(t) = μᵢ + Σⱼ ∫ Φᵢⱼ(t − u) dNⱼ(u),   Φᵢⱼ(t) = αᵢⱼ e^(−βt)", "2.4"),
  P([T("with parameters μ (baseline intensity), α₁ (self-excitation), α₂ (mutual excitation) and β (decay), and up/down moves encoded as types 1 and 2 carrying marks Zᵢ (jump sizes). Under stationarity (branching ratio below 1, i.e. β − α₁ − α₂ > 0), the standard deviation of the net move has a closed form in {μ, α₁, α₂, β}, providing the Hawkes volatility (Lee and Seo, 2017; Lee, 2024). Figure 2.1 illustrates self-excitation in a simulated path.")]),
  img("figs/fig_hawkes.png", 400, 200),
  figCap("Figure 2.1 – Simulated Hawkes process: each event (top) raises the conditional intensity λ(t) (bottom), which decays exponentially until the next event."),
  H2("2.4 Estimation: MLE and neural networks"),
  P([T("The classical estimator maximises the point-process log-likelihood of Equation 2.5:")]),
  EQ("log L = Σ_k log λ(t_k) − ∫₀^T λ(u) du", "2.5"),
  P([T("which, for the Hawkes intensity, must be maximised numerically over {μ, α₁, α₂, β} — the slow, iterative step. Lee (2023) replaces it by a neural network trained on simulated paths to predict the parameters directly. The reference architecture is a three-layer LSTM (Hochreiter and Schmidhuber, 1997); this work additionally investigates GRU, deep-LSTM and attention-based encoders (Vaswani et al., 2017), whose ability to relate distant events directly may better capture the long-range memory of jump clustering. Neural point-process models such as the neural Hawkes process (Mei and Eisner, 2017) provide further context.")]),

  H1("3 Related Work"),
  P([T("The Hawkes process (Hawkes, 1971), with finance-oriented introductions such as Muni Toke (2011), is the central framework. On estimation and volatility, Lee and Seo (2017) formalise the marked Hawkes model and derive volatility estimators, and Lee (2024) deepens the application to filtered high-frequency prices; the reference work, Lee (2023), introduces estimation via a recurrent network. On cryptocurrencies, Taddeo, Salvi and Rivera-Castro (2024) show that the Hawkes process outperforms the Poisson process for jump frequency in crypto log-returns, while Primo (2018) studies the intraday clustering of the Brazilian stock market with Hawkes processes. Table 3.1 positions this work, highlighting the gap at the intersection of neural-network estimation, the cryptocurrency market, attention-based architectures and volatility.")]),
  tabCap("Table 3.1 – Positioning of the work against the literature."),
  compTable,
  P([T(" ")], { spacing: { after: 120 }, indent: { firstLine: 0 } }),

  H1("4 Dataset"),
  P([T("To remain faithful to the reference paper, the dataset must provide the same type of information used by Lee (2023): a mid-price derived from the best bid/ask (the NBBO, in the US-equities case), sampled on a fixed grid of Δt = 0.1 s, recording the time and the new price on each change, with the minimum tick size ϑ scaling the price volatility. The cryptocurrency analogue is the level-1 order book (best bid/ask), available freely and at high frequency.")]),
  P([T("The chosen source is Binance Public Data (data.binance.vision), a free, no-sign-up repository of historical market data distributed as daily CSV files. Two products are used, for the BTCUSDT and ETHUSDT USDT-margined perpetual futures (the two most liquid instruments), as summarised in Table 4.1.")]),
  tabCap("Table 4.1 – Data products from Binance Public Data."),
  dataTable,
  P([T(" ")], { spacing: { after: 120 }, indent: { firstLine: 0 } }),
  P([T("The bookTicker files contain best_bid_price, best_bid_qty, best_ask_price, best_ask_qty and millisecond timestamps (available since 2019-09-21). From them the mid-price is reconstructed as m(t) = ½(best_bid + best_ask), sampled on the 0.1 s grid; changes are detected and labelled up (1) or down (2), and the corresponding inter-arrival times and jump sizes (in tick units) are extracted — exactly the inputs required by the Hawkes model. The aggTrades files provide the transaction flow for activity and validation. The tick size ϑ is taken from the contract's exchangeInfo (PRICE_FILTER), and intraday seasonality is handled by re-clocking time into business time (Bouchaud et al., 2018). The exact date range is left parameterizable: the source and method are fixed, while the specific window — and whether to include a high-volatility stress period — is set during TG-2.")]),

  H1("5 Methodology"),
  P([T("The methodology is an end-to-end pipeline. (i) Data: BTCUSDT and ETHUSDT perpetual futures; mid-price filtered at 0.1 s; parameter ranges calibrated by MLE on real windows. (ii) Simulation: an Ogata thinning engine generates thousands of paths of about 2,000 steps, used as training data with the true parameters as regression targets. (iii) Baseline: the MLE estimator, reproducing Lee's (2023) protocol. (iv) Architectures: the three-layer LSTM (replication), GRU and deep-LSTM variants, and an attention-based encoder (Transformer, the main proposal); in all cases the output layer has four units with a softplus activation and prediction of β − α₁ − α₂, ensuring positivity and branching ratio below 1 (Lee, 2023). (v) Training: the Adam optimiser (Kingma and Ba, 2014) with an MSE loss on the four parameters. (vi) Evaluation: accuracy (MSE against MLE on simulation and real data), computational cost (inference time), robustness, and the application to intraday Hawkes volatility compared with the realized-volatility benchmark (Andersen, Bollerslev and Labys, 2003).")]),

  H1("6 Success Criteria and Risk Analysis"),
  H2("6.1 Success criteria"),
  P([T("(C1) Replication: qualitatively reproduce Lee (2023) — MSE of the same order as MLE and inference time below 1% of MLE. (C2) Real-data pipeline: end-to-end estimation on real crypto data, respecting the model constraints (positivity, branching ratio below 1). (C3) Architectural contribution: at least one attention-based architecture with MSE less than or equal to the LSTM baseline in simulation. (C4) Volatility: an intraday volatility series that is stable and consistent with the realized-volatility benchmark. Criteria C1 and C2 constitute the minimum deliverable; C3 and C4 are the higher-contribution extensions.")]),
  H2("6.2 Risk analysis"),
  P([T("Table 6.1 summarises the main risks, their probability (P) and impact (I) — low (L), medium (M) or high (H) — and the mitigation strategies.")]),
  tabCap("Table 6.1 – Risks and mitigation strategies."),
  riskTable,
  P([T(" ")], { spacing: { after: 120 }, indent: { firstLine: 0 } }),

  H1("7 Activity Schedule"),
  P([T("The schedule below distributes the activities over the second half of 2026 (TG-2). The core of the work — data collection, simulator, MLE baseline and LSTM — is prioritised in the first half of the semester, securing the minimum deliverable (C1 and C2); the attention-based architectures and the volatility application occupy the second half, with the final weeks reserved for writing the monograph and preparing the defense, as shown in Figure 7.1.")]),
  img("figs/fig_gantt.png", 450, 213),
  figCap("Figure 7.1 – Planned activity schedule for TG-2 (second half of 2026)."),

  H1("8 Conclusion"),
  P([T("This proposal outlines a project that combines a well-established stochastic model (Hawkes processes) with modern machine-learning techniques (recurrent networks and attention mechanisms), applied to a market of growing relevance and accessible data (cryptocurrencies). The incremental structure — from the faithful replication of Lee (2023) to the architectural and application extensions — balances low delivery risk with contribution potential, and lays out a clear path toward TG-2. The next steps are to finish collecting and cleaning the high-frequency data, implement the simulator and the MLE baseline, reproduce and validate the reference LSTM, design and train the alternative architectures, run the comparative experiments and the volatility application, and write the final monograph.")]),

  CTITLE_TOC("References"),
  ...refs,
];

// ---------- DOCUMENT ----------
const headerRoman = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ })] })] });
const headerArabic = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ })] })] });
const emptyHeader = new Header({ children: [new Paragraph({ children: [] })] });

const A4 = { width: 11906, height: 16838 };
const MARGINS = { top: 1701, bottom: 1134, left: 1701, right: 1134, header: 851, footer: 851 };

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SZ, color: BLACK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: FONT, color: BLACK }, paragraph: { spacing: { before: 360, after: 240, line: LINE }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: BLACK }, paragraph: { spacing: { before: 280, after: 160, line: LINE }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZ, bold: true, font: FONT, color: BLACK }, paragraph: { spacing: { before: 220, after: 120, line: LINE }, outlineLevel: 2 } },
    ],
  },
  sections: [
    // Section 1: cover + title page (no page number shown)
    { properties: { page: { size: A4, margin: MARGINS } },
      headers: { default: emptyHeader },
      children: [...cover, ...titlePage] },
    // Section 2: pre-textual (roman numerals)
    { properties: { type: SectionType.NEXT_PAGE, page: { size: A4, margin: MARGINS, pageNumbers: { formatType: NumberFormat.LOWER_ROMAN } } },
      headers: { default: headerRoman },
      children: [...approval, ...resumo, ...abstract, ...listAbbr, ...listSyms, ...contents] },
    // Section 3: textual + post-textual (arabic, continuing count)
    { properties: { type: SectionType.NEXT_PAGE, page: { size: A4, margin: MARGINS, pageNumbers: { formatType: NumberFormat.DECIMAL } } },
      headers: { default: headerArabic },
      children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("Report_TG1.docx", buf); console.log("OK", buf.length); });
