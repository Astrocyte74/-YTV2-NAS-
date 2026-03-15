export interface ResearchMeta {
  status?: 'ok' | 'fallback' | 'error';
  compare?: boolean;
  objective?: string;
  queries?: string[];
  entities?: string[];
  comparison_axes?: string[];
  freshness_sensitive?: boolean;
  requires_source_backed_answer?: boolean;
  provider_mode?: string;
  tool_mode?: string;
  depth?: string;
  provider_chain?: string[];
  source_domains?: string[];
  user_notice?: string;
  fallback_events?: Array<{
    from?: string;
    to?: string;
    reason?: string;
    query?: string;
  }>;
  tool_decisions?: Array<{
    provider?: string;
    query?: string;
    tools?: string[];
    reason?: string;
    site_target?: string | null;
    options?: Record<string, unknown> | null;
  }>;
  result_count?: number;
  source_count?: number;
  by_provider?: Array<{
    provider: string;
    tool_runs: number;
    tools: string[];
    query_count: number;
    result_count: number;
    avg_latency_ms: number;
    unique_source_count?: number;
  }>;
  errors?: string[];
  planner_llm_provider?: string;
  planner_llm_model?: string;
  synth_llm_provider?: string;
  synth_llm_model?: string;
}

export interface ResearchMessageLike {
  content: string;
  research?: ResearchMeta;
}

export interface ResearchReaderOptions {
  includeSummary: boolean;
  includeSources: boolean;
  includeResearchDetails: boolean;
  includeProviderBreakdown: boolean;
  compactTables: boolean;
}

export type ResearchReaderSection =
  | 'includeSummary'
  | 'includeSources'
  | 'includeResearchDetails'
  | 'includeProviderBreakdown';

export interface DerivedResearchSourceItem {
  url: string;
  domain: string;
  label: string;
  displayUrl: string;
}

export interface DerivedResearchReport {
  title: string;
  notice?: string;
  summary?: string;
  bodyMarkdown: string;
  tableMarkdown?: string;
  narrativeMarkdown?: string;
  sourceUrls: string[];
  sourceItems: DerivedResearchSourceItem[];
}

const SOURCE_FILTER_NOTICE_FRAGMENT = 'filtered out because stronger sources were available';
const URL_RE = /https?:\/\/[^\s)>\]]+/g;

function isSourcesHeading(line: string): boolean {
  const normalized = line
    .trim()
    .replace(/^#+\s*/, '')
    .replace(/[*_`]/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s*:\s*$/, '')
    .trim()
    .toLowerCase();
  return normalized === 'sources' || normalized === 'source attribution';
}

function stripMarkdownInline(text: string): string {
  return text
    .replace(/^#{1,6}\s+/, '')
    .replace(/^\*\*([\s\S]+)\*\*$/, '$1')
    .replace(/^__([\s\S]+)__$/, '$1')
    .replace(/[`*_>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function isLowSignalSummary(text: string): boolean {
  const normalized = stripMarkdownInline(text)
    .replace(/[:.]+$/, '')
    .trim()
    .toLowerCase();
  return normalized === 'answer' || normalized === 'short answer' || normalized === 'executive summary';
}

function normalizeSourceUrl(url: string): string {
  return url.replace(/[.,);]+$/, '');
}

function getSourceDisplayItem(url: string): DerivedResearchSourceItem {
  try {
    const parsed = new URL(url);
    const domain = parsed.hostname.replace(/^www\./i, '').toLowerCase();
    return { url, domain, label: domain, displayUrl: url };
  } catch {
    const fallback = url.replace(/^https?:\/\//i, '').split('/')[0] || url;
    return { url, domain: fallback, label: fallback, displayUrl: url };
  }
}

export function isSourceFilterNotice(notice?: string): boolean {
  if (!notice) return false;
  return notice.toLowerCase().includes(SOURCE_FILTER_NOTICE_FRAGMENT);
}

export function isGenericEvidenceTitle(title?: string): boolean {
  if (!title) return false;
  const normalized = stripMarkdownInline(title)
    .replace(/[–—-]/g, ' ')
    .replace(/[:.!?]+$/g, '')
    .replace(/[“”"'`]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
  return [
    'what the evidence says',
    'key points from the evidence',
    'what the sources say',
    'key points from the sources',
    'what the research says',
    'key points from the research',
    'what evidence says',
    'key points from evidence',
    'what sources say',
    'key points from sources',
    'what research says',
    'key points from research',
  ].includes(normalized);
}

function looksLikeStructuralLine(line: string, nextLine?: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return false;
  if (/^#{1,6}\s+/.test(trimmed)) return true;
  if (/^\s*[-*]\s+/.test(trimmed)) return true;
  if (/^\s*\d+\.\s+/.test(trimmed)) return true;
  if (/^```/.test(trimmed)) return true;
  if (trimmed === '---' || trimmed === '***') return true;
  if (trimmed.includes('|') && nextLine && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(nextLine.trim())) return true;
  return false;
}

function isTableDelimiterLine(line: string): boolean {
  return /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(line.trim());
}

function isTableRowLine(line: string): boolean {
  const trimmed = line.trim();
  return Boolean(trimmed) && trimmed.includes('|');
}

function splitLeadingTableSection(bodyMarkdown: string): {
  tableMarkdown?: string;
  narrativeMarkdown?: string;
} {
  const lines = bodyMarkdown.split('\n');
  let index = 0;
  while (index < lines.length && !lines[index].trim()) index += 1;

  const tableLines: string[] = [];
  let capturedAnyTable = false;

  while (index < lines.length) {
    const current = lines[index];
    const next = lines[index + 1];
    if (isTableRowLine(current) && typeof next === 'string' && isTableDelimiterLine(next)) {
      capturedAnyTable = true;
      tableLines.push(current, next);
      index += 2;
      while (index < lines.length && isTableRowLine(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      while (index < lines.length && !lines[index].trim()) {
        const following = lines[index + 1];
        const followingNext = lines[index + 2];
        if (
          typeof following === 'string' &&
          typeof followingNext === 'string' &&
          isTableRowLine(following) &&
          isTableDelimiterLine(followingNext)
        ) {
          tableLines.push('');
          index += 1;
        } else {
          break;
        }
      }
      continue;
    }
    break;
  }

  if (!capturedAnyTable) {
    return { narrativeMarkdown: bodyMarkdown.trim() || undefined };
  }

  while (index < lines.length && !lines[index].trim()) index += 1;
  return {
    tableMarkdown: tableLines.join('\n').trim() || undefined,
    narrativeMarkdown: lines.slice(index).join('\n').trim() || undefined,
  };
}

export function getReportTitleFromContent(content: string, fallback: string): string {
  const lines = (content || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  for (const line of lines) {
    if (/^note:/i.test(line)) continue;
    if (isSourcesHeading(line)) continue;
    const stripped = stripMarkdownInline(line);
    if (!stripped) continue;
    if (stripped.length <= 88) return stripped;
    return `${stripped.slice(0, 85).trimEnd()}...`;
  }

  return fallback.trim() || 'Research';
}

export function deriveResearchReport(
  message: ResearchMessageLike,
  fallbackTitle: string
): DerivedResearchReport | null {
  if (!message.research) return null;

  const rawLines = (message.content || '').split('\n');
  if (rawLines.length === 0) return null;

  let inSources = false;
  let notice: string | undefined;
  let titleLine: string | undefined;
  const bodyLines: string[] = [];
  const sourceUrls: string[] = [];

  for (const line of rawLines) {
    const trimmed = line.trim();
    if (!notice && /^note:/i.test(trimmed)) {
      notice = trimmed.replace(/^note:\s*/i, '').trim();
      continue;
    }
    if (isSourcesHeading(trimmed)) {
      inSources = true;
      continue;
    }
    if (inSources) {
      const matches = trimmed.match(URL_RE) || [];
      matches.forEach((url) => sourceUrls.push(normalizeSourceUrl(url)));
      continue;
    }
    if (!titleLine) {
      const stripped = stripMarkdownInline(trimmed);
      if (stripped && stripped.length >= 16 && stripped.length <= 140 && !looksLikeStructuralLine(trimmed)) {
        titleLine = stripped;
        continue;
      }
    }
    bodyLines.push(line);
  }

  const dedupedSources = Array.from(new Set(sourceUrls));
  const filteredBodyLines = [...bodyLines];
  while (filteredBodyLines.length > 0 && !filteredBodyLines[0].trim()) {
    filteredBodyLines.shift();
  }

  const summaryLines: string[] = [];
  while (filteredBodyLines.length > 0) {
    const current = filteredBodyLines[0];
    const next = filteredBodyLines[1];
    if (!current.trim()) {
      filteredBodyLines.shift();
      if (summaryLines.length > 0) break;
      continue;
    }
    if (looksLikeStructuralLine(current, next)) break;
    summaryLines.push(filteredBodyLines.shift() || '');
  }

  let summary = summaryLines.map((line) => line.trim()).join(' ').trim();
  if (isLowSignalSummary(summary)) summary = '';

  const bodyMarkdown = filteredBodyLines.join('\n').trim();
  const { tableMarkdown, narrativeMarkdown } = splitLeadingTableSection(bodyMarkdown);

  return {
    title: titleLine || getReportTitleFromContent(message.content, fallbackTitle),
    notice,
    summary: summary || undefined,
    bodyMarkdown,
    tableMarkdown,
    narrativeMarkdown,
    sourceUrls: dedupedSources,
    sourceItems: dedupedSources.map(getSourceDisplayItem),
  };
}
