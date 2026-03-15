import type { ReactNode } from 'react';

import { ResearchMarkdown } from './ResearchMarkdown';
import {
  deriveResearchReport,
  isGenericEvidenceTitle,
  isSourceFilterNotice,
  type ResearchMessageLike,
  type ResearchMeta,
  type ResearchReaderOptions,
  type ResearchReaderSection,
} from './research-report';

export interface ResearchReportViewProps {
  message: ResearchMessageLike;
  fallbackTitle?: string;
  options?: Partial<ResearchReaderOptions>;
  tone?: 'dark' | 'paper';
  onDismissSection?: (section: ResearchReaderSection) => void;
}

const DEFAULT_OPTIONS: ResearchReaderOptions = {
  includeSummary: true,
  includeSources: true,
  includeResearchDetails: true,
  includeProviderBreakdown: true,
  compactTables: false,
};

function providerLabel(provider?: string): string {
  if (!provider) return '';
  if (provider === 'inception') return 'Mercury';
  if (provider === 'openrouter') return 'OpenRouter';
  return provider;
}

function modelSuffix(model?: string): string {
  if (!model) return '';
  return ` (${model.split('/').pop()})`;
}

function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="reader-report-detail-block">
      <div className="reader-report-detail-label">{label}</div>
      <div className="reader-report-detail-body">{children}</div>
    </div>
  );
}

function DismissButton({
  label,
  section,
  onDismiss,
}: {
  label: string;
  section: ResearchReaderSection;
  onDismiss?: (section: ResearchReaderSection) => void;
}) {
  if (!onDismiss) return null;
  return (
    <button
      type="button"
      className="reader-section-dismiss"
      onClick={() => onDismiss(section)}
      aria-label={`Hide ${label}`}
      title={`Hide ${label}`}
    >
      ×
    </button>
  );
}

function renderResearchDetails(research: ResearchMeta) {
  return (
    <details className="reader-report-details">
      <summary>Show details</summary>
      <div className="reader-report-details-grid">
        {research.objective ? (
          <DetailRow label="Objective">
            <p>{research.objective}</p>
          </DetailRow>
        ) : null}
        {research.queries?.length ? (
          <DetailRow label="Queries">
            <ul>
              {Array.from(new Set(research.queries)).slice(0, 6).map((query) => (
                <li key={query}>{query}</li>
              ))}
            </ul>
          </DetailRow>
        ) : null}
        {research.provider_chain?.length ? (
          <DetailRow label="Provider path">
            <p>{research.provider_chain.join(' -> ')}</p>
          </DetailRow>
        ) : null}
        {research.planner_llm_provider ? (
          <DetailRow label="Planner">
            <p>{providerLabel(research.planner_llm_provider)}{modelSuffix(research.planner_llm_model)}</p>
          </DetailRow>
        ) : null}
        {research.synth_llm_provider ? (
          <DetailRow label="Synthesis">
            <p>{providerLabel(research.synth_llm_provider)}{modelSuffix(research.synth_llm_model)}</p>
          </DetailRow>
        ) : null}
      </div>
    </details>
  );
}

export function ResearchReportView({
  message,
  fallbackTitle = 'Research',
  options,
  tone = 'dark',
  onDismissSection,
}: ResearchReportViewProps) {
  const documentOptions: ResearchReaderOptions = { ...DEFAULT_OPTIONS, ...options };
  const report = deriveResearchReport(message, fallbackTitle);
  const research = message.research;

  if (!report) {
    return (
      <section
        className={[
          'research-reader',
          tone === 'paper' ? 'research-reader--paper' : '',
          documentOptions.compactTables ? 'research-reader--compact-tables' : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        <div className="reader-report reader-report--fallback">
          <div className="reader-report-body reader-report-prose">
            <div className="assistant-markdown">
              <ResearchMarkdown content={message.content || ''} />
            </div>
          </div>
        </div>
      </section>
    );
  }

  const noticeIsSourceFilter = isSourceFilterNotice(report.notice);
  const suppressHeroTitle = isGenericEvidenceTitle(report.title);
  const titleLength = report.title.trim().length;
  const titleClassName = [
    'reader-report-title',
    titleLength >= 72 ? 'reader-report-title--long' : '',
    titleLength >= 108 ? 'reader-report-title--xlong' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <section
      className={[
        'research-reader',
        tone === 'paper' ? 'research-reader--paper' : '',
        documentOptions.compactTables ? 'research-reader--compact-tables' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="reader-report">
        <header className="reader-report-header">
          <div className="reader-report-header-inner">
            {!suppressHeroTitle ? <h1 className={titleClassName}>{report.title}</h1> : null}
            {report.notice && !noticeIsSourceFilter ? (
              <p className="reader-report-notice">Note: {report.notice}</p>
            ) : null}
            {documentOptions.includeSummary && report.summary ? (
              <div className="reader-report-summary-card">
                <div className="reader-report-section-heading reader-report-section-heading--compact">
                  <div className="reader-report-summary-label">Executive summary</div>
                  <DismissButton label="summary" section="includeSummary" onDismiss={onDismissSection} />
                </div>
                <p className="reader-report-summary">{report.summary}</p>
              </div>
            ) : null}
          </div>
        </header>

        <section className="reader-report-main">
          {report.tableMarkdown ? (
            <section className="reader-report-compare">
              <div className="reader-report-compare-panel">
                <div className="reader-report-compare-heading">
                  <div className="reader-report-compare-kicker">Quick comparison</div>
                  <h2 className="reader-report-compare-title">At a glance</h2>
                </div>
                <div className="reader-report-body reader-report-prose reader-report-prose--compare">
                  <div className="assistant-markdown">
                    <ResearchMarkdown content={report.tableMarkdown} />
                  </div>
                </div>
              </div>
            </section>
          ) : null}

          {report.narrativeMarkdown ? (
            <section className="reader-report-narrative-wrap">
              <div className="reader-report-section-break" aria-hidden="true" />
              <div className="reader-report-narrative">
                <div className="reader-report-body reader-report-prose reader-report-prose--narrative">
                  <div className="assistant-markdown">
                    <ResearchMarkdown content={report.narrativeMarkdown} />
                  </div>
                </div>
              </div>
            </section>
          ) : !report.tableMarkdown ? (
            <div className="reader-report-body reader-report-prose">
              <div className="assistant-markdown">
                <ResearchMarkdown content={report.bodyMarkdown || message.content || ''} />
              </div>
            </div>
          ) : null}
        </section>

        {(documentOptions.includeProviderBreakdown && research?.by_provider?.length) ||
        (documentOptions.includeResearchDetails && research) ? (
          <div className="reader-report-bottom-grid">
            {documentOptions.includeProviderBreakdown && research?.by_provider?.length ? (
              <section className="reader-report-provider-card">
                <div className="reader-report-section-heading">
                  <h2 className="reader-report-section-title">Provider breakdown</h2>
                  <div className="reader-report-section-actions">
                    <span className="reader-report-section-meta">{research.by_provider.length}</span>
                    <DismissButton
                      label="provider breakdown"
                      section="includeProviderBreakdown"
                      onDismiss={onDismissSection}
                    />
                  </div>
                </div>
                <div className="reader-report-provider-list">
                  {research.by_provider.map((provider) => (
                    <div key={provider.provider} className="reader-report-provider-item">
                      <div className="reader-report-provider-name">{provider.provider}</div>
                      <div className="reader-report-provider-stats">
                        <span>{provider.result_count} results</span>
                        <span>{provider.tool_runs} runs</span>
                        {provider.unique_source_count ? <span>{provider.unique_source_count} sources</span> : null}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {documentOptions.includeResearchDetails && research ? (
              <section className="reader-report-details-card">
                <div className="reader-report-section-heading">
                  <h2 className="reader-report-section-title">Research details</h2>
                  <DismissButton
                    label="research details"
                    section="includeResearchDetails"
                    onDismiss={onDismissSection}
                  />
                </div>
                {renderResearchDetails(research)}
              </section>
            ) : null}
          </div>
        ) : null}

        {documentOptions.includeSources && report.sourceUrls.length > 0 ? (
          <section className="reader-report-sources reader-report-sources--full">
            <div className="reader-report-section-heading">
              <h2 className="reader-report-section-title">Sources</h2>
              <div className="reader-report-section-actions">
                <span className="reader-report-section-meta">{report.sourceUrls.length}</span>
                <DismissButton label="sources" section="includeSources" onDismiss={onDismissSection} />
              </div>
            </div>
            {noticeIsSourceFilter && report.notice ? (
              <p className="reader-report-source-note">{report.notice}</p>
            ) : null}
            <ul className="reader-report-source-list">
              {report.sourceItems.map((source) => (
                <li key={source.url} className="reader-report-source-item">
                  <a href={source.url} target="_blank" rel="noreferrer" className="reader-report-source-link" title={source.url}>
                    <span className="reader-report-source-domain">{source.label}</span>
                    <span className="reader-report-source-url">{source.displayUrl}</span>
                  </a>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </section>
  );
}
