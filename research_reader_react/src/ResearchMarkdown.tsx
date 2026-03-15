import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export interface ResearchMarkdownProps {
  content: string;
}

function normalizeMarkdownContent(content: string): string {
  return content
    .replace(/\r\n?/g, '\n')
    .replace(/<\/p>\s*<p>/gi, '\n\n')
    .replace(/<\/?p>/gi, '')
    .replace(/<hr\s*\/?>/gi, '\n\n---\n\n')
    .replace(/(?:<br\s*\/?>\s*){2,}/gi, '\n\n')
    .replace(/<br\s*\/?>/gi, '  \n')
    .replace(/&nbsp;/gi, ' ');
}

export function ResearchMarkdown({ content }: ResearchMarkdownProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <div className="assistant-h1">{children}</div>,
        h2: ({ children }) => <div className="assistant-h2">{children}</div>,
        h3: ({ children }) => <div className="assistant-h3">{children}</div>,
        p: ({ children }) => <p className="assistant-paragraph">{children}</p>,
        ul: ({ children }) => <ul className="assistant-list">{children}</ul>,
        ol: ({ children }) => <ol className="assistant-olist">{children}</ol>,
        hr: () => <hr className="assistant-divider" />,
        blockquote: ({ children }) => <blockquote className="assistant-blockquote">{children}</blockquote>,
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer" className="assistant-inline-link">
            {children}
          </a>
        ),
        pre: ({ children }) => <pre className="assistant-code-block">{children}</pre>,
        table: ({ children }) => (
          <div className="assistant-table-wrap">
            <table className="assistant-table">{children}</table>
          </div>
        ),
      }}
    >
      {normalizeMarkdownContent(content)}
    </ReactMarkdown>
  );
}
