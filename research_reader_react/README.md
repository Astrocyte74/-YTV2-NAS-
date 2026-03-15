# Research Reader React

Portable React renderer for the research markdown/report format used in TTS Hub.

This is the frontend half of the "secret sauce":

- markdown normalization for assistant output
- GFM table rendering
- report title / summary / sources extraction
- comparison-table presentation
- research details and provider breakdown cards

## Files

```text
portable/research_reader_react/
└── src/
    ├── index.ts
    ├── ResearchMarkdown.tsx
    ├── ResearchReportView.tsx
    ├── research-reader.css
    └── research-report.ts
```

## Dependencies

- `react`
- `react-dom`
- `react-markdown`
- `remark-gfm`

The portable version intentionally avoids project-specific aliases and avoids `lucide-react`.

## Usage

```tsx
import { ResearchReportView } from './research_reader_react/src';
import './research_reader_react/src/research-reader.css';

const message = {
  content: researchResponse.response,
  research: researchResponse.meta,
};

export function ResearchPane() {
  return (
    <ResearchReportView
      message={message}
      fallbackTitle="Deep Research"
      tone="paper"
      options={{
        compactTables: false,
        includeSummary: true,
        includeSources: true,
      }}
    />
  );
}
```

## Input shape

The component expects:

- `message.content`: the markdown answer text
- `message.research`: metadata object from the research backend

At minimum, useful fields in `message.research` are:

- `objective`
- `queries`
- `provider_chain`
- `by_provider`
- `planner_llm_provider`
- `planner_llm_model`
- `synth_llm_provider`
- `synth_llm_model`

## Notes

- If the content includes a leading title line, a summary paragraph, and a final `Sources` section, the renderer will split those into separate UI sections automatically.
- If the report starts with a markdown table, it is presented as the comparison hero section.
- If no research metadata exists, the component falls back to plain markdown rendering.
