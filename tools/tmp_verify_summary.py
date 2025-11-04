#!/usr/bin/env python3
import os
import asyncio
from youtube_summarizer import YouTubeSummarizer
from modules.summary_variants import format_summary_html

async def main():
    # Prefer local for this quick check
    os.environ.setdefault('LLM_PROVIDER', 'ollama')
    os.environ.setdefault('LLM_MODEL', 'gemma3:12b')
    y = YouTubeSummarizer(llm_provider=os.environ['LLM_PROVIDER'], model=os.environ['LLM_MODEL'])
    text = (
        "This video reviews super-shoe performance at Sydney Marathon 2025. "
        "It compares course characteristics and podium footwear choices, with concrete brand mentions and specific models. "
        "Alphafly 3 dominated, Vaporfly 4 reached the podium, and Asics MetaSpeed Sky Tokyo won the men's race. "
        "MetaSpeed Ray compliance begins on September 11, 2025. The course is scenic, twisty, hilly, and not considered fast."
    )
    meta = {"title":"Test Summary","uploader":"Tester","url":"http://example.com","duration":300,"language":"en"}
    res = await y.process_text_content(content_id='TEST_FORMAT_CHECK', text=text, metadata=meta, summary_type='bullet-points')
    summary_text = (res.get('summary') if isinstance(res, dict) else str(res)) or ''
    print('--- SUMMARY TEXT ---')
    print(summary_text)
    print('\n--- FORMATTED HTML ---')
    print(format_summary_html(summary_text))

if __name__ == '__main__':
    asyncio.run(main())

