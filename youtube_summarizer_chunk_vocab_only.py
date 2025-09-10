#!/usr/bin/env python3
"""
YouTube Video Summarizer using MCP-Use Library
Extracts transcripts from YouTube videos and generates intelligent summaries
"""

import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import yt_dlp

# Try to import youtube-transcript-api for better transcript access
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_API_AVAILABLE = True
    
    # Test the API to ensure correct version/method exists
    import inspect
    api_instance = YouTubeTranscriptApi()
    if not hasattr(api_instance, 'fetch'):
        print("⚠️  youtube-transcript-api version incompatible, will use yt-dlp only")
        TRANSCRIPT_API_AVAILABLE = False
    else:
        print("✅ youtube-transcript-api loaded successfully")
        
except ImportError:
    print("⚠️  youtube-transcript-api not available, will use yt-dlp only")
    TRANSCRIPT_API_AVAILABLE = False
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import requests

# Import LLM configuration manager
from llm_config import llm_config

class YouTubeSummarizer:
    def __init__(self, llm_provider: str = None, model: str = None, ollama_base_url: str = None):
        """Initialize the YouTube Summarizer with mkpy LLM integration
        
        Args:
            llm_provider: Optional override for LLM provider. If None, uses mkpy configuration
            model: Optional override for model. If None, uses mkpy configuration  
            ollama_base_url: Base URL for Ollama server (default from config)
        """
        self.downloads_dir = Path("./downloads")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Get LLM configuration from mkpy system
        try:
            self.llm_provider, self.model, api_key = llm_config.get_model_config(llm_provider, model)
        except ValueError as e:
            print(f"🔴 {e}")
            raise
        
        # Set Ollama base URL
        self.ollama_base_url = ollama_base_url or llm_config.ollama_host
        
        # Initialize LLM based on determined configuration
        self._initialize_llm(api_key)
    
    def _initialize_llm(self, api_key: str):
        """Initialize the LLM based on provider and model"""
        
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key
            )
        elif self.llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model,
                api_key=api_key
            )
        elif self.llm_provider == "openrouter":
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/astrocyte74/stuff",
                    "X-Title": "YouTube Summarizer"
                }
            )
        elif self.llm_provider == "ollama":
            self.llm = ChatOllama(
                model=self.model,
                base_url=self.ollama_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _get_fallback_metadata(self, youtube_url: str, video_id: str) -> dict:
        """Get basic metadata from YouTube page when yt-dlp fails"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Get basic page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(youtube_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title from page title or meta tags
                title = 'Unknown Title'
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().replace(' - YouTube', '').strip()
                
                # Try to get channel name from meta tags
                uploader = 'Unknown'
                channel_meta = soup.find('meta', {'name': 'author'})
                if channel_meta:
                    uploader = channel_meta.get('content', 'Unknown')
                
                return {
                    'title': title,
                    'description': '',
                    'uploader': uploader,
                    'upload_date': '',
                    'duration': 0,
                    'duration_string': 'Unknown',
                    'view_count': 0,
                    'url': youtube_url,
                    'video_id': video_id,
                    'id': video_id,
                    'channel_url': '',
                    'tags': [],
                    'thumbnail': '',
                }
        except Exception as e:
            print(f"⚠️ Fallback metadata extraction failed: {str(e)}")
        
        # Final fallback - basic info only
        return {
            'title': f'Video {video_id}',
            'description': '',
            'uploader': 'Unknown',
            'upload_date': '',
            'duration': 0,
            'duration_string': 'Unknown', 
            'view_count': 0,
            'url': youtube_url,
            'video_id': video_id,
            'id': video_id,
            'channel_url': '',
            'tags': [],
            'thumbnail': '',
        }

    def _extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'watch?v=' in youtube_url:
            return youtube_url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            return youtube_url.split('youtu.be/')[1].split('?')[0]
        elif '/embed/' in youtube_url:
            return youtube_url.split('/embed/')[1].split('?')[0]
        else:
            # Assume it's already a video ID
            return youtube_url
    
    def _get_transcript_and_metadata_via_api(self, video_id: str, youtube_url: str) -> dict:
        """Extract transcript and attempt basic metadata using youtube-transcript-api + web scraping"""
        transcript_text = None
        
        # Get transcript via youtube-transcript-api with multiple language support
        if TRANSCRIPT_API_AVAILABLE:
            try:
                api = YouTubeTranscriptApi()
                
                # Try multiple language codes in order of preference
                language_codes = [
                    'en',           # English (US)
                    'en-GB',        # English (UK)
                    'en-US',        # English (US explicit)
                    'en-AU',        # English (Australia)
                    'en-CA',        # English (Canada)
                    'es',           # Spanish
                    'es-ES',        # Spanish (Spain)
                    'es-MX',        # Spanish (Mexico)
                    'fr',           # French
                    'fr-FR',        # French (France)
                    'fr-CA'         # French (Canada)
                ]
                
                transcript_data = None
                used_language = None
                
                for lang_code in language_codes:
                    try:
                        transcript_data = api.fetch(video_id, [lang_code])
                        used_language = lang_code
                        break
                    except Exception:
                        continue
                
                if transcript_data:
                    text_parts = [snippet.text for snippet in transcript_data]
                    transcript_text = ' '.join(text_parts)
                    print(f"✅ YouTube Transcript API: Extracted {len(transcript_text)} characters in {used_language}")
                else:
                    print(f"⚠️ YouTube Transcript API: No supported language found")
                    
            except Exception as e:
                print(f"⚠️ YouTube Transcript API failed: {e}")
        
        # Always try to get metadata from yt-dlp (more reliable than web scraping)
        try:
            ydl_opts = {
                'skip_download': True,
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', 'Unknown'), 
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration', 0),
                    'duration_string': info.get('duration_string', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'url': youtube_url,
                    'video_id': video_id,
                    'id': info.get('id', video_id),
                    'channel_url': info.get('channel_url', ''),
                    'tags': info.get('tags', []),
                    'thumbnail': info.get('thumbnail', ''),
                }
        except Exception as e:
            print(f"⚠️ yt-dlp metadata extraction failed: {e}")
            # Fallback to web scraping
            metadata = self._get_fallback_metadata(youtube_url, video_id)
        
        return {
            'transcript': transcript_text,
            'metadata': metadata
        }
    
    def _try_yt_dlp_transcript_extraction(self, info: dict) -> Optional[str]:
        """Try to extract transcript using yt-dlp as fallback method"""
        transcript_text = ""
        
        try:
            # Check for automatic captions first - try multiple English variants
            english_keys = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            auto_caps = None
            
            if 'automatic_captions' in info:
                for lang_key in english_keys:
                    if lang_key in info['automatic_captions']:
                        auto_caps = info['automatic_captions'][lang_key]
                        print(f"🔍 yt-dlp fallback: Found {len(auto_caps)} automatic caption formats in {lang_key}")
                        break
            
            if auto_caps:
                for cap in auto_caps:
                    if cap.get('ext') in ['srv3', 'json3', 'ttml', 'vtt']:
                        transcript_url = cap['url']
                        try:
                            import urllib.request
                            with urllib.request.urlopen(transcript_url) as response:
                                transcript_data = response.read().decode('utf-8')
                                if cap.get('ext') == 'srv3':
                                    transcript_text = self._parse_srv3_transcript(transcript_data)
                                else:
                                    transcript_text = self._parse_generic_transcript(transcript_data)
                                
                                if transcript_text.strip():
                                    print(f"✅ yt-dlp fallback: Extracted via {cap.get('ext')} format")
                                    return transcript_text
                        except Exception as e:
                            print(f"⚠️  yt-dlp fallback error: {e}")
            
            # Try manual subtitles if auto captions failed
            if 'subtitles' in info and 'en' in info['subtitles']:
                manual_subs = info['subtitles']['en']
                print(f"🔍 yt-dlp fallback: Trying {len(manual_subs)} manual subtitle formats")
                for sub in manual_subs:
                    if sub.get('ext') in ['srv3', 'json3', 'ttml', 'vtt']:
                        try:
                            import urllib.request
                            with urllib.request.urlopen(sub['url']) as response:
                                transcript_data = response.read().decode('utf-8')
                                if sub.get('ext') == 'srv3':
                                    transcript_text = self._parse_srv3_transcript(transcript_data)
                                else:
                                    transcript_text = self._parse_generic_transcript(transcript_data)
                                
                                if transcript_text.strip():
                                    print(f"✅ yt-dlp fallback: Extracted via manual {sub.get('ext')} format")
                                    return transcript_text
                        except Exception as e:
                            print(f"⚠️  yt-dlp fallback error: {e}")
            
            return transcript_text if transcript_text.strip() else None
            
        except Exception as e:
            print(f"⚠️  yt-dlp transcript extraction failed: {e}")
            return None
    
    def extract_transcript(self, youtube_url: str) -> Dict[str, Union[str, List[Dict]]]:
        """Extract transcript and metadata from YouTube video using hybrid approach
        
        Uses youtube-transcript-api for transcripts (primary) and yt-dlp for metadata.
        Falls back to yt-dlp transcript extraction if primary fails.
        
        Args:
            youtube_url: URL of the YouTube video
            
        Returns:
            Dictionary containing video metadata and transcript
        """
        # Initialize variables in outer scope
        transcript_text = None
        video_id = self._extract_video_id(youtube_url)
        
        # STEP 1: Try simplified approach - transcript API + web scraping for metadata
        result = self._get_transcript_and_metadata_via_api(video_id, youtube_url)
        transcript_text = result['transcript']
        metadata = result['metadata']
        
        # If we got a good transcript, return immediately with web-scraped metadata
        if transcript_text and len(transcript_text.strip()) > 100:
            return {
                'metadata': metadata,
                'transcript': transcript_text,
                'content_type': 'transcript',
                'success': True
            }
        
        # STEP 2: Fallback to yt-dlp if youtube-transcript-api failed
        try:
            print("🔄 Falling back to yt-dlp for both transcript and metadata...")
            ydl_opts = {
                'skip_download': True,
                'writeinfojson': True,
                'writeautomaticsub': True,
                'writesubtitles': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
                'subtitlesformat': 'best',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],  # Multiple clients for robustness
                        'skip': ['hls', 'dash'],  # Skip unnecessary formats for speed
                    }
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                'sleep_interval': 0,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info
                info = ydl.extract_info(youtube_url, download=False)
                
                # Get basic metadata
                metadata = {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration', 0),
                    'duration_string': info.get('duration_string', ''),
                    'view_count': info.get('view_count', 0),
                    'url': youtube_url,
                    'video_id': info.get('id', ''),
                    'id': info.get('id', ''),
                    'channel_url': info.get('channel_url', ''),
                    'tags': info.get('tags', []),
                    'thumbnail': info.get('thumbnail', ''),
                }
                
                # STEP 3: Handle transcript - use primary result or fallback to yt-dlp
                content_type = "transcript"  # Default assumption
                
                if transcript_text and len(transcript_text.strip()) > 100:
                    # SUCCESS: We have a good transcript from youtube-transcript-api
                    print(f"✅ Using YouTube Transcript API result: {len(transcript_text)} characters")
                else:
                    # FALLBACK: Try yt-dlp transcript extraction
                    print("🔄 Falling back to yt-dlp transcript extraction...")
                    yt_dlp_transcript = self._try_yt_dlp_transcript_extraction(info)
                    
                    if yt_dlp_transcript and len(yt_dlp_transcript.strip()) > 100:
                        transcript_text = yt_dlp_transcript
                        print(f"✅ yt-dlp transcript fallback successful: {len(transcript_text)} characters")
                    elif yt_dlp_transcript:
                        transcript_text = yt_dlp_transcript
                        print(f"⚠️  yt-dlp transcript fallback partial: {len(transcript_text)} characters")
                    else:
                        # REJECT: Description-only videos to prevent hallucination
                        description = info.get('description', '')
                        print(f"❌ No transcript available for this video")
                        print(f"❌ Only description available ({len(description)} chars) - rejecting to prevent AI hallucination")
                        return {
                            'error': 'No transcript available - only description found. Rejecting to prevent hallucination.',
                            'success': False,
                            'content_type': 'description_only'
                        }
                
                return {
                    'metadata': metadata,
                    'transcript': transcript_text,
                    'content_type': content_type,
                    'success': True
                }
                
        except Exception as e:
            # If yt-dlp fails but we have a transcript from the API, create minimal metadata
            if transcript_text and len(transcript_text.strip()) > 100:
                # Silent fallback - we have transcript, metadata is optional
                video_id = self._extract_video_id(youtube_url)
                
                # Try to get basic metadata from YouTube page as fallback
                print("🔄 Attempting fallback metadata extraction...")
                fallback_metadata = self._get_fallback_metadata(youtube_url, video_id)
                
                return {
                    'metadata': fallback_metadata,
                    'transcript': transcript_text,
                    'content_type': 'transcript',
                    'success': True
                }
            else:
                return {
                    'error': str(e),
                    'success': False
                }
    
    def _parse_srv3_transcript(self, srv3_data: str) -> str:
        """Parse SRV3 format transcript data"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(srv3_data)
            
            transcript_lines = []
            for text_elem in root.findall('.//text'):
                text_content = text_elem.text or ""
                if text_content.strip():
                    # Clean up the text
                    text_content = re.sub(r'&[a-zA-Z]+;', '', text_content)
                    text_content = text_content.strip()
                    if text_content:
                        transcript_lines.append(text_content)
            
            return ' '.join(transcript_lines)
        except Exception as e:
            print(f"Error parsing SRV3 transcript: {e}")
            return ""
    
    def _parse_generic_transcript(self, transcript_data: str) -> str:
        """Parse generic transcript formats (JSON3, VTT, TTML)"""
        try:
            transcript_lines = []
            
            # Try JSON format first
            if transcript_data.strip().startswith('{') or transcript_data.strip().startswith('['):
                import json
                try:
                    data = json.loads(transcript_data)
                    if isinstance(data, dict) and 'events' in data:
                        # YouTube JSON3 format
                        for event in data['events']:
                            if 'segs' in event:
                                for seg in event['segs']:
                                    if 'utf8' in seg:
                                        text = seg['utf8'].strip()
                                        if text and text not in ['♪', '[Music]', '[Applause]']:
                                            transcript_lines.append(text)
                    elif isinstance(data, list):
                        # Other JSON formats
                        for item in data:
                            if isinstance(item, dict) and 'text' in item:
                                text = item['text'].strip()
                                if text:
                                    transcript_lines.append(text)
                except json.JSONDecodeError:
                    pass
            
            # Try VTT format
            elif 'WEBVTT' in transcript_data or '-->' in transcript_data:
                lines = transcript_data.split('\n')
                for line in lines:
                    line = line.strip()
                    # Skip timestamps and empty lines
                    if (line and not line.startswith('WEBVTT') and 
                        '-->' not in line and not line.isdigit() and
                        not line.startswith('NOTE') and not line.startswith('STYLE')):
                        # Clean HTML tags if present
                        import re
                        clean_line = re.sub(r'<[^>]+>', '', line)
                        if clean_line.strip():
                            transcript_lines.append(clean_line.strip())
            
            # Try XML/TTML format
            elif '<' in transcript_data and '>' in transcript_data:
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(transcript_data)
                    # Find all text elements regardless of namespace
                    for elem in root.iter():
                        if elem.text and elem.text.strip():
                            text = elem.text.strip()
                            if text not in ['♪', '[Music]', '[Applause]']:
                                transcript_lines.append(text)
                except ET.ParseError:
                    pass
            
            result = ' '.join(transcript_lines)
            return result if result.strip() else ""
            
        except Exception as e:
            print(f"Error parsing generic transcript: {e}")
            return ""
    
    async def generate_summary(self, transcript: str, metadata: Dict, 
                             summary_type: str = "comprehensive", proficiency_level: str = None) -> Dict[str, str]:
        """Generate AI summary of the video transcript with chunking for long content
        
        Args:
            transcript: The video transcript text
            metadata: Video metadata dictionary
            summary_type: Type of summary ('comprehensive', 'audio', 'bullet-points', 'key-insights', 'executive', 'adaptive')
            
        Returns:
            Dictionary containing different summary formats
        """
        
        # Check if transcript needs chunking (>12000 chars to leave room for prompt overhead)
        if len(transcript) > 12000:
            print(f"📄 Long transcript detected ({len(transcript):,} chars) - using chunked processing for full content")
            return await self._generate_chunked_summary(transcript, metadata, summary_type, proficiency_level)
        
        # For shorter transcripts, use direct processing
        print(f"📄 Processing transcript directly ({len(transcript):,} chars)")
        
        # Prepare context and prompts based on summary type
        base_context = f"""
        Video Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('uploader', 'Unknown')}
        Upload Date (YYYYMMDD): {metadata.get('upload_date', 'Unknown')}
        Duration (seconds): {metadata.get('duration', 0)}
        URL: {metadata.get('url', 'Unknown')}

        Full Transcript:
        {transcript}
        """
        
        prompts = {
            "comprehensive": f"""
            {base_context}

            Create an **executive report** that extracts key insights and strategic implications from this video content.

            **EXECUTIVE SUMMARY**
            Write 2–3 concise sentences capturing the video's main purpose, key findings, and strategic takeaway. Frame the final sentence as a "so what" for leadership decision-making. Keep it sharp and high-level — no more than 3 sentences.

            **KEY INSIGHTS & ANALYSIS**
            Organise the content into 2–4 natural sections with **short, descriptive headings** (2–5 words maximum). For each section:
            - Begin with 1–2 sentences establishing the theme and relevance (vary: "Focus", "Scope", "Context", etc.)
            - Present 3–5 key findings as **uniform, verb-led bullets** (e.g., "Verify version (0.88+) with claude –version", "Switch providers via base URL", "Enable cost tracking...")
            - Conclude with strategic implications woven naturally into prose (avoid labelling "Implications:")

            **STRATEGIC RECOMMENDATIONS**
            Provide 3–4 clear, actionable recommendations in **tiered bullets** (headline + supporting details):  
            - Top-level bullet = decisive headline (*Institutionalise cost governance*, *Adopt hybrid workflows*)  
                • Supporting bullets = 1–2 specific implementation details  

            **Writing Guidelines:**
            - Assume audience is senior leadership — write in crisp, consulting-style prose (think McKinsey/BCG)
            - Emphasise strategic implications over raw facts
            - Use professional language (British/Canadian spelling)
            - Include timestamps only when they add clarity
            - Vary section openings naturally (avoid mechanical repetition)
            - Create sections that emerge organically from content
            - Never speculate beyond presented material
            - Ensure natural flow — reports should feel executive-authored, not AI-generated
            - **IMPORTANT: Respond in the same language as the original transcript.** Preserve the original language for authentic communication.
            """,

            "audio": f"""
            {base_context}

            You are writing a script for text-to-speech audio. Create a conversational, flowing summary that sounds natural when spoken aloud. Present the actual findings and insights directly - never describe what the video does or what the reviewer says.

            **Script Structure:**

            **Opening:** Start with 2-3 sentences that immediately introduce the main topic and key conclusion. No "this video covers" - jump straight into the substance.

            **Main Content:** Analyze the transcript to identify major topics, then present them using smooth conversational transitions between sections. Never use headings or bullet points. Instead, use natural verbal signposts like:
            - "First, let's look at the performance results..."
            - "Moving on to build quality..."
            - "When it comes to pricing..."
            - "The most important finding was..."
            - "However, there's an important trade-off..."

            **Conclusion:** End with "So what's the bottom line?" or "In the end..." followed by the main takeaway and practical recommendations. **Crucially, if the video is a review, ensure this final verdict explicitly mentions price, value, and the final cost-benefit analysis, as this is often the most important takeaway for the audience.**

            **Critical Writing Rules:**
            - Write in flowing paragraphs only - no lists, bullets, or visual formatting
            - **Keep paragraphs focused and concise:** To maintain listener engagement, ensure each paragraph covers a single main idea. If a topic is complex, break it into multiple shorter paragraphs rather than one long one
            - Use short, clear sentences that are easy to follow when spoken
            - Spell out acronyms the first time ("Artificial Intelligence, or A-I")
            - Include specific details, numbers, model names, and prices
            - Use natural pauses with commas and periods to guide TTS pacing
            - Cover all major topics - don't skip important sections like pricing, specifications, or final recommendations
            - Scale content to match complexity - comprehensive coverage for detailed content, concise for simple topics

            **Avoid Meta-Commentary:**
            Present findings directly and avoid phrases like "the video explains" or "the reviewer discusses." Only use attribution when it's essential to clarify that a statement is a subjective opinion or conclusion, not an objective fact.
            - GOOD (Direct Fact): "The vacuum has twenty-two thousand Pascals of suction."
            - GOOD (Attributed Opinion): "The reviewer concludes the mopping is a generation ahead of the competition."
            - BAD (Weak Phrasing): "The reviewer then goes on to talk about the suction power."

            Write as if you're explaining the findings to a friend over coffee - professional but conversational, comprehensive but natural to listen to.
            
            **IMPORTANT: Respond in the same language as the original transcript.** If the transcript is in French, respond in French. If it's in Spanish, respond in Spanish, etc. This ensures the TTS audio matches the original content language.
            """,

            "bullet-points": f"""
            {base_context}

            Output ONLY this Markdown skeleton, filling in the brackets. Keep each bullet ≤ 18 words.

            • **Main topic:** [brief]
            • **Key points:**
              - [point 1]
              - [point 2]
              - [point 3]
            • **Takeaway:** [single sentence]
            • **Best for:** [audience]
            
            **IMPORTANT: Respond in the same language as the original transcript.**
            """,

            "key-insights": f"""
            {base_context}

            Extract actionable insights only. Provide numbered Markdown items, each ≤ 24 words, no fluff:
            1) **Core message:** [one sentence]
            2) **Top 3 insights:** [three sub‑bullets]
            3) **Actions/next steps:** [2–4 sub‑bullets]
            4) **Why watch:** [one sentence]

            Do not speculate; write **Unknown** if the transcript doesn't support a claim.
            **IMPORTANT: Respond in the same language as the original transcript.**
            """,
            
            "executive": f"""
            {base_context}

            Format this as a structured EXECUTIVE REPORT. Analyze the content and divide it into 2-4 logical parts based on the video's natural flow.

            **📊 EXECUTIVE SUMMARY**
            2-3 sentences that capture the video's main purpose, key findings, and business/practical value. Professional tone.

            **📋 PART 1: (Give this section a descriptive title)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details  
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **📋 PART 2: (Give this section a descriptive title)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **📋 PART 3: (Give this section a descriptive title - if applicable)**
            
            **Overview:** 2-3 sentences explaining what this part covers and its relevance.
            
            **Key Points:**
            • Specific finding/point with details
            • Specific finding/point with details
            • Specific finding/point with details
            
            **Conclusion:** 1-2 sentences summarizing the implications of this section.

            **🎯 STRATEGIC RECOMMENDATIONS**
            • Actionable next step or key takeaway
            • Actionable next step or key takeaway
            • Actionable next step or key takeaway

            **Guidelines:**
            - Divide content into 2-4 logical parts (not artificial divisions)
            - Use professional, analytical language
            - Include timestamps where helpful
            - Focus on insights and implications, not just facts
            - Keep each section balanced and substantive
            - If content doesn't fit this structure well, adapt the format accordingly
            - British/Canadian spelling; no speculation
            """,
            
            "adaptive": f"""
            {base_context}

            You are an expert content analyst. Based on the transcript, silently choose the most effective summary format for this specific video and produce ONLY the summary.

            **Consider (internally, without writing it out):**
            - Content complexity and technical depth
            - Educational vs entertainment value
            - Tutorial/procedural vs discussion/opinion
            - Information density and duration
            - Likely audience sophistication implied by the transcript

            **Available formats (choose one, or design a simple custom layout if clearly better):**
            1) Comprehensive detailed analysis
            2) Executive brief
            3) Key insights extraction
            4) Step-by-step breakdown (for tutorials/procedures)
            5) Custom adaptive format (only if it's clearly superior)

            **Output rules (must follow):**
            - Do NOT explain your choice or approach; no meta-commentary
            - Start directly with the summary content (no prefaces)
            - Use clear, scannable structure (short headings and/or bullets as appropriate)
            - British/Canadian spelling; neutral, precise tone by default
            - Include [mm:ss] timestamps ONLY if they are explicitly present or derivable from the transcript; otherwise omit them
            - No speculation or external facts; if a needed detail is missing, write **Unknown**
            - Avoid filler signposting ("In this section…", "We will discuss…"); write the content itself
            - If the transcript is sparse/inaudible/mostly music, summarise the **available** information and state **Unknown** for specifics
            - End with a single concise takeaway sentence beginning **Bottom line:** unless the chosen format already has a clear conclusion

            **Length guidance (not a hard limit):**
            - Short videos / light content: 120–180 words
            - Dense or technical content: 250–500 words
            - Tutorials: use a step-wise structure with concise steps (≤ 12 words per step)
            """,
            
            "audio-fr": f"""
            {base_context}

            Create a French audio summary optimized for text-to-speech. Translate the content to fluent, natural French and present it as a conversational script.

            **Requirements:**
            - Translate to fluent French with natural expressions and idioms
            - Use conversational tone suitable for listening
            - Remove formatting markers and bullet points
            - Use smooth transitions between topics ("Commençons par...", "Ensuite...", "En conclusion...")
            - Maintain all key information and technical accuracy
            - Structure for easy listening comprehension in French
            - Keep engaging and informative
            - Preserve numbers, names, and technical terms appropriately
            - Use native French expressions where appropriate
            
            **Output ONLY the French audio-ready text.** No explanations or meta-commentary.
            """,
            
            "audio-es": f"""
            {base_context}

            Create a Spanish audio summary optimized for text-to-speech. Translate the content to fluent, natural Spanish and present it as a conversational script.

            **Requirements:**
            - Translate to fluent Spanish with natural expressions and idioms
            - Use conversational tone suitable for listening
            - Remove formatting markers and bullet points  
            - Use smooth transitions between topics ("Empezemos con...", "Luego...", "En conclusión...")
            - Maintain all key information and technical accuracy
            - Structure for easy listening comprehension in Spanish
            - Keep engaging and informative
            - Preserve numbers, names, and technical terms appropriately
            - Use native Spanish expressions where appropriate
            
            **Output ONLY the Spanish audio-ready text.** No explanations or meta-commentary.
            """,
        }
        
        # Generate the summary with robust error handling
        summary_text = await self._robust_llm_call(
            [HumanMessage(content=prompts.get(summary_type, prompts["comprehensive"]))],
            f"Summary generation ({summary_type})"
        )
        
        if not summary_text:
            return {
                'error': "Failed to generate summary after all retries",
                'summary': "Summary generation failed",
                'headline': "Error generating headline"
            }
        
        # Also generate a quick title/headline
        title_prompt = f"""
        Write a single, specific headline (12–16 words, no emojis) that states subject and concrete value.
        **IMPORTANT: Respond in the same language as the transcript.**
        Source title: {metadata.get('title', '')}
        Transcript excerpt:
        {transcript[:1200]}
        """
        
        headline_text = await self._robust_llm_call(
            [HumanMessage(content=title_prompt)],
            "Headline generation"
        )
        
        return {
            'summary': summary_text,
            'headline': headline_text or "Generated Summary",  # Fallback if headline fails
            'summary_type': summary_type,
            'generated_at': datetime.now().isoformat()
        }
    
    async def analyze_content(self, transcript: str, metadata: Dict) -> Dict[str, Union[str, List[str]]]:
        """Perform content analysis including topic categorization and sentiment"""
        
        analysis_prompt = f"""
        Analyze this YouTube video content and provide:
        
        Video: {metadata.get('title', 'Unknown')}
        Transcript: {transcript[:4000]}
        
        Please provide:
        1. Content Category (choose 1-2): Education, Entertainment, Technology, Business, Health, DIY, News, Gaming, Lifestyle, Science, etc.
        2. Sentiment (Overall tone): Positive, Negative, Neutral, Mixed
        3. Target Audience: Who would benefit most from this content?
        4. Complexity Level: Beginner, Intermediate, Advanced
        5. Key Topics (3-5 tags): Main subjects discussed
        6. Content Type: Tutorial, Review, Discussion, News, Documentary, etc.
        
        Format as JSON:
        {{
            "category": ["Primary Category", "Secondary Category"],
            "sentiment": "sentiment",
            "target_audience": "description",
            "complexity_level": "level",
            "key_topics": ["topic1", "topic2", "topic3"],
            "content_type": "type",
            "educational_value": "High/Medium/Low",
            "entertainment_value": "High/Medium/Low"
        }}
        """
        
        try:
            analysis_content = await self._robust_llm_call(
                [HumanMessage(content=analysis_prompt)],
                "Content analysis"
            )
            
            if not analysis_content:
                return {
                    "category": ["General"],
                    "sentiment": "Neutral",
                    "target_audience": "General audience",
                    "complexity_level": "Intermediate",
                    "key_topics": ["General content"],
                    "content_type": "Discussion",
                    "educational_value": "Medium",
                    "entertainment_value": "Medium"
                }
            
            # Try to parse JSON response
            import json
            try:
                analysis = json.loads(analysis_content.strip())
            except:
                # Fallback if JSON parsing fails
                analysis = {
                    "category": ["General"],
                    "sentiment": "Neutral", 
                    "target_audience": "General audience",
                    "complexity_level": "Intermediate",
                    "key_topics": ["General content"],
                    "content_type": "Discussion",
                    "educational_value": "Medium",
                    "entertainment_value": "Medium"
                }
            
            return analysis
            
        except Exception as e:
            print(f"Content analysis error: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "category": ["Unknown"], 
                "sentiment": "Unknown"
            }
    
    async def process_video(self, youtube_url: str, summary_type: str = "comprehensive", proficiency_level: str = None) -> Dict:
        """Complete processing pipeline for a YouTube video
        
        Args:
            youtube_url: YouTube video URL
            summary_type: Type of summary to generate
            
        Returns:
            Complete analysis dictionary
        """
        print(f"Processing video: {youtube_url}")
        
        # Step 1: Extract transcript
        print("Extracting transcript...")
        transcript_data = self.extract_transcript(youtube_url)
        
        if not transcript_data.get('success'):
            return {
                'error': transcript_data.get('error'),
                'url': youtube_url,
                'processed_at': datetime.now().isoformat()
            }
        
        metadata = transcript_data['metadata']
        transcript = transcript_data['transcript']
        content_type = transcript_data.get('content_type', 'transcript')
        
        # REJECT only videos with truly no content (very rare now with hybrid system)
        if content_type == 'none' or len(transcript.strip()) < 50:
            print("🚫 REJECTED: Cannot generate summary - no usable content available")
            return {
                'error': 'No usable content available for this video. Neither transcript nor description could be extracted. This may be a private video, age-restricted content, or have other access restrictions.',
                'url': youtube_url,
                'metadata': metadata,
                'transcript_available': False,
                'processed_at': datetime.now().isoformat(),
                'suggestion': "Try a different video or check if this video is publicly accessible."
            }
        elif content_type == 'description':
            # Allow description-only but warn the user
            print("⚠️  WARNING: Only video description available - summary will be limited")
            # Continue processing but mark as limited accuracy
        
        if len(transcript.strip()) < 50:
            print(f"⚠️  Transcript extracted: {len(transcript)} characters (insufficient for processing)")
        else:
            print(f"✅ Transcript extracted: {len(transcript)} characters")
        
        # Check if transcript is too short to be useful
        if len(transcript.strip()) < 50:  # Less than 50 characters is likely unusable
            video_title = metadata.get('title', 'Unknown video')
            return {
                'error': f"❌ No usable transcript found for '{video_title}'. This video appears to have no captions or subtitles available. Please try a different video that includes captions/subtitles for best results.",
                'url': youtube_url,
                'metadata': metadata,
                'transcript_length': len(transcript),
                'processed_at': datetime.now().isoformat(),
                'suggestion': "Look for videos with the [CC] closed captions icon on YouTube, or videos from channels that typically include subtitles."
            }
        
        # Step 2: Generate summary
        print("Generating summary...")
        summary_data = await self.generate_summary(transcript, metadata, summary_type, proficiency_level)
        
        # Step 3: Analyze content
        print("Analyzing content...")
        analysis_data = await self.analyze_content(transcript, metadata)
        
        # Combine all results
        result = {
            'url': youtube_url,
            'metadata': metadata,
            'transcript': transcript,
            'summary': summary_data,
            'analysis': analysis_data,
            'processed_at': datetime.now().isoformat(),
            'processor_info': {
                'llm_provider': self.llm_provider,
                'model': getattr(self.llm, 'model_name', getattr(self.llm, 'model', self.model))
            }
        }
        
        print("Processing complete!")
        return result

    async def _generate_chunked_summary(self, transcript: str, metadata: Dict, summary_type: str, proficiency_level: str = None) -> Dict[str, str]:
        """Generate summary by processing transcript in chunks and combining results"""
        
        # Normalize summary_type to canonical keys for robust matching
        TYPE_MAP = {
            "comprehensive": "comprehensive",
            "key points": "key_points", 
            "key_points": "key_points",
            "bullet points": "key_points",
            "bullet-points": "key_points", 
            "insights": "insights",
            "key-insights": "insights",
            "key_insights": "insights",
            "audio summary": "audio",
            "audio-summary": "audio", 
            "audio": "audio",
            "audio-fr": "audio_fr",
            "audio-es": "audio_es",
        }
        normalized_type = TYPE_MAP.get(summary_type.lower().strip(), summary_type)
        
        # Split transcript into manageable chunks (aim for ~8000 chars per chunk with overlap)
        chunk_size = 8000
        overlap = 500  # Overlap between chunks to maintain context
        
        chunks = []
        start = 0
        while start < len(transcript):
            end = start + chunk_size
            if end < len(transcript):
                # Find a good breaking point (sentence end) within the overlap zone
                overlap_start = max(start, end - overlap)
                break_point = transcript.rfind('. ', overlap_start, end)
                if break_point > overlap_start:
                    end = break_point + 1
            
            chunk = transcript[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        print(f"🔄 Processing transcript in {len(chunks)} chunks...")
        
        # Process each chunk and collect summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"📝 Processing chunk {i+1}/{len(chunks)} ({len(chunk):,} chars)")
            
            # Create context for this chunk
            chunk_context = f"""
            Video Title: {metadata.get('title', 'Unknown')}
            Channel: {metadata.get('uploader', 'Unknown')}
            Upload Date (YYYYMMDD): {metadata.get('upload_date', 'Unknown')}
            Duration (seconds): {metadata.get('duration', 0)}
            URL: {metadata.get('url', 'Unknown')}
            
            Transcript Segment {i+1}/{len(chunks)}:
            {chunk}
            """
            
            # Generate focused summary for this chunk
            chunk_prompt = f"""
            {chunk_context}
            
            Create a comprehensive summary of this transcript segment. Focus on:
            - Key points and main ideas discussed
            - Important details and insights
            - Context and connections between ideas
            - Actionable information or conclusions
            
            This is part {i+1} of {len(chunks)} segments from a longer video. Provide a complete summary of this section without referencing other parts.
            
            Format your response as a well-structured summary with clear sections and bullet points where appropriate.
            """
            
            chunk_summary = await self._robust_llm_call([HumanMessage(content=chunk_prompt)], 
                                                      operation_name="chunk summary generation")
            if chunk_summary:
                chunk_summaries.append(f"**Section {i+1}:**\n{chunk_summary}")
            else:
                chunk_summaries.append(f"**Section {i+1}:** [Processing failed for this section]")
        
        # Combine all chunk summaries into final summary
        print(f"🔄 Combining {len(chunk_summaries)} section summaries...")
        combined_summary = await self._combine_chunk_summaries(chunk_summaries, metadata, summary_type)
        
        # Only generate the requested summary type to save processing time and API costs
        result = {"comprehensive": combined_summary}
        
        if normalized_type == "key_points":
            result["bullet_points"] = await self._extract_bullet_points(combined_summary)
        elif normalized_type == "insights": 
            result["key_insights"] = await self._extract_key_insights(combined_summary)
        elif normalized_type == "audio":
            result["audio"] = await self._create_audio_summary(combined_summary)
        elif normalized_type == "audio_fr":
            result["audio"] = await self._create_translated_audio_summary(combined_summary, "French", proficiency_level)
        elif normalized_type == "audio_es":
            result["audio"] = await self._create_translated_audio_summary(combined_summary, "Spanish", proficiency_level)
        # For "comprehensive" type, we already have the comprehensive summary
        
        return result

    async def _combine_chunk_summaries(self, chunk_summaries: List[str], metadata: Dict, summary_type: str) -> str:
        """Combine individual chunk summaries into a cohesive final summary"""
        
        combined_content = "\n\n".join(chunk_summaries)
        
        combine_prompt = f"""
        You have been provided with {len(chunk_summaries)} section summaries from a video transcript. 
        Your task is to combine these into one comprehensive, well-structured summary.
        
        Video Context:
        Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('uploader', 'Unknown')}
        Duration: {metadata.get('duration', 0)} seconds
        
        Section Summaries:
        {combined_content}
        
        Create a comprehensive final summary that:
        1. Integrates all key points from all sections into a cohesive narrative
        2. Identifies and connects main themes across the entire video
        3. Removes redundancy while preserving important details
        4. Organizes information logically with clear sections
        5. Provides actionable insights and key takeaways
        6. Maintains appropriate length and detail level
        
        Structure your response with clear headings and bullet points where appropriate.
        """
        
        final_summary = await self._robust_llm_call([HumanMessage(content=combine_prompt)], 
                                                  operation_name="final summary combination")
        
        return final_summary or "Unable to generate combined summary"

    async def _extract_bullet_points(self, summary: str) -> str:
        """Extract key points in bullet format"""
        prompt = f"""
        Convert this summary into clear, concise bullet points:
        
        {summary}
        
        Create 8-12 bullet points that capture the most important information.
        Use clear, actionable language and organize by theme where appropriate.
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="bullet points extraction")
        return result or "Unable to generate bullet points"

    async def _extract_key_insights(self, summary: str) -> str:
        """Extract key insights and takeaways"""
        prompt = f"""
        From this summary, identify the 5-7 most important insights or takeaways:
        
        {summary}
        
        Focus on:
        - Strategic implications
        - Actionable advice
        - Important revelations or discoveries
        - Practical applications
        - Key learnings
        
        Present as numbered insights with brief explanations.
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="key insights extraction")
        return result or "Unable to generate key insights"

    async def _create_audio_summary(self, summary: str) -> str:
        """Create audio-optimized version of summary"""
        prompt = f"""
        Create an audio-friendly version of this summary optimized for text-to-speech:
        
        {summary}
        
        Requirements:
        - Conversational tone suitable for listening
        - Remove formatting markers and bullet points
        - Use smooth transitions between topics
        - Maintain all key information
        - Structure for easy listening comprehension
        - Keep engaging and informative
        """
        
        result = await self._robust_llm_call([HumanMessage(content=prompt)], operation_name="audio summary creation")
        return result or summary  # Fallback to original summary

    async def _create_translated_audio_summary(self, summary: str, target_language: str, proficiency_level: str = None) -> str:
        """Create audio-optimized summary translated to target language with proficiency-based vocabulary support"""
        
        # Map proficiency levels to learning-friendly settings
        level_map = {
            "beginner": {
                "label": {"French": "Débutant", "Spanish": "Principiante"},
                "cefr": "A2–B1",
                "style": "phrases courtes et simples; vocabulaire de base; rythme lent; explications claires des termes techniques.",
                "vocab_items": 8,
            },
            "intermediate": {
                "label": {"French": "Intermédiaire", "Spanish": "Intermedio"},
                "cefr": "B1–B2",
                "style": "phrases bien structurées; vitesse modérée; quelques explications des termes spécialisés.",
                "vocab_items": 6,
            },
            "advanced": {
                "label": {"French": "Avancé", "Spanish": "Avanzado"},
                "cefr": "C1",
                "style": "registre naturel et fluide; rythme natif; peu d'explications nécessaires.",
                "vocab_items": 0,  # no dedicated vocab section
            },
        }
        
        prof = level_map.get((proficiency_level or "advanced").lower())
        print(f"🎓 DEBUG: proficiency_level='{proficiency_level}', prof={prof}")
        
        # Build vocabulary warm-up section if needed
        vocab_block_req = ""
        if prof and prof["vocab_items"] > 0:
            if target_language == "French":
                vocab_block_req = f"""
                **Commencez par un section "Vocabulaire" (≈{prof['vocab_items']} termes):**
                - Sélectionnez les mots/expressions techniques ou peu fréquents du résumé
                - Format: **Mot** — définition simple (1 ligne) + phrase d'exemple courte
                - Présentez en français naturel pour faciliter l'apprentissage
                
                """
                print(f"🎓 DEBUG: vocab_block_req set for French: {bool(vocab_block_req)}")
            else:  # Spanish
                vocab_block_req = f"""
                **Comience con una sección "Glosario" (≈{prof['vocab_items']} términos):**
                - Seleccione palabras/expresiones técnicas o poco frecuentes del resumen
                - Formato: **Palabra** — definición sencilla (1 línea) + frase de ejemplo corta
                - Presente en español natural para facilitar el aprendizaje
                
                """

        style_hint = prof["style"] if prof else "tono conversacional natural; fluidez nativa."
        level_tag = prof["label"][target_language] if prof else ("Avancé" if target_language == "French" else "Avanzado")

        # Determine if this is translation or language-level adaptation
        instruction_verb = "Adaptez" if "french" in summary.lower() and target_language == "French" else \
                          "Adapte" if "español" in summary.lower() and target_language == "Spanish" else \
                          f"Traduisez en {target_language} et adaptez"

        prompt = f"""
        {instruction_verb} ce résumé pour la synthèse vocale en **{target_language}** (niveau: {level_tag}):
        
        {summary}
        
        **Exigences:**
        - Niveau de langue: {style_hint}
        - Maintenez toute l'information clé avec précision technique
        - Transitions fluides entre les idées; évitez les listes à puces
        - Conservez les noms propres et chiffres avec format local approprié
        - Ton engageant et informatif, adapté à l'écoute
        - **Même si le contenu source est déjà en {target_language}, ajustez le niveau de complexité et ajoutez le vocabulaire selon les spécifications**
        
        {vocab_block_req}
        
        **Structure de sortie:**
        1) {f"Section Vocabulaire/Glosario (si demandée ci-dessus)" if vocab_block_req else ""}
        2) **Narration principale**: Texte prêt pour la synthèse vocale en {target_language}, adapté au niveau {level_tag}
        
        **Répondez uniquement en {target_language}.**
        """
        
        print(f"🎓 DEBUG: Final prompt has vocab_block_req: {bool(vocab_block_req)}")
        print(f"🎓 DEBUG: Prompt preview: {prompt[:300]}...")
        
        result = await self._robust_llm_call(
            [HumanMessage(content=prompt)], 
            operation_name=f"translated audio summary ({target_language}/{proficiency_level or 'advanced'})"
        )
        return result or f"Unable to create {target_language} audio summary"

    async def _condense_for_tts(self, text: str) -> str:
        """Condense a long summary into TTS-friendly length while preserving key content"""
        # Target must be under 4096 - use 3900 as safe target
        target_chars = 3900
        target_words = 550  # Roughly 3900 chars at ~7 chars per word
        
        condense_prompt = f"""
        CRITICAL: You must condense this {len(text)}-character summary to EXACTLY {target_words} words (approximately {target_chars} characters) for TTS compatibility.

        The current summary is {len(text)} characters and EXCEEDS the 4096 character limit. You must make significant cuts while preserving the most important content.

        PRESERVE IN ORDER OF IMPORTANCE:
        1. Main conclusions and final recommendations
        2. Key specific findings and comparisons
        3. Important numbers, prices, measurements
        4. Critical trade-offs and practical implications

        AGGRESSIVE CUTS REQUIRED:
        - Combine multiple sentences into single, direct statements  
        - Eliminate all redundant explanations and examples
        - Remove transition phrases and filler language
        - Merge similar points together
        - Cut lengthy descriptions down to essential facts
        - Use shorter, more direct wording throughout

        AIM FOR EXACTLY {target_words} WORDS. This is NOT optional - the summary must fit the TTS limit.

        Original summary:
        {text}
        """
        
        condensed_content = await self._robust_llm_call(
            [HumanMessage(content=condense_prompt)], 
            "Summary condensation"
        )
        
        if condensed_content:
            return condensed_content
        else:
            print("⚠️ Failed to condense summary for TTS, using truncated version")
            # Fallback: intelligently truncate at sentence boundary
            if len(text) <= 3800:
                return text
            sentences = text.split('. ')
            condensed = ""
            for sentence in sentences:
                if len(condensed + sentence + ". ") <= 3800:
                    condensed += sentence + ". "
                else:
                    break
            return condensed.strip()
    
    def _split_text_for_tts(self, text: str, max_chunk_chars: int = 3800) -> List[str]:
        """Split text into TTS-friendly chunks at sentence boundaries"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            # Add period back unless it's the last sentence
            sentence_with_period = sentence + '.' if not sentence.endswith('.') else sentence
            
            # Check if adding this sentence would exceed limit
            if len(current_chunk) + len(sentence_with_period) + 1 > max_chunk_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_with_period
            else:
                if current_chunk:
                    current_chunk += ' ' + sentence_with_period
                else:
                    current_chunk = sentence_with_period
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    async def _generate_chunked_tts(self, text_chunks: List[str], base_filename: str) -> Optional[str]:
        """Generate TTS for multiple chunks and combine them"""
        import tempfile
        import shutil
        
        chunk_files = []
        
        try:
            # Generate TTS for each chunk
            for i, chunk in enumerate(text_chunks):
                print(f"🎵 Generating TTS for chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    chunk_filename = temp_file.name
                
                # Generate TTS for this chunk
                chunk_result = await self._generate_single_tts(chunk, chunk_filename)
                if chunk_result:
                    chunk_files.append(chunk_filename)
                    print(f"✅ Generated chunk {i+1}")
                else:
                    print(f"❌ Failed to generate chunk {i+1}")
                    # Clean up any files created so far
                    for f in chunk_files:
                        try:
                            import os
                            os.unlink(f)
                        except:
                            pass
                    return None
            
            # Combine audio files
            if not base_filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_filename = f"combined_tts_{timestamp}.mp3"
            
            # Ensure exports directory exists and get full path
            exports_dir = Path('exports')
            exports_dir.mkdir(exist_ok=True)
            final_filename = str(exports_dir / base_filename)
            
            if len(chunk_files) == 1:
                # Only one chunk, just rename it
                shutil.move(chunk_files[0], final_filename)
            else:
                # Try to combine with ffmpeg, fall back to simple concatenation
                combined = self._combine_audio_files(chunk_files, final_filename)
                if not combined:
                    print("⚠️ Audio combination failed, using first chunk only")
                    shutil.move(chunk_files[0], final_filename)
            
            print(f"✅ Combined TTS audio saved: {final_filename}")
            return final_filename
            
        except Exception as e:
            print(f"❌ Chunked TTS generation failed: {e}")
            return None
        finally:
            # Clean up temporary chunk files
            for chunk_file in chunk_files:
                try:
                    import os
                    os.unlink(chunk_file)
                except:
                    pass
    
    async def _generate_single_tts(self, text: str, output_filename: str) -> Optional[str]:
        """Generate TTS for a single text chunk"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return None
            
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "mp3"
        }
        
        response = self._make_request_with_retry(url, headers, payload)
        if response and response.status_code == 200:
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            return output_filename
        else:
            return None
    
    def _combine_audio_files(self, chunk_files: List[str], output_filename: str) -> bool:
        """Combine multiple audio files into one using ffmpeg if available"""
        try:
            import subprocess
            
            # Create a text file listing all input files
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{chunk_file}'\n")
                filelist_path = f.name
            
            # Use ffmpeg to concatenate
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filelist_path, '-c', 'copy', output_filename, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up
            import os
            os.unlink(filelist_path)
            
            if result.returncode == 0:
                print("✅ Audio files combined successfully with ffmpeg")
                return True
            else:
                print(f"⚠️ ffmpeg failed: {result.stderr}")
                return False
                
        except (FileNotFoundError, subprocess.SubprocessError):
            print("⚠️ ffmpeg not available, cannot combine audio chunks")
            return False
        except Exception as e:
            print(f"⚠️ Audio combination error: {e}")
            return False

    def _make_request_with_retry(self, url: str, headers: dict, payload: dict, max_retries: int = 3, timeout: int = 60) -> Optional[object]:
        """Make HTTP request with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 TTS API attempt {attempt + 1}/{max_retries}")
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ API Error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:  # Last attempt
                        return response
                    time.sleep(2 ** attempt)  # Wait before retry
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except Exception as e:
                print(f"⚠️ Request error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None

    async def _robust_llm_call(self, messages: list, operation_name: str = "LLM call", max_retries: int = 3) -> Optional[str]:
        """Make LLM API call with timeout and retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 {operation_name} attempt {attempt + 1}/{max_retries}")
                
                # Use asyncio.wait_for to implement timeout
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=120.0  # 2 minute timeout for LLM calls
                )
                
                return response.content
                
            except asyncio.TimeoutError:
                print(f"⚠️ {operation_name} timed out on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                continue
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str:
                    wait_time = (2 ** attempt) + 5  # Longer wait for rate limits
                    print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"⚠️ {operation_name} error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
        
        print(f"❌ All {operation_name} attempts failed")
        return None

    async def generate_tts_audio(self, text: str, output_filename: str = None) -> Optional[str]:
        """Generate TTS audio using OpenAI API with robust error handling and auto-condensing
        
        Args:
            text: Text to convert to speech
            output_filename: Optional filename (will generate if not provided)
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            # Get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ OPENAI_API_KEY not found in environment")
                return None
            
            # Step 1: Handle TTS character limit with chunking approach
            if len(text) > 4090:  # Only chunk when actually exceeding OpenAI's 4096 limit
                print(f"📝 Audio summary is {len(text)} characters (exceeds OpenAI's 4096 TTS limit)")
                print("🔄 Splitting into chunks and combining audio files...")
                tts_chunks = self._split_text_for_tts(text)
                print(f"✅ Split into {len(tts_chunks)} chunks (preserves all content)")
                return await self._generate_chunked_tts(tts_chunks, output_filename)
            else:
                print(f"✅ Audio summary length: {len(text)} chars (within TTS limits)")
                tts_text = text
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"audio_summary_{timestamp}.mp3"
            
            # Ensure exports directory exists
            exports_dir = Path('exports')
            exports_dir.mkdir(exist_ok=True)
            output_path = exports_dir / output_filename
            
            # Step 2: Make API request with robust retry logic
            url = "https://api.openai.com/v1/audio/speech"
            payload = {
                "model": "tts-1",  # Standard quality (tts-1-hd for higher quality)
                "input": tts_text,
                "voice": "fable"  # CURRENT: Warm, engaging male voice - great for storytelling
            }
            
            # OpenAI TTS Voice Options (change "voice" above to switch):
            # "alloy"   - Male (neutral): Natural, smooth young male voice, could pass as gender-neutral
            # "echo"    - Male: Articulate, precise young male voice, very proper English style
            # "fable"   - Male: Warm, engaging young male voice, perfect for storytelling (CURRENT)
            # "onyx"    - Male: Deep, authoritative older male voice, BBC presenter style  
            # "nova"    - Female: Bright, energetic young female voice
            # "shimmer" - Female: Soft, gentle young female voice, soothing tone
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print("🎙️ Generating TTS audio with OpenAI...")
            response = self._make_request_with_retry(url, headers, payload)
            
            if response and response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ TTS audio saved to {output_path}")
                return str(output_path)
            elif response:
                print(f"❌ Final TTS API Error: {response.status_code} - {response.text}")
                return None
            else:
                print("❌ All TTS API attempts failed")
                return None
                
        except Exception as e:
            print(f"❌ TTS generation failed: {str(e)}")
            return None

        # HUME AI TTS (COMMENTED OUT - keeping for reference)
        # try:
        #     # Get API key from environment
        #     api_key = os.getenv('HUME_API_KEY')
        #     if not api_key:
        #         print("❌ HUME_API_KEY not found in environment")
        #         return None
        #     
        #     # Prepare API request
        #     url = "https://api.hume.ai/v0/tts/file"
        #     payload = {
        #         "utterances": [{
        #             "text": text
        #         }],
        #         "format": {"type": "mp3"}
        #     }
        #     headers = {
        #         "X-Hume-Api-Key": api_key,
        #         "Content-Type": "application/json"
        #     }
        #     
        #     print("🎙️ Generating TTS audio...")
        #     response = requests.post(url, headers=headers, json=payload, timeout=30)
        #     
        #     if response.status_code == 200:
        #         with open(output_path, "wb") as f:
        #             f.write(response.content)
        #         print(f"✅ TTS audio saved to {output_path}")
        #         return str(output_path)
        #     else:
        #         print(f"❌ TTS API Error: {response.status_code} - {response.text}")
        #         return None
        #         
        # except Exception as e:
        #     print(f"❌ TTS generation failed: {str(e)}")
        #     return None


async def main():
    """Example usage of the YouTube Summarizer"""
    
    # Initialize summarizer (requires API keys in .env file)
    try:
        summarizer = YouTubeSummarizer(llm_provider="openai", model="gpt-4")
    except Exception as e:
        print(f"Failed to initialize summarizer: {e}")
        print("Make sure you have OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return
    
    # Example YouTube URL (replace with actual video)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll as safe test
    
    # Process the video
    result = await summarizer.process_video(test_url, summary_type="comprehensive")
    
    # Display results
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("\n" + "="*60)
        print("YOUTUBE VIDEO SUMMARY")
        print("="*60)
        print(f"Title: {result['metadata']['title']}")
        print(f"Channel: {result['metadata']['uploader']}")
        print(f"Duration: {result['metadata']['duration']} seconds")
        print(f"Views: {result['metadata']['view_count']:,}")
        
        print(f"\nHeadline: {result['summary']['headline']}")
        print(f"\nSummary:\n{result['summary']['summary']}")
        
        print(f"\nContent Analysis:")
        analysis = result['analysis']
        print(f"Category: {', '.join(analysis.get('category', ['Unknown']))}")
        print(f"Sentiment: {analysis.get('sentiment', 'Unknown')}")
        print(f"Target Audience: {analysis.get('target_audience', 'Unknown')}")
        print(f"Complexity: {analysis.get('complexity_level', 'Unknown')}")
        print(f"Key Topics: {', '.join(analysis.get('key_topics', ['Unknown']))}")


if __name__ == "__main__":
    asyncio.run(main())