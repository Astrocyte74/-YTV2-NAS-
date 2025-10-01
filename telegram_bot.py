#!/usr/bin/env python3
"""
YouTube Video Summarizer Bot - Main Orchestrator
Integrates Telegram bot, web dashboard, and modular processing components
Runs both Telegram bot service and web dashboard HTTP server
"""

import asyncio
import os
import re
import sys
import logging
import time
import hashlib
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import threading
import queue
from socketserver import ThreadingMixIn
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
from urllib.parse import urlparse, parse_qs

# Import our modular components
from modules.telegram_handler import YouTubeTelegramBot
from modules.report_generator import JSONReportGenerator
from youtube_summarizer import YouTubeSummarizer
from modules.metrics import metrics
from modules.event_stream import report_event_stream, emit_report_event

# SQLite backend support
try:
    from modules.sqlite_content_index import SQLiteContentIndex
    USING_SQLITE = True
except ImportError:
    print("SQLite backend not available, using JSON fallback")
    USING_SQLITE = False

# Load environment variables from .env file and stack.env
load_dotenv()

# Try to load .env.nas first (for NAS deployment)
nas_env_path = Path('./.env.nas')
if nas_env_path.exists():
    load_dotenv(nas_env_path)
    print(f"📁 Loaded NAS environment from {nas_env_path}")

# Also try to load from stack.env in parent directories for API keys
stack_env_paths = [
    Path('./stack.env'),
    Path('../stack.env'), 
    Path('../../stack.env'),
    Path('/Users/markdarby/projects/YTv3/stack.env')  # Explicit path for production
]

for env_path in stack_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📁 Loaded environment from {env_path}")
        break

# Also try to source .envrc for project-specific overrides
try:
    import subprocess
    result = subprocess.run(['bash', '-c', 'source .envrc 2>/dev/null && env'], 
                          capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if '=' in line and not line.startswith('_'):
                key, value = line.split('=', 1)
                os.environ[key] = value
        print(f"📁 Loaded environment from .envrc")
except Exception as e:
    pass  # Continue if .envrc sourcing fails

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

BOT_INSTANCE: Optional[YouTubeTelegramBot] = None  # Set in main()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in its own thread."""

    daemon_threads = True
    allow_reuse_address = True


# Template loading utility
def load_template(template_name: str) -> Optional[str]:
    """Load HTML template from file"""
    try:
        template_path = Path(template_name)
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        else:
            logger.error(f"Template not found: {template_name}")
            return None
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}")
        return None

# Extract report metadata utility
def extract_report_metadata(file_path: Path) -> Dict:
    """Extract metadata from HTML or JSON report file"""
    try:
        # Skip hidden files
        if file_path.name.startswith('.') or file_path.name.startswith('._'):
            raise ValueError(f"Skipping hidden file: {file_path.name}")
            
        # Handle JSON reports (preferred)
        if file_path.suffix == '.json':
            return extract_json_report_metadata(file_path)
        
        # Handle HTML reports (legacy)
        elif file_path.suffix == '.html':
            return extract_html_report_metadata(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
    except Exception as e:
        logger.warning(f"Error processing report {file_path.name}: {e}")
        return {
            'filename': file_path.name,
            'title': file_path.stem,
            'channel': 'Unknown Channel',
            'model': 'Unknown',
            'summary_preview': 'Error loading preview',
            'thumbnail_url': '',
            'created_date': 'Unknown',
            'created_time': 'Unknown',
            'timestamp': 0,
            'url': '',
            'video_id': '',
            'duration': 0
        }

def extract_json_report_metadata(file_path: Path) -> Dict:
    """Extract metadata from JSON report file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Extract data from JSON structure
        video_info = report.get('video', {})
        summary_info = report.get('summary', {})
        processing_info = report.get('processing', {})
        metadata = report.get('metadata', {})
        
        # Get file modification time for fallback
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Try to parse generated_at timestamp
        generated_at = metadata.get('generated_at', '')
        try:
            if generated_at:
                report_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            else:
                report_time = mtime
        except:
            report_time = mtime
        
        return {
            'filename': file_path.name,
            'title': video_info.get('title', 'Unknown Title'),
            'channel': video_info.get('channel', 'Unknown Channel'),
            'model': processing_info.get('model', 'Unknown Model'),
            'summary_preview': (summary_info.get('content', {}).get('summary', '')[:150] + "...") if summary_info.get('content', {}).get('summary') else "No preview available",
            'thumbnail_url': video_info.get('thumbnail', ''),
            'created_date': report_time.strftime('%B %d, %Y'),
            'created_time': report_time.strftime('%H:%M'),
            'timestamp': report_time.timestamp(),
            'url': video_info.get('url', ''),
            'video_id': video_info.get('video_id', ''),
            'duration': video_info.get('duration', 0)
        }
        
    except Exception as e:
        logger.error(f"Error extracting JSON metadata from {file_path.name}: {e}")
        raise

def extract_html_report_metadata(file_path: Path) -> Dict:
    """Extract metadata from HTML report file (legacy support)"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Extract title
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content)
        title = title_match.group(1).strip() if title_match else file_path.stem
        
        # Extract channel
        channel_match = re.search(r'<div class="channel">([^<]+)</div>', content)
        channel = channel_match.group(1).strip() if channel_match else "Unknown Channel"
        
        # Extract model
        model_match = re.search(r'<div class="model-badge[^"]*">([^<]+)</div>', content)
        model = model_match.group(1).strip() if model_match else "GPT-4"
        
        # Extract summary preview
        summary_match = re.search(r'<div class="content">.*?<p[^>]*>([^<]{0,200})', content, re.DOTALL)
        summary_preview = summary_match.group(1).strip()[:150] + "..." if summary_match else "No preview available"
        
        # Extract thumbnail URL
        thumbnail_match = re.search(r'<img[^>]+src="([^"]+)"[^>]*alt="Video thumbnail"', content)
        thumbnail_url = thumbnail_match.group(1) if thumbnail_match else ""
        
        # Get file info
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        return {
            'filename': file_path.name,
            'title': title,
            'channel': channel,
            'model': model,
            'summary_preview': summary_preview,
            'thumbnail_url': thumbnail_url,
            'created_date': mtime.strftime('%B %d, %Y'),
            'created_time': mtime.strftime('%H:%M'),
            'timestamp': mtime.timestamp(),
            'url': '',  # HTML reports don't typically store the original URL
            'video_id': '',
            'duration': 0
        }
        
    except Exception as e:
        logger.error(f"Error extracting HTML metadata from {file_path.name}: {e}")
        raise

class ModernDashboardHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with modern template system"""

    CORS_ALLOWED_HEADERS = 'Authorization, Content-Type, ngrok-skip-browser-warning, X-Reprocess-Token'
    CORS_ALLOWED_METHODS = 'GET,POST,OPTIONS'

    def send_cors_headers(self):
        origin = self.headers.get('Origin')
        allowed_origin = origin if origin else '*'
        self.send_header('Access-Control-Allow-Origin', allowed_origin)
        self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Headers', self.CORS_ALLOWED_HEADERS)
        self.send_header('Access-Control-Allow-Methods', self.CORS_ALLOWED_METHODS)
        self.send_header('Access-Control-Max-Age', '600')

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/report-events':
            self.serve_api_report_events()
        elif self.path == '/api/metrics':
            self.serve_api_metrics()
        elif self.path == '/status':
            self.serve_status()
        elif self.path.startswith('/api/'):
            self.serve_api()
        elif self.path.endswith('.css'):
            self.serve_css()
        elif self.path.endswith('.js'):
            self.serve_js()
        elif self.path.endswith('.html') and self.path != '/':
            self.serve_report()
        elif self.path.endswith('.json') and self.path != '/':
            self.serve_report()
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/delete-reports':
            self.handle_delete_reports()
        elif self.path == '/api/reprocess':
            self.handle_reprocess_request()
        else:
            self.send_error(404, "Endpoint not found")
    
    def serve_dashboard(self):
        """Serve the modern dashboard using templates"""
        try:
            # Get report files from multiple directories (JSON preferred, HTML legacy)
            report_dirs = [
                Path('./data/reports'),  # JSON reports (primary)
                Path('./exports'),       # HTML reports (legacy)
                Path('.')               # Current directory (legacy)
            ]
            
            all_report_files = []
            
            for report_dir in report_dirs:
                if report_dir.exists():
                    # Get JSON reports (preferred)
                    json_files = [f for f in report_dir.glob('*.json') 
                                   if not f.name.startswith('._')
                                   and not f.name.startswith('.')]
                    all_report_files.extend(json_files)
                    
                    # Get HTML reports (legacy support)
                    html_files = [f for f in report_dir.glob('*.html') 
                                   if f.name not in ['dashboard_template.html', 'report_template.html']
                                   and not f.name.startswith('._')
                                   and not f.name.startswith('.')]
                    all_report_files.extend(html_files)
            
            # Remove duplicates and sort by modification time (newest first)
            unique_files = list(set(all_report_files))
            sorted_files = sorted(unique_files, key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Extract metadata from all reports
            reports_data = []
            for file_path in sorted_files:
                try:
                    metadata = extract_report_metadata(file_path)
                    reports_data.append(metadata)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path.name}: {e}")
                    continue
            
            # Load dashboard template
            template_content = load_template('dashboard_template.html')
            
            if template_content:
                # Get base URL
                base_url = os.getenv('DASHBOARD_URL', 'https://ytv2-vy9k.onrender.com')
                
                # Replace template placeholders
                dashboard_html = template_content.format(
                    reports_data=json.dumps(reports_data, ensure_ascii=False),
                    base_url=base_url
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(dashboard_html.encode('utf-8'))
            else:
                # Fallback error page
                error_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Template Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 2rem; }}
        .error {{ color: #dc2626; background: #fef2f2; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }}
        .report-list {{ margin: 1rem 0; }}
        .report-item {{ margin: 0.5rem 0; }}
    </style>
</head>
<body>
    <h1>🚨 Dashboard Template Error</h1>
    <div class="error">
        <p><strong>Could not load dashboard_template.html</strong></p>
        <p>Please ensure the template file exists in the project directory.</p>
    </div>
    
    <h2>📊 Available Reports ({len(html_files)})</h2>
    <div class="report-list">"""
                
                for file_path in html_files[:20]:  # Show first 20
                    metadata = extract_report_metadata(file_path)
                    error_html += f"""
        <div class="report-item">
            <a href="/{file_path.name}">{metadata['title']}</a>
            <small> - {metadata['channel']} ({metadata['model']})</small>
        </div>"""
                
                if len(html_files) > 20:
                    error_html += f"<p><em>... and {len(html_files) - 20} more reports</em></p>"
                
                error_html += """
    </div>
</body>
</html>"""
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(error_html.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}")
            self.send_error(500, "Internal server error")
    
    def serve_css(self):
        """Serve CSS files"""
        try:
            filename = self.path[1:]  # Remove leading slash
            css_file = Path('static') / filename
            if css_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(css_file.read_bytes())
            else:
                self.send_error(404, "CSS file not found")
        except Exception as e:
            logger.error(f"Error serving CSS {self.path}: {e}")
            self.send_error(500, "Error serving CSS")
    
    def serve_js(self):
        """Serve JavaScript files"""
        try:
            filename = self.path[1:]  # Remove leading slash
            js_file = Path('static') / filename
            if js_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(js_file.read_bytes())
            else:
                self.send_error(404, "JavaScript file not found")
        except Exception as e:
            logger.error(f"Error serving JS {self.path}: {e}")
            self.send_error(500, "Error serving JavaScript")
    
    def serve_report(self):
        """Serve individual report pages"""
        try:
            filename = self.path[1:]  # Remove leading slash
            
            # Look for HTML files first (legacy)
            html_paths = [Path(filename), Path('./exports') / filename]
            for path in html_paths:
                if path.exists() and path.suffix == '.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(path.read_bytes())
                    return
            
            # Look for JSON files in data/reports
            if filename.endswith('.json'):
                json_path = Path('./data/reports') / filename
                if json_path.exists():
                    self.serve_json_report(json_path)
                    return
            else:
                # Try adding .json extension if not present
                json_filename = filename + '.json'
                json_path = Path('./data/reports') / json_filename
                if json_path.exists():
                    self.serve_json_report(json_path)
                    return
            
            self.send_error(404, "Report not found")
        except Exception as e:
            logger.error(f"Error serving report {self.path}: {e}")
            self.send_error(500, "Error serving report")
    
    def serve_json_report(self, json_path: Path):
        """Serve a JSON report wrapped in professional HTML template"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Extract data for display
            video_info = report_data.get('video', {})
            summary = report_data.get('summary', {})
            processing = report_data.get('processing', {})
            stats = report_data.get('stats', {})
            metadata = report_data.get('metadata', {})
            
            title = video_info.get('title', 'Unknown Video')
            channel = video_info.get('channel', 'Unknown Channel')
            duration_str = video_info.get('duration_string', '')
            url = video_info.get('url', '')
            thumbnail = video_info.get('thumbnail', '')
            view_count = video_info.get('view_count', 0)
            upload_date = video_info.get('upload_date', '')
            
            # Properly extract summary content
            summary_content = summary.get('content', {})
            if isinstance(summary_content, dict):
                summary_text = summary_content.get('summary', 'No summary available')
                headline = summary_content.get('headline', '')
                summary_type = summary_content.get('summary_type', 'comprehensive')
            else:
                summary_text = str(summary_content) if summary_content else 'No summary available'
                headline = ''
                summary_type = summary.get('type', 'comprehensive')
            
            # Get analysis data
            analysis = summary.get('analysis', {})
            categories = analysis.get('category', [])
            target_audience = analysis.get('target_audience', '')
            
            # Get processing info
            model = processing.get('model', 'Unknown')
            provider = processing.get('llm_provider', 'Unknown')
            
            # Format dates
            try:
                if upload_date and len(upload_date) == 8:
                    formatted_date = f"{upload_date[4:6]}/{upload_date[6:8]}/{upload_date[:4]}"
                else:
                    formatted_date = upload_date or 'Unknown'
            except:
                formatted_date = 'Unknown'
            
            # Format view count
            if view_count:
                if view_count >= 1000000:
                    formatted_views = f"{view_count/1000000:.1f}M views"
                elif view_count >= 1000:
                    formatted_views = f"{view_count/1000:.1f}K views"
                else:
                    formatted_views = f"{view_count:,} views"
            else:
                formatted_views = 'Views unknown'
            
            # Discover audio file if present (support legacy and new prefixes)
            audio_url = ''
            try:
                video_id = video_info.get('video_id', '')
                if video_id:
                    candidates = []
                    for pattern in [f'audio_{video_id}_*.mp3', f'{video_id}_*.mp3']:
                        candidates.extend(Path('./exports').glob(pattern))
                    if candidates:
                        latest = max(candidates, key=lambda p: p.stat().st_mtime)
                        base_url = os.getenv('NGROK_URL', '')
                        audio_url = (base_url.rstrip('/') + f"/exports/{latest.name}") if base_url else f"/exports/{latest.name}"
            except Exception:
                audio_url = ''

            # Build mini-player script safely (avoid f-string brace escaping)
            script_block = ''
            if audio_url:
                _dur = video_info.get('duration', 0)
                script_block = (
                    """
<script>(function(){
  const a = document.getElementById('summaryAudio');
  if (!a) return;
  const el = {
    play: document.getElementById('playPauseBtn'),
    cur:  document.getElementById('cur'),
    dur:  document.getElementById('dur'),
    seek: document.getElementById('seek'),
    speedBtn: document.getElementById('speedBtn'),
    bar:  document.getElementById('listenSticky'),
    mPlay:document.getElementById('mPlayPause'),
    mCur: document.getElementById('mCur'),
    mDur: document.getElementById('mDur'),
    mSeek:document.getElementById('mSeek'),
    mSpeed:document.getElementById('mSpeed'),
  };
  const fmt = (s) => {
    if (isNaN(s)) return '--:--';
    s = Math.max(0, Math.floor(s));
    const m = Math.floor(s/60), r = s%60;
    return `${m}:${r.toString().padStart(2,'0')}`;
  };
  function syncUI(){
    const { currentTime, duration, paused } = a;
    const pct = duration ? (currentTime/duration)*100 : 0;
    if (el.cur) el.cur.textContent = fmt(currentTime);
    if (el.dur) el.dur.textContent = fmt(duration);
    if (el.seek) el.seek.value = pct;
    if (el.mCur) el.mCur.textContent = fmt(currentTime);
    if (el.mDur) el.mDur.textContent = fmt(duration);
    if (el.mSeek) el.mSeek.value = pct;
    if (el.play) el.play.textContent = paused ? '▶' : '⏸';
    if (el.mPlay) el.mPlay.textContent = paused ? '▶' : '⏸';
  }
  function toggle(){ a.paused ? a.play() : a.pause(); }
  if (el.play) el.play.addEventListener('click', toggle);
  if (el.mPlay) el.mPlay.addEventListener('click', toggle);
  function setSeek(p){ if (a.duration) a.currentTime = (p/100)*a.duration; }
  if (el.seek) el.seek.addEventListener('input', e => setSeek(e.target.value));
  if (el.mSeek) el.mSeek.addEventListener('input', e => setSeek(e.target.value));
  const speeds = [1,1.25,1.5,1.75,2];
  function cycleSpeed(btn){
    const i = (speeds.indexOf(a.playbackRate)+1) % speeds.length;
    a.playbackRate = speeds[i];
    btn.textContent = `${speeds[i]}×`;
    if (btn === el.speedBtn && el.mSpeed) el.mSpeed.textContent = btn.textContent;
    if (btn === el.mSpeed && el.speedBtn) el.speedBtn.textContent = btn.textContent;
  }
  if (el.speedBtn) el.speedBtn.addEventListener('click', () => cycleSpeed(el.speedBtn));
  if (el.mSpeed) el.mSpeed.addEventListener('click', () => cycleSpeed(el.mSpeed));
  a.addEventListener('loadedmetadata', () => {
    syncUI();
    try {
      var v = __DUR__;
      if (v && a.duration) {
        var diff = Math.max(0, v - Math.floor(a.duration));
        var mm = Math.floor(diff/60), ss = diff%60;
        var elSave = document.getElementById('save');
        if (elSave) elSave.textContent = mm + ':' + String(ss).padStart(2,'0');
      }
    } catch(e) {}
  });
  a.addEventListener('timeupdate', syncUI);
  a.addEventListener('play', () => { const bar = document.getElementById('listenSticky'); if (bar) bar.hidden = false; syncUI(); });
  a.addEventListener('pause', syncUI);
  syncUI();
})();</script>
""".replace('__DUR__', str(_dur))
                )

            # Compact, professional HTML template
            html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{title} - YTV2</title>
    <style>
      :root {{ --border:#e5e7eb; --bg:#fff; --text:#0f172a; --bg-page:#f8fafc; --ring:#e5e7eb; --text-1:#0f172a; --text-2:#334155; --brand:#1d4ed8; }}
      * {{ box-sizing: border-box; }}
      body {{ margin:0; font-family: Inter, -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:var(--text); background:var(--bg-page); }}
      .topbar {{ position:sticky; top:0; z-index:40; backdrop-filter:blur(8px); background:rgba(255,255,255,.72); border-bottom:1px solid var(--border); }}
      .topbar-inner {{ max-width:1120px; margin:0 auto; height:56px; padding:0 16px; display:flex; align-items:center; justify-content:space-between; }}
      .btn-ghost {{ height:36px; padding:0 12px; border:1px solid #e5e7eb; border-radius:10px; background:#fff; color:#111827; font-size:14px; display:inline-flex; align-items:center; gap:6px; text-decoration:none; }}
      .btn-ghost:hover {{ background:#f9fafb; }}
      .btn-primary {{ height:36px; padding:0 12px; border-radius:10px; background:#111827; color:#fff; font-size:14px; text-decoration:none; display:inline-flex; align-items:center; }}
      .container {{ max-width:1120px; margin:0 auto; padding:16px 20px; }}
      .grid {{ display:grid; grid-template-columns:1fr; gap:16px; }}
      @media (min-width:768px) {{ .grid {{ grid-template-columns:1fr 1.1fr; }} }}
      .thumb {{ aspect-ratio:16/9; border-radius:14px; overflow:hidden; border:1px solid var(--ring); background:#fff; }}
      .thumb img {{ width:100%; height:100%; object-fit:cover; display:block; }}
      .title {{ margin:4px 0 8px; font-weight:700; font-size:22px; line-height:1.25; color:var(--text-1); }}
      .title.clamp {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
      .facts {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:8px; }}
      .chip {{ height:28px; padding:0 10px; border:1px solid var(--ring); border-radius:10px; background:rgba(255,255,255,.9); display:inline-flex; align-items:center; gap:6px; font-size:12.5px; }}
      .chip .label {{ color:#64748b; font-size:12px; }}
      .chip .value {{ color:#0f172a; font-weight:600; }}
      .chip.duration {{ background:#e0f2fe; border-color:#bfdbfe; }}
      .chip.views {{ background:#dcfce7; border-color:#bbf7d0; }}
      .chip.date {{ background:#ede9fe; border-color:#ddd6fe; }}
      .chip.model {{ background:#f3f4f6; }}
      .chip.type {{ background:#f3f4f6; }}
      .divider {{ border-top:2px solid #e0e7ff; margin:10px 0 6px; }}
      .icon-badge {{ padding:6px; border-radius:8px; background:#dbeafe; color:#1d4ed8; display:inline-flex; align-items:center; }}
      .card {{ margin-top:8px; background:var(--bg); border:1px solid var(--ring); border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.05); padding:14px 16px; }}
      .card h2 {{ margin:0 0 8px; font-size:15px; font-weight:700; color:var(--text-2); display:flex; align-items:center; gap:8px; }}
      .summary {{ max-width:78ch; white-space:pre-line; font-size:15px; line-height:1.6; color:var(--text-2); }}
      .summary p {{ margin:10px 0; hyphens:auto; }}
      /* Inline mini audio player */
      .listen-inline {{ margin-top:8px; }}
      .listen-card {{ display:flex; align-items:center; gap:12px; border:1px solid var(--ring); background:#fff; border-radius:12px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
      .listen-btn {{ width:36px; height:36px; border-radius:999px; border:1px solid var(--ring); background:#111827; color:#fff; display:grid; place-items:center; font-weight:700; cursor:pointer; }}
      .listen-info {{ flex:1; min-width:0; }}
      .listen-title {{ font-weight:600; font-size:14px; line-height:1.2; }}
      .listen-meta {{ color:#64748b; font-size:12px; margin-top:2px; }}
      .listen-actions {{ display:flex; align-items:center; gap:8px; }}
      #seek {{ width:100%; margin-top:6px; }}
      .chip.ghost {{ background:#fff; }}
      /* Sticky bottom mini-player for mobile */
      .listen-sticky {{ position: sticky; bottom:12px; margin:0 auto; left:0; right:0; max-width:520px; display:flex; align-items:center; gap:8px; background:#fff; border:1px solid var(--ring); padding:8px 10px; border-radius:999px; box-shadow:0 8px 24px rgba(0,0,0,.14); }}
      .listen-sticky .meta {{ flex:1; min-width:0; }}
      .listen-sticky button {{ height:34px; min-width:34px; border-radius:999px; border:1px solid var(--ring); background:#111827; color:#fff; }}
      .listen-sticky small {{ color:#64748b; display:block; }}
      #listenSticky[hidden] {{ display:none; }}
      @media (max-width:900px) {{ .listen-inline {{ display:none; }} }}
      @media (min-width:901px) {{ #listenSticky {{ display:none !important; }} }}
    </style>
    <script>
      function copyLink(url) {{
        const toCopy = url || window.location.href;
        navigator.clipboard.writeText(toCopy).catch(()=>{{}});
      }}
    </script>
</head>
<body>
  <header class=\"topbar\">
    <div class=\"topbar-inner\">
      <a class=\"btn-ghost\" href=\"/\">← Back</a>
      <div style=\"display:flex;align-items:center;gap:8px;\">
        {f'<a class="btn-ghost" href="{url}" target="_blank">Open on YouTube</a>' if url else ''}
        <button class=\"btn-ghost\" onclick=\"copyLink('{url}')\">Copy link</button>
        
      </div>
    </div>
  </header>

  <main class=\"container\">
    <section class=\"grid\">
      <div class=\"thumb\">{f'<img src="{thumbnail}" alt="" />' if thumbnail else ''}</div>
      <div>
        <h1 class=\"title clamp\">{title}</h1>
        <div class=\"facts\">
          <div class=\"chip brand\"><span class=\"label\">Channel:</span><span class=\"value\">{channel}</span></div>
          <div class=\"chip duration\"><span class=\"label\">Duration:</span><span class=\"value\">{duration_str or '—'}</span></div>
          <div class=\"chip views\"><span class=\"label\">Views:</span><span class=\"value\">{formatted_views}</span></div>
          <div class=\"chip date\"><span class=\"label\">Uploaded:</span><span class=\"value\">{formatted_date}</span></div>
          <div class=\"chip model\"><span class=\"label\">AI Model:</span><span class=\"value\">{model}{(' ('+provider+')') if provider else ''}</span></div>
        </div>
        {f'''<div class="listen-inline">
          <div class="listen-card">
            <button id="playPauseBtn" class="listen-btn" aria-label="Play audio summary">▶</button>
            <div class="listen-info">
              <div class="listen-title">Audio summary</div>
              <div class="listen-meta"><span id="cur">0:00</span> / <span id="dur">--:--</span> • <span id="save"></span></div>
              <input id="seek" type="range" min="0" max="100" value="0" aria-label="Seek audio" />
            </div>
            <div class="listen-actions">
              <button id="speedBtn" class="chip">1×</button>
              <a class="chip ghost" href="{audio_url}" download>Download</a>
            </div>
          </div>
        </div>''' if audio_url else ''}
      </div>
    </section>

    {f'''<div class="listen-sticky" id="listenSticky" hidden>
      <button id="mPlayPause" aria-label="Play audio">▶</button>
      <div class="meta">
        <strong>Audio summary</strong>
        <small><span id="mCur">0:00</span> / <span id="mDur">--:--</span></small>
        <input id="mSeek" type="range" min="0" max="100" value="0" aria-label="Seek audio" />
      </div>
      <button id="mSpeed">1×</button>
    </div>
    <audio id="summaryAudio" preload="metadata"><source src="{audio_url}" type="audio/mpeg" /></audio>''' if audio_url else ''}

    <div class=\"divider\"></div>

    <section class=\"card\">
      <div class=\"flex\" style=\"display:flex;align-items:center;gap:8px;margin-bottom:8px;\"><span class=\"icon-badge\">📄</span><h2 style=\"margin:0;\">Summary</h2></div>
      {f'<div class="summary">{headline}</div>' if headline else ''}
      <div class=\"summary\">{summary_text}</div>
    </section>
  </main>
  {script_block if audio_url else ''}
</body>
</html>"""
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error serving JSON report {json_path}: {e}")
            self.send_error(500, "Error serving JSON report")
    
    def serve_status(self):
        """Serve system status endpoint"""
        try:
            # Check if Telegram bot is configured
            telegram_configured = bool(os.getenv('TELEGRAM_BOT_TOKEN'))
            users_configured = bool(os.getenv('TELEGRAM_ALLOWED_USERS'))
            
            # Count reports
            report_generator = JSONReportGenerator()
            json_reports = len(report_generator.list_reports())
            
            html_reports = 0
            for report_dir in [Path('./exports'), Path('.')]:
                if report_dir.exists():
                    html_reports += len([f for f in report_dir.glob('*.html') 
                                       if f.name not in ['dashboard_template.html', 'report_template.html']
                                       and not f.name.startswith('._')])
            
            status_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "dashboard": "running",
                    "telegram_bot": "configured" if telegram_configured and users_configured else "not_configured"
                },
                "reports": {
                    "json_reports": json_reports,
                    "html_reports": html_reports,
                    "total_reports": json_reports + html_reports
                },
                "configuration": {
                    "telegram_configured": telegram_configured,
                    "users_configured": users_configured,
                    "web_port": os.getenv('WEB_PORT', 6452)
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(status_data, indent=2).encode())
            
        except Exception as e:
            logger.error(f"Error serving status: {e}")
            error_data = {"status": "error", "message": str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode())
    
    def serve_api(self):
        """Serve API endpoints"""
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query_params = parse_qs(parsed.query)

            if path == '/api/reports':
                self.serve_api_reports(query_params)
            elif path.startswith('/api/reports/'):
                report_id = path.rsplit('/', 1)[-1]
                self.serve_api_report_detail(report_id)
            elif path == '/api/config':
                self.serve_api_config()
            elif path == '/api/metrics':
                self.serve_api_metrics()
            else:
                self.send_error(404, "API endpoint not found")
        except Exception as e:
            logger.error(f"Error serving API {self.path}: {e}")
            self.send_error(500, "API error")
    
    def serve_api_reports(self, query_params=None):
        """Serve reports list API endpoint"""
        try:
            latest_only = False
            if query_params:
                values = query_params.get('latest')
                if values:
                    latest_only = any(str(value).lower() in ('1', 'true', 'yes') for value in values)

            report_generator = JSONReportGenerator()
            json_reports = report_generator.list_reports()

            api_reports = []
            for report in json_reports:
                api_reports.append({
                    "id": report.get('filename', '').replace('.json', ''),
                    "filename": report['filename'],
                    "title": report['title'],
                    "channel": report['channel'],
                    "duration": report.get('duration', 0),
                    "created_date": report.get('created_date', ''),
                    "created_time": report.get('created_time', ''),
                    "timestamp": report.get('timestamp', ''),
                    "type": "json",
                    "url": report.get('url', ''),
                    "video_id": report.get('video_id', ''),
                    "thumbnail_url": report.get('thumbnail', ''),
                    "model": report.get('model', 'Unknown'),
                    "summary_preview": report.get('summary_preview', '')
                })

            if not latest_only:
                html_dirs = [Path('./exports'), Path('.')]
                for report_dir in html_dirs:
                    if report_dir.exists():
                        html_files = [f for f in report_dir.glob('*.html')
                                      if f.name not in ['dashboard_template.html', 'report_template.html']
                                      and not f.name.startswith('._')]
                        for html_file in html_files:
                            try:
                                metadata = extract_html_report_metadata(html_file)
                                api_reports.append({
                                    "id": html_file.stem,
                                    "filename": metadata['filename'],
                                    "title": metadata['title'],
                                    "channel": metadata['channel'],
                                    "duration": 0,
                                    "created_date": metadata['created_date'],
                                    "created_time": metadata['created_time'],
                                    "timestamp": metadata['timestamp'],
                                    "type": "html",
                                    "url": "",
                                    "video_id": ""
                                })
                            except Exception:
                                continue

            api_reports.sort(key=lambda x: x['timestamp'], reverse=True)

            response_data = {
                "reports": api_reports,
                "total": len(api_reports),
                "json_count": len(json_reports),
                "html_count": len(api_reports) - len(json_reports)
            }

            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if latest_only:
                payload = {"report": api_reports[0] if api_reports else None}
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())
            else:
                self.wfile.write(json.dumps(response_data, ensure_ascii=False, indent=2).encode())

        except Exception as e:
            logger.error(f"Error serving reports API: {e}")
            error_data = {"error": "Failed to load reports", "message": str(e)}
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode())

    def serve_api_report_events(self):
        """Stream report ingest events via server-sent events."""
        client = None
        try:
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()

            client = report_event_stream.register(self)
            self.close_connection = False

            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
            except Exception:
                report_event_stream.unregister(client)
                return

            while True:
                try:
                    message = client.queue.get(timeout=15)
                except queue.Empty:
                    try:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        break
                    continue

                try:
                    self.wfile.write(message.encode('utf-8'))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    break
        finally:
            if client is not None:
                report_event_stream.unregister(client)
            self.close_connection = True

    def serve_api_metrics(self):
        """Return current processing metrics."""
        snapshot = metrics.snapshot()
        self.send_response(200)
        self.send_cors_headers()
        self.send_header('Content-type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(snapshot, ensure_ascii=False).encode('utf-8'))

    def handle_reprocess_request(self):
        """Schedule a headless reprocess job via the Telegram bot event loop."""
        global BOT_INSTANCE

        if BOT_INSTANCE is None or BOT_INSTANCE.loop is None:
            self.send_response(503)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'bot-unavailable'}).encode('utf-8'))
            return

        content_length = int(self.headers.get('Content-Length') or 0)
        if content_length <= 0:
            self.send_response(400)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'empty-request'}).encode('utf-8'))
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode('utf-8'))
        except Exception as exc:
            self.send_response(400)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'invalid-json', 'details': str(exc)}).encode('utf-8'))
            return

        required_token = os.getenv('REPROCESS_AUTH_TOKEN')
        provided_token = self.headers.get('X-Reprocess-Token')

        if required_token and provided_token != required_token:
            self.send_response(403)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'forbidden'}).encode('utf-8'))
            return

        video_id = payload.get('video_id') or payload.get('id')
        video_url = payload.get('url')
        summary_types = payload.get('summary_types')
        force = bool(payload.get('force', False))
        regenerate_audio = payload.get('regenerate_audio', True)

        if not video_id and video_url:
            match = BOT_INSTANCE.youtube_url_pattern.search(video_url)
            if match:
                video_id = match.group(1)

        if not video_id:
            self.send_response(400)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'missing-video-id'}).encode('utf-8'))
            return

        if isinstance(summary_types, str):
            summary_types = [item.strip() for item in summary_types.split(',') if item.strip()]
        elif not isinstance(summary_types, list):
            summary_types = None

        try:
            loop = BOT_INSTANCE.loop

            future = asyncio.run_coroutine_threadsafe(
                BOT_INSTANCE.reprocess_video(
                    video_id,
                    summary_types=summary_types,
                    force=force,
                    regenerate_audio=bool(regenerate_audio),
                    video_url=video_url,
                ),
                loop,
            )

            def _done_callback(fut: asyncio.Future):
                try:
                    fut.result()
                except Exception:
                    logger.exception("Reprocess task failed")

            future.add_done_callback(_done_callback)

            self.send_response(202)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'scheduled',
                'video_id': video_id,
                'summary_types': summary_types,
            }
            emit_report_event(
                'reprocess-scheduled',
                {
                    'video_id': video_id,
                    'summary_types': summary_types,
                },
            )
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        except Exception as exc:
            logger.exception("Failed to schedule reprocess job")
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'schedule-failed', 'details': str(exc)}).encode('utf-8'))
    
    def serve_api_report_detail(self, report_id):
        """Serve individual report API endpoint"""
        try:
            # Look for JSON report first
            json_file = Path('/app/data/reports') / f"{report_id}.json"
            if json_file.exists():
                report_data = json.loads(json_file.read_text())
                self.send_response(200)
                self.send_cors_headers()
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(report_data, ensure_ascii=False, indent=2).encode())
                return
            
            # Look for HTML report (legacy)
            html_dirs = [Path('./exports'), Path('.')]
            for html_dir in html_dirs:
                html_file = html_dir / f"{report_id}.html"
                if html_file.exists():
                    metadata = extract_html_report_metadata(html_file)
                    # For HTML reports, we return limited metadata
                    response_data = {
                        "type": "html",
                        "metadata": metadata,
                        "message": "HTML report - limited API data available"
                    }
                    self.send_response(200)
                    self.send_cors_headers()
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data, ensure_ascii=False, indent=2).encode())
                    return

            # Report not found
            self.send_error(404, "Report not found")
            
        except Exception as e:
            logger.error(f"Error serving report detail API: {e}")
            error_data = {"error": "Failed to load report", "message": str(e)}
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode())

    def serve_api_config(self):
        """Serve configuration API endpoint"""
        try:
            config_data = {
                "telegram": {
                    "bot_configured": bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                    "users_configured": bool(os.getenv('TELEGRAM_ALLOWED_USERS'))
                },
                "web": {
                    "port": int(os.getenv('WEB_PORT', 6452)),
                    "ngrok_url": os.getenv('NGROK_URL', '')
                },
                "directories": {
                    "json_reports": str(Path('./data/reports').absolute()),
                    "html_reports": str(Path('./exports').absolute())
                },
                "version": "1.0.0"
            }
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'public, max-age=300')  # Cache for 5 minutes
            self.end_headers()
            self.wfile.write(json.dumps(config_data, indent=2).encode())
            
        except Exception as e:
            logger.error(f"Error serving config API: {e}")
            error_data = {"error": "Failed to load configuration", "message": str(e)}
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode())
    
    def handle_delete_reports(self):
        """Handle POST request to delete selected reports"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            filenames = request_data.get('files', [])
            if not filenames:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No files specified"}).encode())
                return
            
            deleted_count = 0
            errors = []
            
            # Delete files from all possible locations
            search_dirs = [
                Path('./data/reports'),  # JSON reports
                Path('./exports'),       # HTML reports
                Path('.')               # Legacy location
            ]
            
            for filename in filenames:
                deleted = False
                for search_dir in search_dirs:
                    if search_dir.exists():
                        # Try different extensions
                        for ext in ['.json', '.html', '']:
                            file_path = search_dir / (filename + ext if not filename.endswith(ext) else filename)
                            if file_path.exists():
                                try:
                                    file_path.unlink()
                                    deleted = True
                                    deleted_count += 1
                                    logger.info(f"Deleted report: {file_path}")
                                    break
                                except Exception as e:
                                    errors.append(f"Failed to delete {file_path}: {e}")
                    if deleted:
                        break
                
                if not deleted:
                    errors.append(f"File not found: {filename}")
            
            # Send response
            response_data = {
                "deleted": deleted_count,
                "errors": errors,
                "message": f"Successfully deleted {deleted_count} reports"
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            logger.error(f"Error handling delete request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

def start_http_server():
    """Start the HTTP server for dashboard access"""
    port = int(os.getenv('WEB_PORT', 6452))
    
    # Find available port if the default is taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            logger.info(f"Port {port} is busy, using port {port + 1}")
            port = port + 1
    
    try:
        server_address = ('', port)
        httpd = ThreadedHTTPServer(server_address, ModernDashboardHTTPRequestHandler)
        
        logger.info(f"🌐 HTTP Server started on port {port}")
        logger.info(f"📊 Dashboard available at: http://localhost:{port}")
        
        def run_server():
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("HTTP Server stopped")
                httpd.shutdown()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return httpd, port
        
    except Exception as e:
        logger.error(f"Failed to start HTTP server: {e}")
        return None, None

async def main():
    """Main async function to run both Telegram bot and HTTP server"""
    logger.info("🤖 Starting YouTube Summarizer Bot with Modern Dashboard")
    
    # Start HTTP server in a separate thread
    httpd, port = start_http_server()
    
    if httpd:
        logger.info(f"🌐 Dashboard server running at: http://localhost:{port}")
    else:
        logger.error("❌ Failed to start HTTP server")
        return
    
    # Initialize and start Telegram bot
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    allowed_users_env = os.getenv('TELEGRAM_ALLOWED_USERS', '')
    
    if not telegram_token:
        logger.warning("⚠️  TELEGRAM_BOT_TOKEN not found. Running dashboard-only mode.")
        logger.info(f"✅ Dashboard-only mode ready at: http://localhost:{port}")
        
        try:
            # Keep dashboard server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down dashboard...")
            if httpd:
                httpd.shutdown()
        return
    
    # Parse allowed users
    allowed_user_ids = []
    if allowed_users_env:
        try:
            allowed_user_ids = [int(user_id.strip()) for user_id in allowed_users_env.split(',') if user_id.strip()]
        except ValueError as e:
            logger.warning(f"Error parsing TELEGRAM_ALLOWED_USERS: {e}")
    
    if not allowed_user_ids:
        logger.warning("⚠️  No allowed users configured. Set TELEGRAM_ALLOWED_USERS environment variable.")
        logger.info(f"✅ Dashboard-only mode ready at: http://localhost:{port}")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down dashboard...")
            if httpd:
                httpd.shutdown()
        return
    
    logger.info(f"👥 Telegram bot authorized for {len(allowed_user_ids)} users")
    
    # Create and start Telegram bot
    try:
        telegram_bot = YouTubeTelegramBot(telegram_token, allowed_user_ids)
        global BOT_INSTANCE
        BOT_INSTANCE = telegram_bot

        logger.info("🚀 Starting Telegram bot...")

        # Run both services concurrently
        try:
            await asyncio.gather(
                telegram_bot.run(),
                run_dashboard_monitor(httpd)
            )
        finally:
            BOT_INSTANCE = None

    except Exception as e:
        logger.error(f"Error running Telegram bot: {e}")
        if httpd:
            httpd.shutdown()
        raise

async def run_dashboard_monitor(httpd):
    """Monitor the dashboard server"""
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("🛑 Shutting down dashboard monitor...")
        if httpd:
            httpd.shutdown()

# Main execution
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
