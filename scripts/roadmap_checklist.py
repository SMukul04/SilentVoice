import os
import re
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

PORT = 8089
ROADMAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'ROADMAP.md'))

def parse_roadmap():
    if not os.path.exists(ROADMAP_PATH):
        return []
    
    with open(ROADMAP_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    parsed = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Task
        task_match = re.match(r'^(\s*)-\s*\[([ xX])\]\s*(.*)$', line)
        if task_match:
            indent = len(task_match.group(1))
            checked = task_match.group(2).lower() == 'x'
            text = task_match.group(3)
            parsed.append({
                "index": idx,
                "type": "task",
                "indent": indent,
                "checked": checked,
                "text": text
            })
            continue
            
        # Header
        header_match = re.match(r'^(#+)\s*(.*)$', stripped)
        if header_match:
            level = len(header_match.group(1))
            text = header_match.group(2)
            parsed.append({
                "index": idx,
                "type": "header",
                "level": level,
                "text": text
            })
            continue
            
        # Blockquote
        if stripped.startswith('>'):
            text = stripped.lstrip('>').strip()
            parsed.append({
                "index": idx,
                "type": "blockquote",
                "text": text
            })
            continue
            
        # HR
        if stripped == '---':
            parsed.append({
                "index": idx,
                "type": "hr"
            })
            continue
            
        # Empty
        if not stripped:
            parsed.append({
                "index": idx,
                "type": "empty"
            })
            continue
            
        # Plain text
        parsed.append({
            "index": idx,
            "type": "text",
            "text": stripped
        })
        
    return parsed

def toggle_task(line_idx):
    if not os.path.exists(ROADMAP_PATH):
        return False, "File not found"
        
    with open(ROADMAP_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if line_idx < 0 or line_idx >= len(lines):
        return False, "Invalid line index"
        
    line = lines[line_idx]
    task_match = re.match(r'^(\s*)-\s*\[([ xX])\]\s*(.*)$', line)
    if not task_match:
        return False, "Line is not a task"
        
    indent = task_match.group(1)
    current_status = task_match.group(2)
    text = task_match.group(3)
    
    new_status = ' ' if current_status.lower() == 'x' else 'x'
    new_line = f"{indent}- [{new_status}] {text}\n"
    lines[line_idx] = new_line
    
    with open(ROADMAP_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    return True, new_status == 'x'

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SilentVoice Interactive Roadmap</title>
    <!-- Outfit Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- FontAwesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 28, 47, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-blue: #00d2ff;
            --accent-purple: #9d4edd;
            --accent-green: #00f5d4;
            --text-main: #f8f9fa;
            --text-muted: #94a3b8;
            --text-glow: 0 0 10px rgba(0, 210, 255, 0.5);
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 210, 255, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(157, 78, 221, 0.05) 0%, transparent 40%);
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }

        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-weight: 800;
            font-size: 2.2rem;
            margin: 0 0 10px 0;
            background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tagline {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin: 0 0 20px 0;
            font-style: italic;
        }

        /* Progress Bar */
        .progress-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            height: 24px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
        }

        .progress-text {
            position: absolute;
            width: 100%;
            text-align: center;
            font-weight: 700;
            font-size: 0.9rem;
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
            z-index: 10;
        }

        /* Filter Controls */
        .filters {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            justify-content: center;
        }

        .filter-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .filter-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-main);
        }

        .filter-btn.active {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
            border-color: transparent;
            box-shadow: 0 0 12px rgba(0, 210, 255, 0.3);
        }

        /* Content List */
        .content-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 25px 0 10px 0;
            color: #fff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 15px 0 5px 0;
            color: var(--accent-purple);
        }

        .blockquote {
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid var(--accent-blue);
            padding: 12px 20px;
            margin: 10px 0;
            border-radius: 0 12px 12px 0;
            color: var(--text-muted);
        }

        .text-line {
            color: var(--text-muted);
            line-height: 1.6;
        }

        .hr-line {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            margin: 20px 0;
        }

        /* Task Item styling */
        .task-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid transparent;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .task-item:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.05);
            transform: translateX(4px);
        }

        .task-item.checked {
            background: rgba(0, 245, 212, 0.03);
        }

        .task-item.checked:hover {
            background: rgba(0, 245, 212, 0.05);
        }

        /* Custom Checkbox */
        .checkbox-container {
            width: 22px;
            height: 22px;
            border: 2px solid var(--text-muted);
            border-radius: 6px;
            margin-right: 14px;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.25s ease;
            background: transparent;
            flex-shrink: 0;
        }

        .task-item.checked .checkbox-container {
            border-color: var(--accent-green);
            background: var(--accent-green);
            box-shadow: 0 0 10px rgba(0, 245, 212, 0.4);
        }

        .checkbox-icon {
            color: #0b0f19;
            font-size: 0.8rem;
            opacity: 0;
            transform: scale(0.5);
            transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .task-item.checked .checkbox-icon {
            opacity: 1;
            transform: scale(1);
        }

        /* Task Text */
        .task-text {
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.25s ease;
            color: var(--text-main);
        }

        .task-item.checked .task-text {
            color: var(--text-muted);
            text-decoration: line-through;
        }
        
        /* Indents */
        .indent-4 { margin-left: 20px; }
        .indent-8 { margin-left: 40px; }
        .indent-12 { margin-left: 60px; }

        /* Notification toast */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(20, 28, 47, 0.9);
            border: 1px solid var(--accent-blue);
            color: #fff;
            padding: 12px 24px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            backdrop-filter: blur(8px);
            z-index: 1000;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        /* Loading spinner */
        .loader {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: var(--accent-blue);
            animation: spin 1s ease-in-out infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤟 SilentVoice</h1>
            <p class="tagline">Giving Every Gesture a Voice.</p>
        </header>

        <div class="progress-container">
            <div class="progress-text">Roadmap Progress: <span id="progress-pct">0%</span></div>
            <div class="progress-bar" id="progress-bar"></div>
        </div>

        <div class="filters">
            <button class="filter-btn active" onclick="setFilter('all')">All</button>
            <button class="filter-btn" onclick="setFilter('pending')">Pending</button>
            <button class="filter-btn" onclick="setFilter('completed')">Completed</button>
        </div>

        <div id="roadmap-content">
            <div class="loading-container">
                <div class="loader"></div>
                <p>Loading roadmap...</p>
            </div>
        </div>
    </div>

    <div class="toast" id="toast">Saved successfully!</div>

    <script>
        let currentFilter = 'all';
        let roadmapData = [];

        async function fetchRoadmap() {
            try {
                const response = await fetch('/api/roadmap');
                roadmapData = await response.json();
                renderRoadmap();
            } catch (err) {
                console.error("Failed to fetch roadmap:", err);
                document.getElementById('roadmap-content').innerHTML = `
                    <div style="text-align: center; color: #ff6b6b; padding: 20px;">
                        <i class="fa-solid fa-circle-exclamation fa-2x mb-2"></i>
                        <p>Error loading roadmap. Please make sure the server is running.</p>
                    </div>
                `;
            }
        }

        function renderRoadmap() {
            const container = document.getElementById('roadmap-content');
            container.innerHTML = '';
            
            let totalTasks = 0;
            let completedTasks = 0;

            roadmapData.forEach(item => {
                if (item.type === 'task') {
                    totalTasks++;
                    if (item.checked) completedTasks++;
                }
            });

            // Update global progress
            const progressPct = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
            document.getElementById('progress-pct').innerText = progressPct + '%';
            document.getElementById('progress-bar').style.width = progressPct + '%';

            let listContainer = document.createElement('div');
            listContainer.className = 'content-list';

            roadmapData.forEach(item => {
                if (item.type === 'task') {
                    if (currentFilter === 'pending' && item.checked) return;
                    if (currentFilter === 'completed' && !item.checked) return;

                    const row = document.createElement('div');
                    row.className = `task-item ${item.checked ? 'checked' : ''} indent-${item.indent || 0}`;
                    row.onclick = () => toggleItem(item.index);

                    row.innerHTML = `
                        <div class="checkbox-container">
                            <i class="fa-solid fa-check checkbox-icon"></i>
                        </div>
                        <span class="task-text">${item.text}</span>
                    `;
                    listContainer.appendChild(row);
                } else if (item.type === 'header') {
                    const el = document.createElement(`h${Math.min(item.level + 1, 6)}`);
                    el.innerText = item.text;
                    listContainer.appendChild(el);
                } else if (item.type === 'blockquote') {
                    const el = document.createElement('div');
                    el.className = 'blockquote';
                    el.innerText = item.text;
                    listContainer.appendChild(el);
                } else if (item.type === 'hr') {
                    const el = document.createElement('hr');
                    el.className = 'hr-line';
                    listContainer.appendChild(el);
                } else if (item.type === 'text') {
                    const el = document.createElement('p');
                    el.className = 'text-line';
                    el.innerText = item.text;
                    listContainer.appendChild(el);
                }
            });

            container.appendChild(listContainer);
        }

        async function toggleItem(index) {
            try {
                const response = await fetch('/api/toggle', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index })
                });
                const result = await response.json();
                if (result.success) {
                    const item = roadmapData.find(i => i.index === index);
                    if (item) {
                        item.checked = result.checked;
                        renderRoadmap();
                        showToast(result.checked ? "Task marked complete!" : "Task marked pending!");
                    }
                } else {
                    showToast("Error: " + result.error);
                }
            } catch (err) {
                console.error("Error toggling item:", err);
                showToast("Connection error. Could not update task.");
            }
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            renderRoadmap();
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.className = 'show';
            setTimeout(() => {
                toast.classList.remove('show');
            }, 2500);
        }

        // Initial fetch
        fetchRoadmap();
    </script>
</body>
</html>
"""

class RoadmapHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
        
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/api/roadmap':
            data = parse_roadmap()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            
    def do_POST(self):
        if self.path == '/api/toggle':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                params = json.loads(post_data.decode('utf-8'))
                line_idx = params.get('index')
                if line_idx is not None:
                    success, checked_or_error = toggle_task(line_idx)
                    if success:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "checked": checked_or_error}).encode('utf-8'))
                        return
                    else:
                        error_msg = checked_or_error
                else:
                    error_msg = "Missing line index"
            except Exception as e:
                error_msg = str(e)
                
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server = HTTPServer(('127.0.0.1', PORT), RoadmapHandler)
    url = f"http://127.0.0.1:{PORT}"
    print("\n=============================================")
    print("SilentVoice Interactive Roadmap has started!")
    print(f"URL: {url}")
    print(f"Editing file: {ROADMAP_PATH}")
    print("Press Ctrl+C in this terminal to stop.")
    print("=============================================\n")
    
    # Auto open browser
    webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down interactive roadmap server...")
        server.server_close()

if __name__ == '__main__':
    run_server()
