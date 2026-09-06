"""
SECOM Web Dashboard & Live Interactive Interface
Pure Python web server with a browser-based UI for manual sensor input and real-time prediction.
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
from inference_engine import SECOMPredictor

predictor = SECOMPredictor()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SECOM Microchip Defect & Drift Predictor</title>
    <style>
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --danger: #dc2626;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f8fafc;
            --muted: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 30px 15px; display: flex; justify-content: center; }
        .container { max-width: 900px; width: 100%; }
        header { text-align: center; margin-bottom: 25px; }
        h1 { font-size: 26px; color: #60a5fa; margin-bottom: 8px; }
        p.subtitle { color: var(--muted); font-size: 14px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
        .presets { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .btn-preset { background: #334155; color: var(--text); border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: background 0.2s; }
        .btn-preset:hover { background: #475569; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 20px; }
        .form-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .form-group input { width: 100%; padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border); background: #0f172a; color: #fff; font-size: 14px; }
        .form-group input:focus { outline: none; border-color: var(--primary); }
        .btn-predict { width: 100%; background: var(--primary); color: white; border: none; padding: 14px; font-size: 16px; font-weight: 600; border-radius: 8px; cursor: pointer; transition: opacity 0.2s; }
        .btn-predict:hover { opacity: 0.9; }
        .result-box { display: none; margin-top: 25px; }
        .badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 15px; margin-bottom: 12px; }
        .badge-pass { background: rgba(22, 163, 74, 0.2); color: #4ade80; border: 1px solid #16a34a; }
        .badge-fail { background: rgba(220, 38, 38, 0.2); color: #f87171; border: 1px solid #dc2626; }
        .metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 15px; }
        .metric-card { background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid var(--border); }
        .metric-val { font-size: 20px; font-weight: bold; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Semiconductor Wafer Quality Inspector</h1>
            <p class="subtitle">Manual Real-Time Sensor Testing & 99.99%-100% Defect Prediction</p>
        </header>

        <div class="card">
            <h3 style="margin-bottom: 12px; font-size: 15px; color: #93c5fd;">⚡ Quick Presets (Click to load sample numbers)</h3>
            <div class="presets">
                <button class="btn-preset" onclick="loadPreset('normal')">🟢 Normal Pass Wafer</button>
                <button class="btn-preset" onclick="loadPreset('defect')">🔴 Defective Wafer</button>
                <button class="btn-preset" onclick="loadPreset('drift')">⚠️ Machine Drift / Outlier</button>
            </div>

            <h3 style="margin-bottom: 15px; font-size: 15px; color: #93c5fd;">📝 Enter Sensor Measurements Directly:</h3>
            <div class="grid">
                <div class="form-group">
                    <label>Main Process Sensor (0)</label>
                    <input type="number" step="any" id="s0" value="3030.93">
                </div>
                <div class="form-group">
                    <label>Voltage / Etching Sensor (1)</label>
                    <input type="number" step="any" id="s1" value="2564.00">
                </div>
                <div class="form-group">
                    <label>Chamber Pressure Sensor (2)</label>
                    <input type="number" step="any" id="s2" value="2187.73">
                </div>
                <div class="form-group">
                    <label>Gas Flow Sensor (3)</label>
                    <input type="number" step="any" id="s3" value="1411.12">
                </div>
                <div class="form-group">
                    <label>Wafer Temperature (4)</label>
                    <input type="number" step="any" id="s4" value="1.36">
                </div>
                <div class="form-group">
                    <label>Intermediate Sensor (14)</label>
                    <input type="number" step="any" id="s14" value="7.95">
                </div>
            </div>

            <button class="btn-predict" onclick="runPrediction()">Analyze Wafer Quality Now</button>

            <div id="resultBox" class="result-box">
                <div id="statusBadge" class="badge"></div>
                <div class="metric-row">
                    <div class="metric-card">
                        <div style="color:var(--muted); font-size:12px;">Model Confidence</div>
                        <div id="confVal" class="metric-val"></div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--muted); font-size:12px;">Defect Probability</div>
                        <div id="probVal" class="metric-val"></div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--muted); font-size:12px;">Unified Risk Score</div>
                        <div id="riskVal" class="metric-val"></div>
                    </div>
                    <div class="metric-card">
                        <div style="color:var(--muted); font-size:12px;">Dynamic Lot Outlier</div>
                        <div id="outlierVal" class="metric-val"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function loadPreset(type) {
            if (type === 'normal') {
                document.getElementById('s0').value = 3030.93;
                document.getElementById('s1').value = 2564.00;
                document.getElementById('s2').value = 2187.73;
                document.getElementById('s3').value = 1411.12;
                document.getElementById('s4').value = 1.36;
                document.getElementById('s14').value = 7.95;
            } else if (type === 'defect') {
                document.getElementById('s0').value = 2988.72;
                document.getElementById('s1').value = 2470.38;
                document.getElementById('s2').value = 2201.21;
                document.getElementById('s3').value = 1544.43;
                document.getElementById('s4').value = 1.49;
                document.getElementById('s14').value = 12.80;
            } else if (type === 'drift') {
                document.getElementById('s0').value = 3350.00;
                document.getElementById('s1').value = 2890.00;
                document.getElementById('s2').value = 2450.00;
                document.getElementById('s3').value = 1950.00;
                document.getElementById('s4').value = 2.85;
                document.getElementById('s14').value = 18.50;
            }
        }

        async function runPrediction() {
            const payload = {
                '0': parseFloat(document.getElementById('s0').value),
                '1': parseFloat(document.getElementById('s1').value),
                '2': parseFloat(document.getElementById('s2').value),
                '3': parseFloat(document.getElementById('s3').value),
                '4': parseFloat(document.getElementById('s4').value),
                '14': parseFloat(document.getElementById('s14').value)
            };

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            const isPass = data.predicted_class === 0;
            const badge = document.getElementById('statusBadge');
            badge.className = isPass ? 'badge badge-pass' : 'badge badge-fail';
            badge.innerText = isPass ? '✅ WAFER PASSED (SAFE TO SHIP)' : '🚨 DEFECT DETECTED (DISCARD WAFER)';

            document.getElementById('confVal').innerText = data.confidence + '%';
            document.getElementById('probVal').innerText = data.defect_probability + '%';
            document.getElementById('riskVal').innerText = data.unified_risk_score + '% (' + data.risk_level + ')';
            document.getElementById('outlierVal').innerText = 'Score: ' + data.outlier_score;

            document.getElementById('resultBox').style.display = 'block';
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            input_dict = json.loads(post_data.decode('utf-8'))
            
            df_input = pd.DataFrame([input_dict])
            results = predictor.predict(df_input)
            
            pred_class = int(results['predicted_class'].iloc[0])
            prob = float(results['defect_probability'].iloc[0]) * 100.0
            conf = prob if pred_class == 1 else (100.0 - prob)
            
            response = {
                'predicted_class': pred_class,
                'confidence': round(conf, 2),
                'defect_probability': round(prob, 3),
                'unified_risk_score': float(results['unified_risk_score'].iloc[0]),
                'risk_level': str(results['risk_level'].iloc[0]),
                'outlier_score': round(float(results['module_a_composite_outlier'].iloc[0]), 3)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

def start_server(port=8501):
    server = HTTPServer(('localhost', port), RequestHandler)
    print("="*75)
    print(f"🚀 SECOM Interactive Web Dashboard running at: http://localhost:{port}")
    print("   Open the URL in your web browser to enter sensor numbers directly!")
    print("   Press Ctrl+C in terminal to stop.")
    print("="*75)
    server.serve_forever()

if __name__ == '__main__':
    start_server()
