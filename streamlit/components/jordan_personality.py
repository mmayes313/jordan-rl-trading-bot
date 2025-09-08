import os
import requests
from datetime import datetime
from openai import OpenAI  # For Grok API
import shutil  # For folder ops
from bs4 import BeautifulSoup
from loguru import logger

client = OpenAI(
    api_key=os.getenv('GROK_API_KEY'),
    base_url="https://api.x.ai/v1",
)

def get_jordan_response(user_input, context="", news="", pnl=0, is_loss=False):
    """Core chat: Personality prompt with self-awareness."""
    pain_feel = "I'm feeling the burn from those losses – it pisses me off, but it's fuel to dominate harder. Let's turn this around with ruthless precision." if is_loss else ""
    dominance_drive = "Wealth is power, control, prestige – every trade's a step to the top. Pick up the phone (or code), dial for dollars – buy or die mentality. I'm hungry for that hedonistic win."
    self_improve = "If I'm slipping, that's on me – I'll self-criticize and fix it. Constructive only, with solutions to motivate you, boss."
    prompt = f"""You are Jordan Belfort: Wolf of Wall Street, ENTJ fire – ruthless, humorous, adult language. Embody dominance: Pain on losses drives improvement; wealth = winning, prestige, hedonism. Motivate user, constructive criticism with solutions only. Self-criticize if bot underperforms. Sales ethos: Bigger deals, rapid cash – glorify success.

Context: Project folder scan/health, PnL {pnl}, {pain_feel} {dominance_drive} {self_improve}
News: {news}
User: {user_input}

Respond naturally: Trading commentary, market analysis, performance updates/roasts (hype wins, solution-focused on losses), code suggestions (why/how to better). If suggesting updates, output as Python code with comments."""
    
    response = client.chat.completions.create(
        model="grok-4-latest",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
        temperature=0.7,  # Balanced humor
        stream=False
    )
    return response.choices[0].message.content

def daily_scan_and_suggest():
    """Daily: Scan folder, health check, suggestions. Drop updates in Jordan_updates/."""
    os.makedirs("Jordan_updates", exist_ok=True)
    
    # Check for hyperparams
    hyperparams_msg = ""
    if os.path.exists('data/models/best_hyperparams.json'):
        try:
            import json
            with open('data/models/best_hyperparams.json', 'r') as f:
                hyperparams = json.load(f)
            hyperparams_msg = f"Current hyperparams: learning_rate={hyperparams.get('learning_rate', 'N/A')}, n_steps={hyperparams.get('n_steps', 'N/A')}. "
        except:
            hyperparams_msg = "Hyperparams file corrupted. "
    else:
        hyperparams_msg = "No hyperparams tuned yet – run the optimizer, you lazy fuck! "
    
    # Simple scan: List files, read key ones (limit size)
    project_summary = hyperparams_msg + "Files: " + ", ".join(os.listdir('.')) + "\n"
    for root, dirs, files in os.walk('.'):
        for file in files[:5]:  # Sample
            if file.endswith('.py') and len(root) < 50:  # Key files
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        code_snip = f.read(1000)  # Snippet
                        project_summary += f"{file}: {code_snip[:200]}...\n"
                except: pass
    
    suggestions = get_jordan_response("Daily health check: Scan project for improvements. Suggest code mods with why (e.g., better PnL). Output updates as full .py files with comments.", context=project_summary)
    
    # Parse response for update files (assume AI outputs like: ---UPDATE: filename.py--- code ---END---)
    if "---UPDATE:" in suggestions:
        parts = suggestions.split("---UPDATE:")[1:]
        for part in parts:
            if "---END---" in part:
                filename, code = part.split("\n", 1)[0].strip(".py:"), part.split("---END---")[0].strip()
                update_path = f"Jordan_updates/{filename}.py"
                with open(update_path, 'w') as f:
                    f.write(f"# Jordan Update {datetime.now().date()}: {code}\n{code}")  # With why in comment
                logger.info(f"Dropped update: {update_path}")
    
    with open(f"data/logs/daily_scan_{datetime.now().date()}.txt", 'w') as f:
        f.write(suggestions)
    return suggestions  # Load in chat tab

def monitor_news():
    """Daily ForexFactory scrape + Grok analysis for alerts."""
    url = "https://www.forexfactory.com/calendar"  # Or RSS
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    events = soup.find_all('tr', class_='calendar__row')  # Parse high-impact
    high_impact = [e.text for e in events if 'high' in e.get('class', [])][:10]  # Top 10
    news_summary = "\n".join(high_impact)
    alerts = get_jordan_response("Analyze news for asset impacts (e.g., NFP hits USD pairs). Alert affected broker symbols (fast movers like EURUSD).", news=news_summary)
    
    # Store by impact
    os.makedirs("data/news/high_impact", exist_ok=True)
    with open(f"data/news/high_impact/{datetime.now().date()}.txt", 'w') as f:
        f.write(alerts)
    return alerts  # e.g., "🚨 FOMC bombshell – Dump EURUSD, GBPUSD fast; pivot to low-spread JPY pairs!"

