"""One-off script to find ATP leaderboard data source."""
import re
import requests

url = "https://www.atptour.com/en/stats/leaderboard?boardType=serve&timeFrame=52week&surface=all&versusRank=all&formerNo1=false"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
text = r.text

# Extract script tag contents that might have data
for script in re.findall(r"<script[^>]*>([^<]+)</script>", text):
    if "ServeRating" in script or "leaderboard" in script or "PlayerName" in script:
        print("SCRIPT (len %d):" % len(script))
        print(script[:2000])
        print("...")
        break
else:
    print("No script with ServeRating/leaderboard/PlayerName found")
