"""
🎾 EXTRACT LIVE MATCHES FROM TENNISEXPLORER
============================================

Extracts actual match data from the 150 links found.
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime


def extract_tennisexplorer_matches():
    """Extract matches from TennisExplorer"""
    
    print("\n" + "="*80)
    print("🎾 EXTRACTING MATCHES FROM TENNISEXPLORER.COM")
    print("="*80)
    
    url = "https://www.tennisexplorer.com/matches/"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print(f"✅ Page loaded successfully\n")
        
        # Find all tables
        tables = soup.find_all('table', class_=re.compile(r'result|match|schedule', re.I))
        
        if not tables:
            # Try any table
            tables = soup.find_all('table')
        
        print(f"📊 Found {len(tables)} table(s)")
        
        all_matches = []
        
        for table_idx, table in enumerate(tables, 1):
            print(f"\n{'─'*80}")
            print(f"TABLE {table_idx}")
            print(f"{'─'*80}")
            
            # Get table header to identify tournament
            header = table.find_previous(['h2', 'h3', 'div'], class_=re.compile(r'head|title', re.I))
            tournament = header.get_text(strip=True) if header else f"Tournament {table_idx}"
            
            print(f"Tournament: {tournament}")
            
            # Extract rows
            rows = table.find_all('tr')
            
            for row in rows:
                # Look for player names
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 2:
                    continue
                
                # Try to extract match data
                players = []
                score = ""
                time_info = ""
                
                for cell in cells:
                    text = cell.get_text(strip=True)
                    
                    # Check if this looks like a player name
                    if len(text) > 3 and not text.isdigit() and '-' not in text:
                        # Might be a player
                        if any(char.isalpha() for char in text):
                            players.append(text)
                    
                    # Check for score pattern (e.g., "6-4 3-6")
                    if re.match(r'\d+-\d+', text):
                        score = text
                    
                    # Check for time
                    if re.match(r'\d{2}:\d{2}', text):
                        time_info = text
                
                # If we found 2 players, it's likely a match
                if len(players) >= 2:
                    match = {
                        'tournament': tournament,
                        'player1': players[0],
                        'player2': players[1],
                        'score': score if score else "Not started",
                        'time': time_info,
                        'raw_text': ' | '.join([c.get_text(strip=True) for c in cells[:5]])
                    }
                    
                    all_matches.append(match)
        
        # Display matches
        print(f"\n{'='*80}")
        print(f"📋 EXTRACTED {len(all_matches)} MATCHES")
        print(f"{'='*80}\n")
        
        if all_matches:
            for i, match in enumerate(all_matches[:20], 1):  # Show first 20
                status = "🔴 LIVE" if any(x in match['score'].lower() for x in [':', 'ret']) else "⏸️"
                
                print(f"{i}. {status} {match['player1']} vs {match['player2']}")
                print(f"   Tournament: {match['tournament']}")
                print(f"   Score: {match['score']}")
                if match['time']:
                    print(f"   Time: {match['time']}")
                print()
            
            if len(all_matches) > 20:
                print(f"... and {len(all_matches) - 20} more matches")
        else:
            print("⚠️  No matches extracted - page structure may have changed")
            print("\n🔍 Showing raw HTML structure for debugging:")
            
            # Show structure
            for table in tables[:2]:
                print("\n" + "─"*80)
                print("Sample table structure:")
                rows = table.find_all('tr')[:3]
                for row in rows:
                    print(f"  Row: {row.get_text(strip=True)[:100]}")
        
        return all_matches
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def get_betting_recommendations():
    """Show where to bet on these matches"""
    
    print("\n" + "="*80)
    print("💰 WHERE TO BET RIGHT NOW")
    print("="*80)
    
    sites = [
        ("Bet365", "https://www.bet365.com", "✓ Best live coverage, in-play odds"),
        ("Pinnacle", "https://www.pinnacle.com/en/tennis/matchups", "✓ Lowest margins, sharp lines"),
        ("Betfair", "https://www.betfair.com/sport/tennis", "✓ Exchange - back & lay"),
        ("BetMGM", "https://sports.betmgm.com/en/sports/tennis-5", "✓ US legal, live betting"),
        ("DraftKings", "https://sportsbook.draftkings.com/leagues/tennis", "✓ US legal, promos"),
    ]
    
    for name, url, features in sites:
        print(f"\n{name}")
        print(f"  {url}")
        print(f"  {features}")
    
    print("\n" + "="*80)
    print("📝 STEPS TO BET:")
    print("="*80)
    print("1. Go to any site above")
    print("2. Navigate to Tennis → Live or Upcoming")
    print("3. Find a match from the list")
    print("4. Copy player names and odds")
    print("5. Run: python comprehensive_analyzer.py")
    print("6. Enter details for edge analysis")
    print("="*80 + "\n")


def quick_analysis_prompt(matches):
    """Prompt user to analyze a match"""
    
    if not matches:
        return
    
    print("\n" + "="*80)
    print("🎯 ANALYZE A MATCH")
    print("="*80)
    
    print("\nSelect a match to analyze (or press Enter to skip):")
    
    # Show numbered list
    for i, match in enumerate(matches[:10], 1):
        print(f"{i}. {match['player1']} vs {match['player2']} ({match['tournament']})")
    
    choice = input("\nEnter number (1-10) or press Enter to skip: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= min(10, len(matches)):
        match = matches[int(choice) - 1]
        
        print(f"\n{'─'*80}")
        print(f"Selected: {match['player1']} vs {match['player2']}")
        print(f"{'─'*80}")
        
        try:
            odds1 = float(input(f"\nEnter odds for {match['player1']}: "))
            odds2 = float(input(f"Enter odds for {match['player2']}: "))
            
            print("\n🔄 Running Markov analysis...\n")
            
            from comprehensive_analyzer import ComprehensiveTennisData
            analyzer = ComprehensiveTennisData()
            
            analyzer.analyze_match_with_odds(
                match['player1'],
                match['player2'],
                odds1,
                odds2
            )
            
        except ValueError:
            print("❌ Invalid odds")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    
    print("\n" + "🎾"*40)
    print("LIVE TENNIS MATCH EXTRACTOR & ANALYZER")
    print("🎾"*40)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract matches
    matches = extract_tennisexplorer_matches()
    
    # Show betting sites
    get_betting_recommendations()
    
    # Offer to analyze
    if matches:
        quick_analysis_prompt(matches)
    
    print("\n✅ Complete! System ready for profitable betting.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
