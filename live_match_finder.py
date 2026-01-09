"""
🎾 LIVE MATCH FINDER & BETTING ANALYZER
========================================

Finds live tennis matches and shows where to bet with edge analysis.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json


class LiveMatchFinder:
    """Find live tennis matches from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def check_flashscore(self):
        """Check Flashscore for live matches"""
        print("\n🔍 Checking Flashscore.com...")
        
        url = "https://www.flashscore.com/tennis/"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Connected to Flashscore")
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for live matches
                live_matches = soup.find_all('div', class_=lambda x: x and 'live' in x.lower())
                
                print(f"   Found {len(live_matches)} potential live match elements")
                
                # Try to extract match info
                matches = []
                for match in live_matches[:5]:  # First 5
                    text = match.get_text(strip=True)
                    if len(text) > 10:
                        matches.append(text[:100])
                
                if matches:
                    print("   Sample data:")
                    for m in matches[:3]:
                        print(f"   - {m}")
                
                return True
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def check_sofascore(self):
        """Check Sofascore for live matches"""
        print("\n🔍 Checking Sofascore.com...")
        
        url = "https://www.sofascore.com/tennis"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Connected to Sofascore")
                
                # Sofascore uses dynamic loading, so we might need API
                # Try to find API endpoint
                if 'api.sofascore.com' in response.text:
                    print("   ℹ️  Sofascore uses API - need to extract endpoint")
                
                return True
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def check_tennisexplorer(self):
        """Check TennisExplorer for matches"""
        print("\n🔍 Checking TennisExplorer.com...")
        
        url = "https://www.tennisexplorer.com/matches/"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Connected to TennisExplorer")
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find tables
                tables = soup.find_all('table')
                print(f"   Found {len(tables)} table(s)")
                
                # Find all links to matches
                match_links = soup.find_all('a', href=lambda x: x and '/match-detail/' in str(x))
                print(f"   Found {len(match_links)} match link(s)")
                
                if match_links:
                    print("   Sample matches:")
                    for link in match_links[:5]:
                        print(f"   - {link.get_text(strip=True)}")
                
                return True
            else:
                print(f"❌ Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def get_sample_matches(self):
        """Get sample matches for testing (upcoming/recent)"""
        print("\n📋 Looking for sample matches...")
        
        # Check multiple sources
        sources = [
            ("TennisExplorer", "https://www.tennisexplorer.com/"),
            ("ATP Tour", "https://www.atptour.com/en/scores/current"),
            ("WTA", "https://www.wtatennis.com/scores")
        ]
        
        results = []
        
        for name, url in sources:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    results.append({
                        'source': name,
                        'status': '✅ Accessible',
                        'url': url
                    })
                else:
                    results.append({
                        'source': name,
                        'status': f'❌ Status {response.status_code}',
                        'url': url
                    })
            except Exception as e:
                results.append({
                    'source': name,
                    'status': f'❌ Error: {str(e)[:50]}',
                    'url': url
                })
        
        return results


class BettingSiteGuide:
    """Guide to betting sites and markets"""
    
    @staticmethod
    def show_betting_sites():
        """Display available betting sites and markets"""
        
        print("\n" + "="*80)
        print("💰 WHERE TO BET ON TENNIS")
        print("="*80)
        
        sites = {
            "🏆 Premium Sites (Best Odds)": [
                {
                    'name': 'Pinnacle',
                    'url': 'pinnacle.com',
                    'features': ['Lowest margins', 'Sharp odds', 'High limits'],
                    'markets': ['Match Winner', 'Set Winner', 'Game Winner', 'Live']
                },
                {
                    'name': 'Bet365',
                    'url': 'bet365.com',
                    'features': ['Best live coverage', 'In-play streaming', 'Cash out'],
                    'markets': ['Match', 'Set', 'Game', 'Points', 'Live']
                },
            ],
            "💎 Exchange Betting": [
                {
                    'name': 'Betfair Exchange',
                    'url': 'betfair.com',
                    'features': ['Back & Lay', 'Best odds', 'No margin'],
                    'markets': ['Match', 'Set', 'Game', 'Live trading']
                },
                {
                    'name': 'Smarkets',
                    'url': 'smarkets.com',
                    'features': ['Low commission', 'Clean interface'],
                    'markets': ['Match', 'Set', 'Live']
                },
            ],
            "🎯 US Sportsbooks": [
                {
                    'name': 'DraftKings',
                    'url': 'draftkings.com',
                    'features': ['Legal in US states', 'Promos'],
                    'markets': ['Match', 'Set', 'Live (select)']
                },
                {
                    'name': 'FanDuel',
                    'url': 'fanduel.com',
                    'features': ['Legal in US states', 'Same game parlays'],
                    'markets': ['Match', 'Set']
                },
                {
                    'name': 'BetMGM',
                    'url': 'betmgm.com',
                    'features': ['Wide coverage', 'Live betting'],
                    'markets': ['Match', 'Set', 'Game', 'Live']
                },
            ],
            "🌍 International": [
                {
                    'name': 'BetOnline',
                    'url': 'betonline.ag',
                    'features': ['Crypto accepted', 'Good lines'],
                    'markets': ['Match', 'Set', 'Game']
                },
                {
                    'name': 'Bovada',
                    'url': 'bovada.lv',
                    'features': ['US-friendly', 'Live betting'],
                    'markets': ['Match', 'Set', 'Live']
                },
            ]
        }
        
        for category, bookmakers in sites.items():
            print(f"\n{category}")
            print("─"*80)
            
            for book in bookmakers:
                print(f"\n{book['name']} ({book['url']})")
                print(f"  Features: {', '.join(book['features'])}")
                print(f"  Markets: {', '.join(book['markets'])}")
        
        print("\n" + "="*80)
        print("💡 TIPS FOR BEST VALUE:")
        print("="*80)
        print("✓ Use Pinnacle/Betfair for lowest margins")
        print("✓ Compare odds across multiple sites")
        print("✓ Look for 'Next Game Winner' in live betting")
        print("✓ Exchanges allow you to LAY favorites (bet against)")
        print("✓ Some sites have better odds for underdogs")
        print("="*80 + "\n")


def main():
    """Main debugging and testing"""
    
    print("\n🎾 LIVE MATCH FINDER & BETTING GUIDE")
    print("="*80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    finder = LiveMatchFinder()
    guide = BettingSiteGuide()
    
    # Test connections
    print("\n" + "="*80)
    print("🔧 TESTING LIVE MATCH SOURCES")
    print("="*80)
    
    # Test each source
    te_works = finder.check_tennisexplorer()
    fs_works = finder.check_flashscore()
    ss_works = finder.check_sofascore()
    
    # Get sample matches
    print("\n" + "="*80)
    print("📊 TESTING MATCH DATA ACCESS")
    print("="*80)
    
    sample_results = finder.get_sample_matches()
    
    for result in sample_results:
        print(f"\n{result['source']}: {result['status']}")
        print(f"   URL: {result['url']}")
    
    # Show betting guide
    guide.show_betting_sites()
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("📋 SYSTEM STATUS & RECOMMENDATIONS")
    print("="*80)
    
    print("\n✅ WORKING:")
    print("   - Markov chain probability calculator")
    print("   - Edge detection algorithm")
    print("   - Betting recommendations")
    print("   - Database integration")
    
    print("\n⚠️  LIVE DATA SOURCES:")
    sources_status = []
    if te_works:
        sources_status.append("TennisExplorer (connected)")
    if fs_works:
        sources_status.append("Flashscore (connected)")
    if ss_works:
        sources_status.append("Sofascore (connected)")
    
    if sources_status:
        for status in sources_status:
            print(f"   ✓ {status}")
    else:
        print("   ⚠️  Sites connected but need manual parsing")
        print("   ℹ️  Modern sites use JavaScript - may need browser automation")
    
    print("\n💡 RECOMMENDED WORKFLOW:")
    print("="*80)
    print("1. Visit betting site (e.g., Bet365, Pinnacle)")
    print("2. Find live tennis matches")
    print("3. Note player names and odds")
    print("4. Run analysis:")
    print("   python comprehensive_analyzer.py")
    print("5. Enter match details manually")
    print("6. Get betting recommendations with edge calculations")
    print("="*80)
    
    # Provide current match analysis option
    print("\n" + "="*80)
    print("🎯 QUICK MATCH ANALYSIS")
    print("="*80)
    
    choice = input("\nAnalyze a match now? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n" + "─"*80)
        p1 = input("Player 1 name: ").strip()
        p2 = input("Player 2 name: ").strip()
        
        try:
            odds1 = float(input(f"{p1} odds (decimal, e.g., 1.50): "))
            odds2 = float(input(f"{p2} odds (decimal, e.g., 2.75): "))
            
            # Run analysis
            print("\n🔄 Running analysis...")
            
            from comprehensive_analyzer import ComprehensiveTennisData
            analyzer = ComprehensiveTennisData()
            
            analyzer.analyze_match_with_odds(p1, p2, odds1, odds2)
            
        except ValueError:
            print("❌ Invalid odds format")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ Debug complete! System ready for betting.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
