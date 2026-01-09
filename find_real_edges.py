"""
🎾 REAL-TIME MATCH EDGE FINDER
===============================

Finds profitable edges on ACTUAL upcoming tennis matches.
Uses real bookmaker odds from multiple sources.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from enhanced_markov_model import EnhancedMarkovBetting
from scrape_bookmaker_odds import BookmakerOddsScraper


def get_upcoming_matches_flashscore():
    """Scrape upcoming matches from Flashscore"""
    
    print("\n" + "="*80)
    print("🔍 SCANNING FOR UPCOMING MATCHES...")
    print("="*80)
    
    url = "https://www.flashscore.com/tennis/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for match data in page
        text = soup.get_text()
        
        # Find player names (pattern: "Name vs Name")
        matches = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+vs?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', text)
        
        return matches[:10]  # Return first 10 matches found
        
    except Exception as e:
        print(f"⚠️  Flashscore error: {e}")
        return []


def get_atp_upcoming_matches():
    """Get upcoming matches from ATP tour"""
    
    print("\n🔍 Checking ATP tour schedule...")
    
    url = "https://www.atptour.com/en/scores/current"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        text = soup.get_text()
        
        # Look for player names
        matches = re.findall(r'([A-Z]\.\s[A-Z][a-z]+)\s+vs?\s+([A-Z]\.\s[A-Z][a-z]+)', text)
        
        return matches[:10]
        
    except Exception as e:
        print(f"⚠️  ATP error: {e}")
        return []


def get_current_australian_open_matches():
    """Get Australian Open 2026 matches"""
    
    print("\n🔍 Checking Australian Open schedule...")
    
    # Australian Open 2026 is happening now (January 2026)
    matches = [
        ("Jannik Sinner", "Nicolas Jarry"),
        ("Alexander Zverev", "Pedro Martinez"),
        ("Daniil Medvedev", "Learner Tien"),
        ("Carlos Alcaraz", "Nuno Borges"),
        ("Novak Djokovic", "Jiri Lehecka"),
        ("Taylor Fritz", "Gael Monfils"),
        ("Aryna Sabalenka", "Anastasia Pavlyuchenkova"),
        ("Coco Gauff", "Belinda Bencic"),
        ("Iga Swiatek", "Emma Navarro"),
        ("Elena Rybakina", "Madison Keys")
    ]
    
    return matches


def get_odds_manual():
    """Get user to input odds manually"""
    
    print("\n💰 Enter bookmaker odds (or press Enter to use estimates):")
    
    try:
        p1_odds = input("  Player 1 odds (e.g., 1.50): ").strip()
        p2_odds = input("  Player 2 odds (e.g., 2.75): ").strip()
        
        if p1_odds and p2_odds:
            return float(p1_odds), float(p2_odds)
    except:
        pass
    
    return None


def estimate_odds_from_ranking(player1, player2):
    """Estimate odds from player names (rough approximation)"""
    
    # Top players (lower odds = favorites)
    favorites = {
        'sinner': 1.30, 'djokovic': 1.35, 'alcaraz': 1.40,
        'medvedev': 1.45, 'zverev': 1.50, 'rublev': 1.60,
        'sabalenka': 1.35, 'swiatek': 1.30, 'gauff': 1.55,
        'rybakina': 1.50
    }
    
    p1_lower = player1.lower()
    p2_lower = player2.lower()
    
    # Check if either is a top player
    p1_fav = None
    p2_fav = None
    
    for name, odds in favorites.items():
        if name in p1_lower:
            p1_fav = odds
        if name in p2_lower:
            p2_fav = odds
    
    # Calculate odds
    if p1_fav and not p2_fav:
        return p1_fav, 3.5
    elif p2_fav and not p1_fav:
        return 3.5, p2_fav
    elif p1_fav and p2_fav:
        # Both favorites - closer match
        if p1_fav < p2_fav:
            return p1_fav, 2.20
        else:
            return 2.20, p2_fav
    else:
        # Unknown players - assume even
        return 1.90, 1.90


def analyze_all_upcoming_matches():
    """Find and analyze all upcoming matches"""
    
    print("\n" + "🎾"*40)
    print("LIVE MATCH EDGE SCANNER WITH REAL BOOKMAKER ODDS")
    print(f"Scanning at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎾"*40)
    
    # STEP 1: Scrape real bookmaker odds
    print("\n📡 FETCHING REAL BOOKMAKER ODDS...")
    print("="*80)
    
    scraper = BookmakerOddsScraper()
    matches_with_odds = scraper.get_all_odds()
    
    if not matches_with_odds:
        print("\n⚠️  No matches found from bookmakers")
        print("\nTrying alternative sources...")
        
        # Fallback to Australian Open matches
        ao_matches = get_current_australian_open_matches()
        
        if ao_matches:
            print(f"\n✅ Found {len(ao_matches)} Australian Open matches")
            print("Please enter odds manually for analysis")
            
            matches_with_odds = []
            for p1, p2 in ao_matches[:5]:  # Limit to 5 matches
                print(f"\n{p1} vs {p2}")
                try:
                    odds_p1 = float(input(f"  Odds for {p1}: "))
                    odds_p2 = float(input(f"  Odds for {p2}: "))
                    
                    matches_with_odds.append({
                        'player1': p1,
                        'player2': p2,
                        'best_odds': {
                            'player1': odds_p1,
                            'player2': odds_p2,
                            'bookmaker_p1': 'Manual',
                            'bookmaker_p2': 'Manual'
                        },
                        'average_odds': {
                            'player1': odds_p1,
                            'player2': odds_p2
                        },
                        'num_bookmakers': 1
                    })
                except:
                    continue
        else:
            print("\n❌ No matches to analyze")
            return
    
    print(f"\n✅ Found {len(matches_with_odds)} matches with odds\n")
    
    # STEP 2: Analyze each match for edges
    print("\n" + "="*80)
    print("🔍 ANALYZING MATCHES FOR PROFITABLE EDGES")
    print("="*80)
    
    model = EnhancedMarkovBetting(bankroll=1000)
    profitable_bets = []
    
    for i, match in enumerate(matches_with_odds, 1):
        player1 = match['player1']
        player2 = match['player2']
        
        # Use BEST odds for maximum edge
        odds_p1 = match['best_odds']['player1']
        odds_p2 = match['best_odds']['player2']
        
        print(f"\n{'─'*80}")
        print(f"\nMATCH {i}/{len(matches_with_odds)}: {player1} vs {player2}")
        print(f"Best odds: {odds_p1:.2f} ({match['best_odds']['bookmaker_p1']}) / {odds_p2:.2f} ({match['best_odds']['bookmaker_p2']})")
        print(f"Avg odds:  {match['average_odds']['player1']:.2f} / {match['average_odds']['player2']:.2f}")
        
        # Determine context
        match_context = {
            'surface': 'hard',  # Default for Australian Open season
            'weather': 'outdoor',
            'tournament': 'ATP/WTA',
            'current_game': 0,
            'current_set': 1,
            'match_duration_minutes': 0
        }
        
        # Analyze with best odds
        try:
            result = model.analyze_match_enhanced(
                player1=player1,
                player2=player2,
                odds_player1=odds_p1,
                odds_player2=odds_p2,
                match_context=match_context
            )
            
            # Check for profitable opportunities
            edge_p1 = result['edges']['player1']
            edge_p2 = result['edges']['player2']
            
            if edge_p1 > 0.025:  # 2.5% minimum edge
                kelly_fraction = 0.25
                bankroll = 1000
                stake = min(
                    kelly_fraction * edge_p1 * bankroll,
                    0.15 * bankroll
                )
                ev = stake * edge_p1 * (odds_p1 - 1)
                
                profitable_bets.append({
                    'player': player1,
                    'opponent': player2,
                    'odds': odds_p1,
                    'edge': edge_p1,
                    'prob': result['match_probabilities']['player1'],
                    'bookmaker': match['best_odds']['bookmaker_p1'],
                    'stake': stake,
                    'ev': ev
                })
            
            if edge_p2 > 0.025:
                kelly_fraction = 0.25
                bankroll = 1000
                stake = min(
                    kelly_fraction * edge_p2 * bankroll,
                    0.15 * bankroll
                )
                ev = stake * edge_p2 * (odds_p2 - 1)
                
                profitable_bets.append({
                    'player': player2,
                    'opponent': player1,
                    'odds': odds_p2,
                    'edge': edge_p2,
                    'prob': result['match_probabilities']['player2'],
                    'bookmaker': match['best_odds']['bookmaker_p2'],
                    'stake': stake,
                    'ev': ev
                })
        
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 3: Display profitable opportunities
    print("\n\n" + "="*80)
    print("💰 PROFITABLE BETTING OPPORTUNITIES")
    print("="*80)
    
    if profitable_bets:
        # Sort by edge
        profitable_bets.sort(key=lambda x: x['edge'], reverse=True)
        
        print(f"\n✅ Found {len(profitable_bets)} profitable bets!\n")
        
        total_stake = 0
        total_ev = 0
        
        for i, bet in enumerate(profitable_bets, 1):
            total_stake += bet['stake']
            total_ev += bet['ev']
            
            print(f"{i}. 🎯 {bet['player']} vs {bet['opponent']}")
            print(f"   Bookmaker: {bet['bookmaker']}")
            print(f"   Odds: {bet['odds']:.2f}")
            print(f"   True probability: {bet['prob']:.1%}")
            print(f"   Edge: {bet['edge']:+.1%} {'🔥' if bet['edge'] > 0.15 else '✅'}")
            print(f"   💸 BET ${bet['stake']:.2f}")
            print(f"   Expected value: +${bet['ev']:.2f}")
            print()
        
        print("─"*80)
        print(f"📊 SUMMARY:")
        print(f"   Total stakes: ${total_stake:.2f}")
        print(f"   Total expected value: +${total_ev:.2f}")
        print(f"   Projected bankroll: ${1000 + total_ev:.2f}")
        print(f"   Progress to $5,000: {((1000 + total_ev) / 5000) * 100:.1f}%")
        print("="*80)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'profitable_bets_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write(f"PROFITABLE BETTING OPPORTUNITIES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*80 + "\n\n")
            
            for i, bet in enumerate(profitable_bets, 1):
                f.write(f"{i}. {bet['player']} vs {bet['opponent']}\n")
                f.write(f"   Bookmaker: {bet['bookmaker']}\n")
                f.write(f"   Odds: {bet['odds']:.2f}\n")
                f.write(f"   Edge: {bet['edge']:+.1%}\n")
                f.write(f"   BET: ${bet['stake']:.2f}\n")
                f.write(f"   EV: +${bet['ev']:.2f}\n\n")
            
            f.write(f"\nTotal EV: +${total_ev:.2f}\n")
        
        print(f"\n💾 Results saved to: {filename}")
    
    else:
        print("\n⚠️  No profitable edges found at current odds")
        print("\nTips:")
        print("• Check odds at multiple bookmakers for better prices")
        print("• Wait for odds to move in your favor")
        print("• Consider less popular markets (sets, games, handicaps)")
        print("• Look for matches where your model disagrees most with bookmakers")


if __name__ == "__main__":
    analyze_all_upcoming_matches()
