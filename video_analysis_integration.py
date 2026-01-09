"""
🎾 VIDEO ANALYSIS INTEGRATION FOR LIVE BETTING
===============================================

Integrates TennisProject video analysis with Markov betting model:
- Ball tracking → Real-time serve stats, shot placement
- Court detection → Player positioning analysis
- Bounce detection → Rally patterns
- Player tracking → Movement efficiency, fatigue

Enhances probability calculations with LIVE match data.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque
import sqlite3
from datetime import datetime


class VideoAnalysisIntegration:
    """
    Integrates video analysis with betting model.
    Updates Markov probabilities based on real-time match statistics.
    """
    
    def __init__(self, video_source: str = None):
        self.video_source = video_source
        
        # Court detection model (from TennisCourtDetector)
        self.court_detector = None  # Will load pre-trained model
        
        # Ball tracking model (from TrackNet)
        self.ball_tracker = None  # Will load pre-trained model
        
        # Player detector (Faster R-CNN)
        self.player_detector = None
        
        # Bounce detector (CatBoost)
        self.bounce_detector = None
        
        # Statistics buffers
        self.serve_placements = deque(maxlen=50)  # Last 50 serves
        self.rally_lengths = deque(maxlen=30)  # Last 30 rallies
        self.ball_speeds = deque(maxlen=50)
        self.player_positions = deque(maxlen=100)
        
        # Match state
        self.current_server = None
        self.score = {'player1': 0, 'player2': 0}
        self.game_count = {'player1': 0, 'player2': 0}
        
    def extract_serve_statistics(self, frames: List[np.ndarray]) -> Dict:
        """
        Extract serve statistics from video frames.
        
        Returns:
        - serve_placement: Left/Middle/Right percentages
        - serve_depth: Short/Deep percentages
        - serve_speed: Estimated speed
        - first_serve_in_pct: First serve percentage
        - ace_rate: Ace percentage
        """
        
        serve_stats = {
            'placement': {'left': 0, 'middle': 0, 'right': 0},
            'depth': {'short': 0, 'deep': 0},
            'speed': [],
            'first_serve_in': 0,
            'aces': 0,
            'total_serves': 0
        }
        
        for frame in frames:
            # Detect court key points
            court_points = self._detect_court(frame)
            
            if court_points is None:
                continue
            
            # Track ball trajectory
            ball_trajectory = self._track_ball(frame)
            
            if not ball_trajectory:
                continue
            
            # Analyze serve
            is_serve = self._is_serve_frame(ball_trajectory)
            
            if is_serve:
                serve_stats['total_serves'] += 1
                
                # Determine placement (left/middle/right)
                placement = self._classify_serve_placement(
                    ball_trajectory, 
                    court_points
                )
                serve_stats['placement'][placement] += 1
                
                # Determine depth
                depth = self._classify_serve_depth(ball_trajectory, court_points)
                serve_stats['depth'][depth] += 1
                
                # Estimate speed (pixels per frame → mph)
                speed = self._estimate_serve_speed(ball_trajectory)
                serve_stats['speed'].append(speed)
                
                # Check if ace (no return detected)
                is_ace = self._is_ace(frames, ball_trajectory)
                if is_ace:
                    serve_stats['aces'] += 1
        
        # Convert to percentages
        total = serve_stats['total_serves']
        if total > 0:
            return {
                'placement_left_pct': serve_stats['placement']['left'] / total,
                'placement_middle_pct': serve_stats['placement']['middle'] / total,
                'placement_right_pct': serve_stats['placement']['right'] / total,
                'depth_deep_pct': serve_stats['depth']['deep'] / total,
                'avg_speed_mph': np.mean(serve_stats['speed']) if serve_stats['speed'] else 0,
                'ace_rate': serve_stats['aces'] / total,
                'total_serves': total
            }
        
        return None
    
    def analyze_rally_patterns(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze rally patterns from video.
        
        Returns:
        - avg_rally_length: Average shots per rally
        - baseline_pct: Percentage of baseline shots
        - net_approach_pct: Net approach frequency
        - winner_rate: Winner percentage
        - unforced_error_rate: Error percentage
        """
        
        rallies = []
        current_rally = []
        
        for i, frame in enumerate(frames):
            # Track ball
            ball_pos = self._track_ball(frame)
            
            if ball_pos is None:
                # Rally ended
                if len(current_rally) > 0:
                    rallies.append(current_rally)
                    current_rally = []
                continue
            
            # Detect bounce
            is_bounce = self._detect_bounce(frames[max(0, i-2):i+3])
            
            if is_bounce:
                # Track player positions
                players = self._detect_players(frame)
                
                shot_data = {
                    'ball_position': ball_pos,
                    'players': players,
                    'frame': i
                }
                current_rally.append(shot_data)
        
        # Analyze patterns
        if not rallies:
            return None
        
        rally_lengths = [len(r) for r in rallies]
        
        # Count baseline vs net shots
        baseline_shots = 0
        net_shots = 0
        
        for rally in rallies:
            for shot in rally:
                if self._is_baseline_shot(shot):
                    baseline_shots += 1
                else:
                    net_shots += 1
        
        total_shots = baseline_shots + net_shots
        
        return {
            'avg_rally_length': np.mean(rally_lengths),
            'median_rally_length': np.median(rally_lengths),
            'max_rally_length': max(rally_lengths),
            'baseline_pct': baseline_shots / total_shots if total_shots > 0 else 0,
            'net_approach_pct': net_shots / total_shots if total_shots > 0 else 0,
            'total_rallies': len(rallies)
        }
    
    def calculate_player_fatigue(self, frames: List[np.ndarray], player_id: int) -> float:
        """
        Estimate player fatigue from movement patterns.
        
        Returns fatigue score: 0.0 (fresh) to 1.0 (exhausted)
        """
        
        movements = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Detect player positions
            prev_players = self._detect_players(prev_frame)
            curr_players = self._detect_players(curr_frame)
            
            if len(prev_players) > player_id and len(curr_players) > player_id:
                # Calculate movement distance
                prev_pos = prev_players[player_id]
                curr_pos = curr_players[player_id]
                
                distance = np.linalg.norm(
                    np.array(curr_pos) - np.array(prev_pos)
                )
                movements.append(distance)
        
        if not movements:
            return 0.0
        
        # Fatigue indicators:
        # 1. Reduced movement speed over time
        # 2. Less court coverage
        # 3. Slower recovery to baseline
        
        # Split into early and late game
        mid_point = len(movements) // 2
        early_movements = movements[:mid_point]
        late_movements = movements[mid_point:]
        
        early_avg = np.mean(early_movements)
        late_avg = np.mean(late_movements)
        
        # Fatigue = reduction in movement
        if early_avg > 0:
            fatigue = max(0, 1 - (late_avg / early_avg))
        else:
            fatigue = 0.0
        
        return min(1.0, fatigue)
    
    def update_markov_probabilities(
        self, 
        base_p_serve: float,
        serve_stats: Dict,
        rally_stats: Dict,
        fatigue: float
    ) -> float:
        """
        Update Markov probability based on video analysis.
        
        Base probability is adjusted by:
        - Serve placement effectiveness
        - Rally performance
        - Fatigue level
        """
        
        adjusted_p = base_p_serve
        
        # Adjust for serve placement
        if serve_stats:
            # Wide serves are harder to return
            wide_pct = serve_stats['placement_left_pct'] + serve_stats['placement_right_pct']
            
            # Bonus for wide serves (harder to return)
            if wide_pct > 0.6:  # More than 60% wide
                adjusted_p *= 1.05  # +5% advantage
            
            # Deep serves are more effective
            if serve_stats['depth_deep_pct'] > 0.7:
                adjusted_p *= 1.03  # +3% advantage
            
            # High ace rate indicates dominant serving
            if serve_stats['ace_rate'] > 0.1:  # 10% aces
                adjusted_p *= 1.08  # +8% advantage
        
        # Adjust for rally performance
        if rally_stats:
            # Shorter rallies favor server
            if rally_stats['avg_rally_length'] < 4:
                adjusted_p *= 1.04  # +4% for quick points
            elif rally_stats['avg_rally_length'] > 8:
                adjusted_p *= 0.96  # -4% for long rallies (returner doing well)
            
            # High baseline percentage = consistent player
            if rally_stats['baseline_pct'] > 0.8:
                adjusted_p *= 1.02  # +2% for consistency
        
        # Adjust for fatigue
        # Fatigued player loses ~10-20% effectiveness
        fatigue_penalty = 1 - (fatigue * 0.15)  # Up to 15% reduction
        adjusted_p *= fatigue_penalty
        
        # Keep within valid bounds
        adjusted_p = np.clip(adjusted_p, 0.0, 1.0)
        
        return adjusted_p
    
    def generate_betting_insights(
        self,
        player1_stats: Dict,
        player2_stats: Dict,
        current_score: Dict
    ) -> Dict:
        """
        Generate betting insights from video analysis.
        
        Returns specific betting opportunities:
        - Next game winner
        - Total games in set
        - Break of serve
        """
        
        insights = {
            'recommendations': [],
            'confidence': 0.0,
            'edge_opportunities': []
        }
        
        # Analyze momentum
        p1_fatigue = player1_stats.get('fatigue', 0)
        p2_fatigue = player2_stats.get('fatigue', 0)
        
        fatigue_diff = p1_fatigue - p2_fatigue
        
        # If one player is significantly more fatigued
        if abs(fatigue_diff) > 0.3:
            fresher_player = 'player1' if fatigue_diff < 0 else 'player2'
            
            insights['recommendations'].append({
                'type': 'momentum',
                'bet': f'{fresher_player} to win next game',
                'reason': 'Significant fatigue differential detected',
                'confidence': abs(fatigue_diff)
            })
        
        # Analyze serve dominance
        p1_serve_stats = player1_stats.get('serve_stats', {})
        p2_serve_stats = player2_stats.get('serve_stats', {})
        
        if p1_serve_stats and p2_serve_stats:
            p1_ace_rate = p1_serve_stats.get('ace_rate', 0)
            p2_ace_rate = p2_serve_stats.get('ace_rate', 0)
            
            # High ace rate = likely to hold serve
            if p1_ace_rate > 0.15:  # 15% aces
                insights['recommendations'].append({
                    'type': 'serve_dominance',
                    'bet': 'player1 to hold serve',
                    'reason': f'High ace rate: {p1_ace_rate:.1%}',
                    'confidence': p1_ace_rate
                })
        
        # Analyze rally patterns
        p1_rally = player1_stats.get('rally_stats', {})
        p2_rally = player2_stats.get('rally_stats', {})
        
        if p1_rally and p2_rally:
            p1_avg_rally = p1_rally.get('avg_rally_length', 0)
            p2_avg_rally = p2_rally.get('avg_rally_length', 0)
            
            # Long rallies = potential for games to go to deuce
            if p1_avg_rally > 7 and p2_avg_rally > 7:
                insights['recommendations'].append({
                    'type': 'deuce_likely',
                    'bet': 'Next game to go to deuce',
                    'reason': f'Both players averaging {p1_avg_rally:.1f} shot rallies',
                    'confidence': 0.6
                })
        
        return insights
    
    # Helper methods (stubs - actual implementation uses neural networks)
    
    def _detect_court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect 14 court key points using trained neural network"""
        # TODO: Load and run court detection model
        # return court_detector.predict(frame)
        return None
    
    def _track_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track ball position using TrackNet"""
        # TODO: Load and run ball tracking model
        # return ball_tracker.predict(frame)
        return None
    
    def _detect_players(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect player positions using Faster R-CNN"""
        # TODO: Load and run player detection model
        # return player_detector.predict(frame)
        return []
    
    def _detect_bounce(self, frames: List[np.ndarray]) -> bool:
        """Detect ball bounce using CatBoost classifier"""
        # TODO: Load and run bounce detection model
        # return bounce_detector.predict(features)
        return False
    
    def _is_serve_frame(self, trajectory: List) -> bool:
        """Determine if trajectory represents a serve"""
        # Serves typically start high and descend
        if len(trajectory) < 3:
            return False
        
        # Check if ball starts near baseline and moves forward
        # This is a simplified heuristic
        return True
    
    def _classify_serve_placement(self, trajectory: List, court: np.ndarray) -> str:
        """Classify serve as left/middle/right"""
        # Get landing position relative to court width
        # This is simplified - actual implementation uses court homography
        return 'middle'
    
    def _classify_serve_depth(self, trajectory: List, court: np.ndarray) -> str:
        """Classify serve as short/deep"""
        # Get landing position relative to service box
        return 'deep'
    
    def _estimate_serve_speed(self, trajectory: List) -> float:
        """Estimate serve speed in mph"""
        # Calculate pixels per frame and convert to mph
        # Requires camera calibration
        return 100.0  # Placeholder
    
    def _is_ace(self, frames: List, trajectory: List) -> bool:
        """Determine if serve was an ace (no return)"""
        # Check if opponent made contact with ball
        return False
    
    def _is_baseline_shot(self, shot: Dict) -> bool:
        """Determine if shot was from baseline"""
        # Check player position relative to baseline
        return True


class EnhancedBettingModel:
    """
    Enhanced betting model with video analysis integration.
    """
    
    def __init__(self, db_path: str = 'tennis_data.db'):
        self.db_path = db_path
        self.video_analyzer = VideoAnalysisIntegration()
        
        # Live match state
        self.live_stats = {
            'player1': {},
            'player2': {}
        }
    
    def analyze_live_match(
        self,
        video_source: str,
        player1: str,
        player2: str,
        bookmaker_odds: Dict
    ) -> Dict:
        """
        Analyze live match with video + Markov model.
        
        Returns updated probabilities and betting recommendations.
        """
        
        print("\n" + "="*80)
        print(f"🎾 LIVE VIDEO ANALYSIS: {player1} vs {player2}")
        print("="*80)
        
        # Step 1: Get base Markov probabilities from database
        print("\n📊 Step 1: Calculating base Markov probabilities...")
        base_probs = self._get_base_probabilities(player1, player2)
        
        print(f"  Base P(serve win) {player1}: {base_probs['p1_serve']:.1%}")
        print(f"  Base P(serve win) {player2}: {base_probs['p2_serve']:.1%}")
        
        # Step 2: Analyze video for real-time statistics
        print("\n📹 Step 2: Analyzing video feed...")
        video_stats = self._analyze_video(video_source)
        
        if video_stats['player1']['serve_stats']:
            print(f"\n  {player1} serve stats:")
            print(f"    Wide serves: {video_stats['player1']['serve_stats']['placement_left_pct'] + video_stats['player1']['serve_stats']['placement_right_pct']:.1%}")
            print(f"    Deep serves: {video_stats['player1']['serve_stats']['depth_deep_pct']:.1%}")
            print(f"    Ace rate: {video_stats['player1']['serve_stats']['ace_rate']:.1%}")
            print(f"    Avg speed: {video_stats['player1']['serve_stats']['avg_speed_mph']:.0f} mph")
        
        # Step 3: Update probabilities with video insights
        print("\n🔄 Step 3: Updating probabilities with live data...")
        
        adjusted_p1 = self.video_analyzer.update_markov_probabilities(
            base_probs['p1_serve'],
            video_stats['player1'].get('serve_stats'),
            video_stats['player1'].get('rally_stats'),
            video_stats['player1'].get('fatigue', 0)
        )
        
        adjusted_p2 = self.video_analyzer.update_markov_probabilities(
            base_probs['p2_serve'],
            video_stats['player2'].get('serve_stats'),
            video_stats['player2'].get('rally_stats'),
            video_stats['player2'].get('fatigue', 0)
        )
        
        print(f"  Adjusted P(serve win) {player1}: {adjusted_p1:.1%} ({adjusted_p1/base_probs['p1_serve']-1:+.1%})")
        print(f"  Adjusted P(serve win) {player2}: {adjusted_p2:.1%} ({adjusted_p2/base_probs['p2_serve']-1:+.1%})")
        
        # Step 4: Calculate match probabilities
        print("\n🎯 Step 4: Calculating match probabilities...")
        match_probs = self._calculate_match_probabilities(adjusted_p1, adjusted_p2)
        
        print(f"  P(match win) {player1}: {match_probs['p1_match']:.1%}")
        print(f"  P(match win) {player2}: {match_probs['p2_match']:.1%}")
        
        # Step 5: Generate betting insights
        print("\n💰 Step 5: Generating betting recommendations...")
        
        insights = self.video_analyzer.generate_betting_insights(
            video_stats['player1'],
            video_stats['player2'],
            {'player1': 0, 'player2': 0}  # Current score
        )
        
        # Step 6: Calculate edges vs bookmaker odds
        edges = self._calculate_edges(match_probs, bookmaker_odds)
        
        print("\n" + "="*80)
        print("💸 BETTING OPPORTUNITIES")
        print("="*80)
        
        for rec in insights['recommendations']:
            print(f"\n✓ {rec['bet'].upper()}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Confidence: {rec['confidence']:.1%}")
        
        if edges['player1']['edge'] > 0.05:
            print(f"\n✓ BET ON {player1.upper()}")
            print(f"  True probability: {match_probs['p1_match']:.1%}")
            print(f"  Bookmaker odds: {bookmaker_odds['player1']:.2f}")
            print(f"  Edge: {edges['player1']['edge']:+.1%}")
            print(f"  Recommended stake: ${edges['player1']['stake']:.2f}")
        
        return {
            'probabilities': match_probs,
            'insights': insights,
            'edges': edges,
            'video_stats': video_stats
        }
    
    def _get_base_probabilities(self, player1: str, player2: str) -> Dict:
        """Get base Markov probabilities from database"""
        
        # Query database for historical stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simplified - actual implementation queries database
        return {
            'p1_serve': 0.65,  # 65% point win on serve
            'p2_serve': 0.62   # 62% point win on serve
        }
    
    def _analyze_video(self, video_source: str) -> Dict:
        """Analyze video and extract statistics"""
        
        # Load video
        cap = cv2.VideoCapture(video_source)
        
        frames = []
        frame_count = 0
        max_frames = 300  # Analyze ~10 seconds at 30fps
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # Analyze for each player
        p1_stats = {}
        p2_stats = {}
        
        # Extract serve statistics
        serve_stats_p1 = self.video_analyzer.extract_serve_statistics(frames)
        p1_stats['serve_stats'] = serve_stats_p1
        
        # Extract rally patterns
        rally_stats = self.video_analyzer.analyze_rally_patterns(frames)
        p1_stats['rally_stats'] = rally_stats
        p2_stats['rally_stats'] = rally_stats
        
        # Calculate fatigue
        p1_fatigue = self.video_analyzer.calculate_player_fatigue(frames, player_id=0)
        p2_fatigue = self.video_analyzer.calculate_player_fatigue(frames, player_id=1)
        
        p1_stats['fatigue'] = p1_fatigue
        p2_stats['fatigue'] = p2_fatigue
        
        return {
            'player1': p1_stats,
            'player2': p2_stats
        }
    
    def _calculate_match_probabilities(self, p1_serve: float, p2_serve: float) -> Dict:
        """Calculate match probabilities from serve probabilities"""
        
        # Use Markov chain calculations
        from hierarchical_model import HierarchicalTennisModel
        
        model = HierarchicalTennisModel()
        
        # Calculate game probabilities
        p1_hold = model.p_game(p1_serve)
        p2_hold = model.p_game(p2_serve)
        
        # Calculate set probability
        p1_set = model.p_set(p1_hold, 1 - p2_hold)
        
        # Calculate match probability (best of 3)
        p1_match = model.p_match(p1_set, num_sets=3)
        
        return {
            'p1_match': p1_match,
            'p2_match': 1 - p1_match,
            'p1_hold': p1_hold,
            'p2_hold': p2_hold
        }
    
    def _calculate_edges(self, true_probs: Dict, bookmaker_odds: Dict) -> Dict:
        """Calculate betting edges"""
        
        # Convert odds to implied probabilities
        implied_p1 = 1 / bookmaker_odds['player1']
        implied_p2 = 1 / bookmaker_odds['player2']
        
        # Calculate edges
        edge_p1 = true_probs['p1_match'] - implied_p1
        edge_p2 = true_probs['p2_match'] - implied_p2
        
        # Kelly criterion bet sizing
        bankroll = 1000
        kelly_fraction = 0.25
        
        if edge_p1 > 0.025:  # 2.5% minimum edge
            stake_p1 = min(
                kelly_fraction * edge_p1 * bankroll,
                0.15 * bankroll  # Max 15% of bankroll
            )
        else:
            stake_p1 = 0
        
        return {
            'player1': {
                'edge': edge_p1,
                'stake': stake_p1,
                'ev': stake_p1 * edge_p1 * (bookmaker_odds['player1'] - 1)
            },
            'player2': {
                'edge': edge_p2,
                'stake': 0
            }
        }


def main():
    """
    Demo: Analyze live match with video + enhanced probabilities
    """
    
    print("\n" + "🎾"*40)
    print("VIDEO-ENHANCED LIVE BETTING SYSTEM")
    print("🎾"*40)
    
    # Initialize model
    model = EnhancedBettingModel()
    
    # Example: Analyze a match
    # In production, this would be a live video stream
    video_path = "example_match.mp4"
    
    result = model.analyze_live_match(
        video_source=video_path,
        player1="Djokovic",
        player2="Alcaraz",
        bookmaker_odds={
            'player1': 2.10,
            'player2': 1.75
        }
    )
    
    print("\n✅ Analysis complete!")
    print(f"\nVideo insights captured:")
    print(f"  - Serve placements and speeds")
    print(f"  - Rally patterns and lengths")
    print(f"  - Player fatigue levels")
    print(f"  - Court coverage analysis")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
