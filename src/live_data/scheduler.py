"""
Scheduler - Automated data collection and prediction generation

Runs:
- Match scraping every 6 hours
- Odds collection every 15 minutes
- Prediction generation every 30 minutes
- Database cleanup daily
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_data.match_scraper import get_all_upcoming_matches
from live_data.odds_scraper import get_tennis_odds
from live_predictions.predictor import LivePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TennisBettingScheduler:
    """Automated scheduler for tennis betting system"""
    
    def __init__(self, bankroll: float = 1000):
        self.scheduler = BackgroundScheduler()
        self.predictor = LivePredictor(bankroll=bankroll)
        self.latest_predictions = None
        self.latest_bets = None
    
    def start(self):
        """Start the scheduler"""
        
        logger.info("🚀 Starting Tennis Betting Scheduler...")
        
        # Job 1: Scrape matches every 6 hours
        self.scheduler.add_job(
            func=self.scrape_matches_job,
            trigger=IntervalTrigger(hours=6),
            id='match_scraper',
            name='Scrape upcoming matches',
            replace_existing=True
        )
        logger.info("✅ Scheduled: Match scraping (every 6 hours)")
        
        # Job 2: Update odds every 15 minutes
        self.scheduler.add_job(
            func=self.update_odds_job,
            trigger=IntervalTrigger(minutes=15),
            id='odds_updater',
            name='Update betting odds',
            replace_existing=True
        )
        logger.info("✅ Scheduled: Odds updates (every 15 minutes)")
        
        # Job 3: Generate predictions every 30 minutes
        self.scheduler.add_job(
            func=self.generate_predictions_job,
            trigger=IntervalTrigger(minutes=30),
            id='prediction_generator',
            name='Generate predictions',
            replace_existing=True
        )
        logger.info("✅ Scheduled: Prediction generation (every 30 minutes)")
        
        # Job 4: Daily cleanup at 2 AM
        self.scheduler.add_job(
            func=self.cleanup_job,
            trigger=CronTrigger(hour=2, minute=0),
            id='daily_cleanup',
            name='Daily database cleanup',
            replace_existing=True
        )
        logger.info("✅ Scheduled: Daily cleanup (2:00 AM)")
        
        # Job 5: Alert on high-value bets (every 10 minutes)
        self.scheduler.add_job(
            func=self.check_alerts_job,
            trigger=IntervalTrigger(minutes=10),
            id='alert_checker',
            name='Check for high-value bets',
            replace_existing=True
        )
        logger.info("✅ Scheduled: Alert checking (every 10 minutes)")
        
        # Start scheduler
        self.scheduler.start()
        logger.info("🎾 Scheduler started successfully!")
        
        # Run initial jobs
        logger.info("\nRunning initial data collection...")
        self.scrape_matches_job()
        self.update_odds_job()
        self.generate_predictions_job()
    
    def scrape_matches_job(self):
        """Scheduled job: Scrape upcoming matches"""
        
        try:
            logger.info("📡 Scraping upcoming matches...")
            matches = get_all_upcoming_matches(days_ahead=3)
            
            logger.info(f"✅ Found {len(matches)} upcoming matches")
            
            # TODO: Store in database
            
        except Exception as e:
            logger.error(f"❌ Match scraping failed: {e}")
    
    def update_odds_job(self):
        """Scheduled job: Update betting odds"""
        
        try:
            logger.info("💰 Updating betting odds...")
            odds = get_tennis_odds(use_api=True)
            
            logger.info(f"✅ Updated odds for {len(odds)} matches")
            
            # TODO: Store in database with timestamp
            
        except Exception as e:
            logger.error(f"❌ Odds update failed: {e}")
    
    def generate_predictions_job(self):
        """Scheduled job: Generate predictions"""
        
        try:
            logger.info("🔮 Generating predictions...")
            
            all_preds, profitable_bets = self.predictor.predict_upcoming_matches(
                days_ahead=2,
                use_odds_api=True
            )
            
            self.latest_predictions = all_preds
            self.latest_bets = profitable_bets
            
            logger.info(f"✅ Generated {len(all_preds)} predictions")
            
            if not profitable_bets.empty:
                logger.info(f"🎯 Found {len(profitable_bets)} profitable bets!")
                
                # Save to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                profitable_bets.to_csv(f'auto_bets_{timestamp}.csv', index=False)
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
    
    def check_alerts_job(self):
        """Scheduled job: Check for high-value betting opportunities"""
        
        try:
            if self.latest_bets is None or self.latest_bets.empty:
                return
            
            # Check for high-value bets (>8% edge)
            high_value = self.latest_bets[self.latest_bets['edge'] > 0.08]
            
            if not high_value.empty:
                logger.warning("🚨 HIGH VALUE ALERT!")
                
                for _, bet in high_value.iterrows():
                    logger.warning(f"   {bet['player1_name']} vs {bet['player2_name']}")
                    logger.warning(f"   Edge: {bet['edge']:.2%}, EV: {bet['expected_value']:.2%}")
                    logger.warning(f"   Recommended stake: ${bet['recommended_stake']:.2f}")
                
                # TODO: Send email/SMS/Slack notification
        
        except Exception as e:
            logger.error(f"❌ Alert check failed: {e}")
    
    def cleanup_job(self):
        """Scheduled job: Clean up old data"""
        
        try:
            logger.info("🧹 Running daily cleanup...")
            
            # TODO: Remove old predictions
            # TODO: Archive completed matches
            # TODO: Vacuum database
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        
        logger.info("🛑 Stopping scheduler...")
        self.scheduler.shutdown()
        logger.info("✅ Scheduler stopped")
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        
        jobs = self.scheduler.get_jobs()
        
        status = {
            'running': self.scheduler.running,
            'num_jobs': len(jobs),
            'jobs': []
        }
        
        for job in jobs:
            status['jobs'].append({
                'id': job.id,
                'name': job.name,
                'next_run': str(job.next_run_time) if job.next_run_time else 'Paused'
            })
        
        return status


def run_scheduler(bankroll: float = 1000):
    """
    Run the automated scheduler
    
    Args:
        bankroll: Starting bankroll
    """
    
    scheduler = TennisBettingScheduler(bankroll=bankroll)
    scheduler.start()
    
    print("\n" + "="*80)
    print("🎾 TENNIS BETTING SCHEDULER RUNNING")
    print("="*80)
    print("\nScheduled Jobs:")
    
    status = scheduler.get_status()
    for job in status['jobs']:
        print(f"  • {job['name']}")
        print(f"    Next run: {job['next_run']}")
    
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep running
        while True:
            import time
            time.sleep(60)
            
            # Print status every hour
            if datetime.now().minute == 0:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')} - Scheduler running")
                
                if scheduler.latest_bets is not None and not scheduler.latest_bets.empty:
                    print(f"   Current opportunities: {len(scheduler.latest_bets)} profitable bets")
    
    except KeyboardInterrupt:
        print("\n\nStopping scheduler...")
        scheduler.stop()
        print("✅ Scheduler stopped successfully")


if __name__ == "__main__":
    # Run scheduler with $1000 bankroll
    run_scheduler(bankroll=1000)
