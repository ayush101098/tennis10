"""
🎾 VIDEO ANALYSIS MODEL SETUP
==============================

Downloads and sets up pre-trained models from TennisProject repo:
1. Court detection model (14 key points)
2. Ball tracking model (TrackNet)
3. Bounce detection model (CatBoost)
4. Player detection model (Faster R-CNN)
"""

import os
import subprocess
import urllib.request
from pathlib import Path


class VideoModelSetup:
    """Setup video analysis models"""
    
    def __init__(self, models_dir: str = "video_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.repos = {
            'tennis_project': 'https://github.com/yastrebksv/TennisProject.git',
            'court_detector': 'https://github.com/yastrebksv/TennisCourtDetector.git',
            'tracknet': 'https://github.com/yastrebksv/TrackNet.git'
        }
    
    def clone_repositories(self):
        """Clone required GitHub repositories"""
        
        print("\n" + "="*80)
        print("📥 CLONING REPOSITORIES")
        print("="*80)
        
        for name, url in self.repos.items():
            repo_path = self.models_dir / name
            
            if repo_path.exists():
                print(f"\n✓ {name} already exists")
                continue
            
            print(f"\n📦 Cloning {name}...")
            print(f"   {url}")
            
            try:
                subprocess.run(
                    ['git', 'clone', url, str(repo_path)],
                    check=True,
                    capture_output=True
                )
                print(f"✅ {name} cloned successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to clone {name}: {e}")
    
    def install_dependencies(self):
        """Install required Python packages"""
        
        print("\n" + "="*80)
        print("📦 INSTALLING DEPENDENCIES")
        print("="*80)
        
        packages = [
            'tensorflow>=2.8.0',
            'keras>=2.8.0',
            'opencv-python>=4.5.0',
            'catboost>=1.0.0',
            'scikit-learn>=1.0.0',
            'pillow>=9.0.0'
        ]
        
        print("\nInstalling packages:")
        for pkg in packages:
            print(f"  - {pkg}")
        
        try:
            subprocess.run(
                ['pip', 'install'] + packages,
                check=True
            )
            print("\n✅ All dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install dependencies: {e}")
    
    def download_pretrained_models(self):
        """Download pre-trained model weights"""
        
        print("\n" + "="*80)
        print("📥 DOWNLOADING PRE-TRAINED MODELS")
        print("="*80)
        
        # Note: The actual model weights need to be trained or downloaded
        # from the repositories. The repos contain training scripts.
        
        print("\n⚠️  MODEL WEIGHTS:")
        print("="*80)
        print("The repositories contain training code but not pre-trained weights.")
        print("\nYou have 2 options:")
        print("\n1. TRAIN MODELS (Recommended for best accuracy):")
        print("   cd video_models/court_detector")
        print("   python train.py --data path/to/dataset")
        print("\n   cd video_models/tracknet")
        print("   python train.py --data path/to/dataset")
        print("\n2. USE TRANSFER LEARNING:")
        print("   Use pre-trained ImageNet weights and fine-tune")
        print("\n3. DOWNLOAD COMMUNITY MODELS:")
        print("   Check GitHub issues/releases for shared weights")
        print("="*80)
    
    def create_integration_config(self):
        """Create configuration file for model integration"""
        
        config = {
            'court_detector': {
                'model_path': str(self.models_dir / 'court_detector' / 'model.h5'),
                'input_size': (640, 360),
                'num_keypoints': 14
            },
            'ball_tracker': {
                'model_path': str(self.models_dir / 'tracknet' / 'model.h5'),
                'input_size': (640, 360),
                'num_frames': 3
            },
            'bounce_detector': {
                'model_path': str(self.models_dir / 'bounce' / 'model.cbm'),
                'features': ['dx', 'dy', 'velocity', 'acceleration']
            },
            'player_detector': {
                'model': 'faster_rcnn',
                'pretrained': True
            }
        }
        
        config_path = self.models_dir / 'config.py'
        
        with open(config_path, 'w') as f:
            f.write("# Video Analysis Model Configuration\n\n")
            f.write(f"CONFIG = {config}\n")
        
        print(f"\n✅ Configuration saved to: {config_path}")
    
    def run_setup(self):
        """Run complete setup"""
        
        print("\n" + "🎾"*40)
        print("VIDEO ANALYSIS MODEL SETUP")
        print("🎾"*40)
        
        # Step 1: Clone repositories
        self.clone_repositories()
        
        # Step 2: Install dependencies
        self.install_dependencies()
        
        # Step 3: Download models (or show instructions)
        self.download_pretrained_models()
        
        # Step 4: Create config
        self.create_integration_config()
        
        print("\n" + "="*80)
        print("✅ SETUP COMPLETE")
        print("="*80)
        
        print("\nNEXT STEPS:")
        print("1. Train models using datasets (see repos for instructions)")
        print("2. Or use pre-trained weights if available")
        print("3. Update model paths in video_models/config.py")
        print("4. Run: python video_analysis_integration.py")
        print("\n" + "="*80 + "\n")


def main():
    setup = VideoModelSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()
