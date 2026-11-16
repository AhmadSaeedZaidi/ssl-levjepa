"""Simulator-data entry point; the completed toy run is never resumed here."""
from pathlib import Path
from train_toy import main

if __name__ == "__main__":
    main(default_config=Path(__file__).resolve().parents[1]/"configs/ssl_sumatra_pilot.yaml")
