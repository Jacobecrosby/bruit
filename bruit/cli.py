# bruit/cli.py
import traceback
import sys
import argparse
from pathlib import Path
from bruit.modules.config_loader import load_config
from bruit.preprocessing import runner as preprocessing_runner

def main():
    run_training = None  # placeholder to supress TensorFlow output
    try:
        parser = argparse.ArgumentParser(prog="bruit", description="BRUIT CLI - A command line interface for BRUIT image processing and analysis")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Preprocess
        prep = subparsers.add_parser("preprocess", help="Run all preprocessing steps on input audio files (.wav)")
        prep.add_argument("--input", "-i", required=True, help="Input image folder")
        prep.add_argument("--quiet","-q", help="Quiet output", action="store_true")

        # train model
        prep = subparsers.add_parser("train", help="Train a model on preprocessed data")
        prep.add_argument("--input", "-i", required=False, help="Preprocessed audio for ML input ")
        prep.add_argument("--quiet","-q", help="Quiet output", action="store_true")

        
        args = parser.parse_args()
        cfg = load_config("configs/parameters.yaml") 

        if args.command == "preprocess":
            preprocessing_runner.run_preprocessing(args.input, config=cfg)
        elif args.command == "train":
            if run_training is None:
                from bruit.train_model import train_model
            train_model.run_training(args.input, quiet=args.quiet)
        #elif args.command == "validate":
        #    if detect is None:
        #        from bruit.validate import runner as validation_runner
        #    validation_runner.run_validation(args.input, config=cfg,quiet=args.quiet)

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ BRUIT CLI crashed: {e} (File: {filename}, Line: {line_number})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ BRUIT CLI crashed: {e} (File: {filename}, Line: {line_number})")
