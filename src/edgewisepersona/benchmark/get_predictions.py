from pathlib import Path
import sys
import json
import random
import copy
import logging
import argparse
from typing import List, Dict, Tuple

from edgewisepersona.benchmark.utils import get_model


def parse_args():
    """Parse command line arguments."""
    available_models = ['gpt', 'deepseek', 'gemini', 'qwen', 'phi4', 'llama', 'gemma3']

    parser = argparse.ArgumentParser(description='Run benchmark evaluation for smart home routine inference.')
    parser.add_argument(
        '--n_predictions',
        type=int,
        default=0,
        help='Number of predictions to make, if 0 - number of routines in the dataset'
    )
    parser.add_argument('--model', type=str, choices=available_models, required=True, help='Model to use for inference')
    parser.add_argument('--sessions_file', type=Path, required=True, help='Sessions file')
    parser.add_argument('--routines_file', type=Path, required=True, help='Routines file')
    parser.add_argument('--output_file', type=Path, default=Path("outputs/predictions.jsonl"), help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_data(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict]]:
    assert args.sessions_file.exists(), f"Sessions file not found: {args.sessions_file}"
    assert args.routines_file.exists(), f"Routines file not found: {args.routines_file}"

    with open(args.routines_file, 'r') as f:
        routines = [json.loads(line) for line in f.readlines()]
        routines = [session['routines'] for session in routines]

    with open(args.sessions_file, 'r') as f:
        sessions = [json.loads(line) for line in f.readlines()]
        sessions = [session['sessions'] for session in sessions]

    assert len(routines) == len(sessions), f"Mismatch between routines and sessions count: {len(routines)} != {len(sessions)}"
    return sessions, routines


def evaluate_model(
        args: argparse.Namespace,
        sessions: List[Dict],
        routines: List[Dict],
        model_name: str,
    ) -> Dict[str, List[float]]:
    logger = logging.getLogger()
    evaluator = get_model(model_name)

    # Check for existing predictions
    processed_users = 0
    if args.output_file.exists():
        with open(args.output_file, 'r') as f:
            processed_users = sum(1 for _ in f)
        logger.info(f"Found {processed_users} existing predictions in {args.output_file}")
    
    # Open output file for appending
    with open(args.output_file, "a") as f:
        for user_idx, user_history in enumerate(sessions):
            # Skip already processed users
            if user_idx < processed_users:
                logger.info(f"Skipping user {user_idx} - already processed")
                continue

            try:
                # Prepare session data without applied routines
                user_sessions_no_routines = copy.deepcopy(user_history)
                for session in user_sessions_no_routines:
                    del session['applied_routines']

                # Get model predictions
                n_routines = len(routines[user_idx])
                n_predictions = args.n_predictions if args.n_predictions > 0 else n_routines
                retrieved_routines = evaluator.infer_routines(
                    json.dumps(user_sessions_no_routines, indent=2),
                    n_routines=n_predictions,
                    max_routines=n_routines
                )
                
                # Save predictions immediately
                f.write(json.dumps(retrieved_routines) + "\n")
                f.flush()  # Ensure data is written to disk

                # Calculate and store metrics
                n_predicted = len(retrieved_routines)
                logger.info(f"User {user_idx}: {n_predicted} predicted routines (num_gt={n_routines})")

            except Exception as e:
                logger.error(f"Error processing user {user_idx}: {str(e)}")
                # Write empty list for failed prediction to maintain alignment with ground truth
                f.write("[]\n")
                f.flush()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger()
    logger.info(f"Starting benchmark evaluation with {args.model.upper()} model")

    # Load and preprocess data
    logger.info("Loading data...")
    sessions, routines = load_data(args)
    logger.info(f"Loaded {len(sessions)} sessions and {len(routines)} routine sets")

    # Create output directory and file
    logger.info(f"Saving predictions to {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate model and save predictions iteratively
    logger.info("Starting model evaluation...")
    evaluate_model(args, sessions, routines, args.model)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()