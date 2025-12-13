"""
Simple Runner Script for Survival Analysis
==========================================

Usage:
    python run_survival_analysis.py --mode training --duration_col survival_time --event_col event
    python run_survival_analysis.py --mode validation --model_path models/cox_model.pkl
"""

import argparse
from pathlib import Path
from src.survival_analysis import train_survival_model, validate_survival_model


def main():
    parser = argparse.ArgumentParser(
        description="Train or validate Cox PH survival model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model with default column names (survival_time, event)
  python run_survival_analysis.py --mode training
  
  # Train a model with custom column names
  python run_survival_analysis.py --mode training --duration_col my_time --event_col my_event
  
  # Train and save with prediction years
  python run_survival_analysis.py --mode training --prediction_years 1 2 3 5 --save_model models/my_model.pkl
  
  # Validate a trained model
  python run_survival_analysis.py --mode validation --model_path models/cox_model.pkl --prediction_years 1 2 3 5
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "validation"],
        required=True,
        help="Mode: 'training' to train a new model or 'validation' to validate an existing model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="IQ_Cancer_Endometrio_merged_NMSP.xlsx",
        help="Path to Excel data file"
    )
    parser.add_argument(
        "--duration_col",
        type=str,
        default="survival_time",
        help="Name of the duration/survival time column (for training mode)"
    )
    parser.add_argument(
        "--event_col",
        type=str,
        default="event",
        help="Name of the event indicator column (for training mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (required for validation mode, optional for training)"
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="Regularization strength (for training mode)"
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.0,
        help="L1 ratio for elastic net (0=L2, 1=L1) (for training mode)"
    )
    parser.add_argument(
        "--prediction_years",
        type=float,
        nargs="+",
        default=None,
        help="List of years for survival probability prediction (e.g., --prediction_years 1 2 3 5)"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save the fitted model (for training mode). "
             "If not specified, defaults to models/cox_model.pkl"
    )
    
    args = parser.parse_args()
    
    if args.mode == "training":
        # Training mode
        if args.save_model is None:
            args.save_model = f"models/cox_model.pkl"
        
        print(f"\n{'='*70}")
        print(f"TRAINING MODE")
        print(f"{'='*70}")
        
        results = train_survival_model(
            data_path=args.data,
            duration_col=args.duration_col,
            event_col=args.event_col,
            penalizer=args.penalizer,
            l1_ratio=args.l1_ratio,
            prediction_years=args.prediction_years,
            save_model=args.save_model
        )
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"\nModel saved to: {args.save_model}")
        print(f"\nAccess results via:")
        print(f"  - results['model']: Fitted Cox PH model")
        print(f"  - results['predictions']['risk_scores']: Risk scores")
        print(f"  - results['predictions']['survival_probabilities']: Survival probabilities")
        print(f"  - results['summary']: Model summary statistics")
        print(f"  - results['hazard_ratios']: Hazard ratios with confidence intervals")
    
    elif args.mode == "validation":
        # Validation mode
        if args.model_path is None:
            parser.error("--model_path is required for validation mode")
        
        print(f"\n{'='*70}")
        print(f"VALIDATION MODE")
        print(f"{'='*70}")
        
        results = validate_survival_model(
            data_path=args.data,
            model_path=args.model_path,
            prediction_years=args.prediction_years
        )
        
        print("\n" + "="*70)
        print("Validation Complete!")
        print("="*70)
        print(f"\nAccess results via:")
        print(f"  - results['risk_scores']: Risk scores (numpy array)")
        print(f"  - results['survival_probabilities']: Survival probabilities (DataFrame)")
        print(f"\nExample:")
        print(f"  risk_scores = results['risk_scores']")
        print(f"  survival_at_1yr = results['survival_probabilities'][1.0]")


if __name__ == "__main__":
    main()
