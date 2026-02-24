from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ModelBuilderV2, ModelBuilderV2Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmallBizPulse v2 methodology models")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/external/yelp_dataset_new"),
        help="Path to Yelp dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/v2_artifacts"),
        help="Directory where v2 artifacts will be written.",
    )
    parser.add_argument(
        "--skip-topic-model",
        action="store_true",
        help="Skip BERTopic diagnostics (Component 2).",
    )
    parser.add_argument(
        "--skip-recommendations",
        action="store_true",
        help="Skip topic-to-recommendation mapping (Component 3).",
    )
    parser.add_argument(
        "--skip-resilience",
        action="store_true",
        help="Skip resilience/vulnerability outputs (Component 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelBuilderV2Config(
        output_root=args.output_root,
        run_topic_model=not args.skip_topic_model,
        run_recommendation_mapping=not args.skip_recommendations,
        run_resilience_analysis=not args.skip_resilience,
    )

    builder = ModelBuilderV2(config)
    artifacts = builder.run(data_root=args.data_root)

    print(f"v2 build complete. Summary: {artifacts.run_summary_path}")


if __name__ == "__main__":
    main()
