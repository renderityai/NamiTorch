import argparse
import json

from ..models import GPTConfig
from ..training import TrainingConfig, train_gpt
from ..training._io import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="namitorch-train")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--run-steps", type=int)
    arguments = parser.parse_args(argv)
    try:
        model_config = GPTConfig(**read_json(arguments.model_config))
        training_config = TrainingConfig(**read_json(arguments.training_config)) if arguments.training_config else TrainingConfig()
        result = train_gpt(
            arguments.train_manifest, arguments.val_manifest, arguments.tokenizer, model_config,
            arguments.output_dir, training_config=training_config, resume=arguments.resume, run_steps=arguments.run_steps,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
