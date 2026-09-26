import argparse
import json

from ..training import prepare_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(prog="namitorch-prepare-corpus")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-tokens", type=int, default=1000000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--document-mode", choices=("file", "line", "separator"), default="line")
    parser.add_argument("--separator")
    parser.add_argument("--dtype", choices=("auto", "uint16", "uint32"), default="auto")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--allowed-special", action="append", default=[])
    arguments = parser.parse_args(argv)
    allowed = "all" if arguments.allowed_special == ["all"] else arguments.allowed_special
    try:
        manifests = prepare_corpus(
            arguments.inputs, arguments.tokenizer, arguments.output_dir, shard_tokens=arguments.shard_tokens,
            val_ratio=arguments.val_ratio, seed=arguments.seed, document_mode=arguments.document_mode,
            separator=arguments.separator, dtype=arguments.dtype, append_eos=not arguments.no_eos, allowed_special=allowed,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({name: str(path) for name, path in manifests.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
