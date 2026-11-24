"""Convert aggregate Vertex token usage into list USD per 1,000 pages."""

import argparse
from decimal import Decimal
import json

from panel import HERE


def quote(pages, input_tokens, output_tokens, reasoning_tokens, prices):
    if pages <= 0 or min(input_tokens, output_tokens, reasoning_tokens) < 0:
        raise ValueError("pages must be positive and token totals must be nonnegative")
    usage = (input_tokens * prices["input_usd_per_million"] +
             (output_tokens + reasoning_tokens) * prices["output_usd_per_million"])
    return usage / pages / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pages", "input-tokens", "output-tokens", "reasoning-tokens"):
        parser.add_argument("--" + field, type=int, required=True)
    args = parser.parse_args()
    prices = json.loads((HERE / "api-pricing.json").read_text(), parse_float=Decimal)
    print(quote(args.pages, args.input_tokens, args.output_tokens, args.reasoning_tokens, prices))


if __name__ == "__main__":
    main()
