#!/usr/bin/env python3

"""Batch inference using ESM3 Forge respecting API rate limits."""

import argparse
import logging
import time
from typing import Iterable, List, Tuple

from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, GenerationConfig
from Bio import SeqIO

# 80% of the official Forge rate limit
SEQ_PER_MIN = 40
AA_PER_MIN = 160_000


def batch_iter(iterable: List[Tuple[str, str]], size: int) -> Iterable[List[Tuple[str, str]]]:
    """Yield successive batches from an iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", type=argparse.FileType("r"))
    parser.add_argument("--model", "-m", default="esm3-medium-2024-03")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    records = list(SeqIO.parse(args.fasta, "fasta"))
    sequences = [(rec.id, str(rec.seq).replace("*", "")) for rec in records]
    logger.info("Loaded %d sequences", len(sequences))

    client = ESM3ForgeInferenceClient(
        model=args.model, url="https://forge.evolutionaryscale.ai", token="TKv9Vv4bY7cd7vBFVr0ap"
    )

    for batch_num, batch in enumerate(batch_iter(sequences, 500), start=1):
        logger.info("Processing batch %d with %d sequences", batch_num, len(batch))
        pos = 0
        while pos < len(batch):
            aa_count = 0
            seqs: List[str] = []
            ids: List[str] = []
            # assemble sub-batch respecting rate limits
            while pos < len(batch) and len(seqs) < SEQ_PER_MIN:
                seq_id, seq = batch[pos]
                if aa_count + len(seq) > AA_PER_MIN:
                    break
                ids.append(seq_id)
                seqs.append(seq)
                aa_count += len(seq)
                pos += 1

            proteins = [ESMProtein(sequence=s) for s in seqs]
            configs = [GenerationConfig(track="structure", schedule="cosine") for _ in proteins]

            start = time.time()
            folds = client.batch_generate(proteins, configs)
            for sid, fold in zip(ids, folds):
                fold.to_pdb(f"./{sid}.pdb")
            logger.info("Generated %d structures", len(folds))

            elapsed = time.time() - start
            wait_seconds = max(len(seqs) / SEQ_PER_MIN, aa_count / AA_PER_MIN) * 60 - elapsed
            if wait_seconds > 0:
                logger.info("Sleeping %.2f seconds to respect rate limit", wait_seconds)
                time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
