#!/usr/bin/env python3
"""Ensure control/user-defined pieces occupy contiguous ID bands and prune artifacts."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from sentencepiece import sentencepiece_model_pb2 as sp_pb2

META_PIECES = {"<unk>", "<s>", "</s>", "<pad>"}


def load_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_model", required=True)
    parser.add_argument("--out_model", required=True)
    parser.add_argument("--control_list", required=True)
    parser.add_argument("--user_list", required=True)
    parser.add_argument("--enforce_contiguous_bands", action="store_true")
    parser.add_argument("--prune_regex", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = sp_pb2.ModelProto()
    model.ParseFromString(Path(args.in_model).read_bytes())

    controls = load_list(Path(args.control_list))
    users = load_list(Path(args.user_list))

    control_map = {piece: None for piece in controls}
    user_map = {piece: None for piece in users}

    meta: list[sp_pb2.ModelProto.SentencePiece] = []
    others: list[sp_pb2.ModelProto.SentencePiece] = []

    prune_pattern = re.compile(args.prune_regex) if args.prune_regex else None

    for piece in model.pieces:
        if prune_pattern and prune_pattern.search(piece.piece):
            continue
        if piece.piece in META_PIECES:
            meta.append(piece)
        elif piece.piece in control_map:
            control_map[piece.piece] = piece
            piece.type = sp_pb2.ModelProto.SentencePiece.CONTROL
        elif piece.piece in user_map:
            user_map[piece.piece] = piece
            piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
        else:
            others.append(piece)

    def ensure_entries(target: dict[str, sp_pb2.ModelProto.SentencePiece | None], piece_type: int) -> Iterable[sp_pb2.ModelProto.SentencePiece]:
        for key, existing in target.items():
            if existing is not None:
                existing.type = piece_type
                yield existing
                continue
            fresh = sp_pb2.ModelProto.SentencePiece()
            fresh.piece = key
            fresh.type = piece_type
            fresh.score = 0.0
            yield fresh

    rebuilt: list[sp_pb2.ModelProto.SentencePiece] = []
    rebuilt.extend(meta)
    rebuilt.extend(ensure_entries(control_map, sp_pb2.ModelProto.SentencePiece.CONTROL))
    rebuilt.extend(ensure_entries(user_map, sp_pb2.ModelProto.SentencePiece.USER_DEFINED))
    rebuilt.extend(others)

    model.ClearField("pieces")
    model.pieces.extend(rebuilt)

    Path(args.out_model).write_bytes(model.SerializeToString())

    if args.verbose:
        control_ids = [i for i, p in enumerate(model.pieces) if p.piece in control_map]
        user_ids = [i for i, p in enumerate(model.pieces) if p.piece in user_map]
        if control_ids:
            print(f"controls start at {control_ids[0]} end at {control_ids[-1]}")
        if user_ids:
            print(f"user-defined start at {user_ids[0]} end at {user_ids[-1]}")


if __name__ == "__main__":
    main()
