"""The text of the model's text tokens, as both modes print it and the gate scores it."""

import codecs


def text_special_ids(tokenizer) -> set[int]:
    return {tokenizer.unk_id(), tokenizer.eos_id(), tokenizer.pad_id()}


class TextDecoder:
    """The text of a stream of text tokens, one token at a time: for valid UTF-8, what the
    tokenizer decodes from the stream's non-special tokens. A byte-fallback piece (`<0xE1>`)
    carries one UTF-8 byte, so a character the vocabulary lacks spans several tokens: its text
    comes out once its sequence is complete, an invalid run of bytes as one U+FFFD (SentencePiece
    writes one per byte), and `flush` returns what is left when the stream ends. Special tokens
    add nothing and do not break a sequence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = text_special_ids(tokenizer)
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def __call__(self, token: int) -> str:
        if token in self.special:
            return ""
        piece = self.tokenizer.id_to_piece(token)
        if self.tokenizer.is_byte(token):
            return self.utf8.decode(bytes([int(piece[1:-1], 16)]))
        return self.utf8.decode(piece.replace("▁", " ").encode())

    def flush(self) -> str:
        return self.utf8.decode(b"", final=True)
