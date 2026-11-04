"""Convert the parity fixture, using a dummy vocab for direct token-ID input."""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
from conversion.qwen import QSAQwen35Model
from conversion.base import gguf

def vocab(self):
    self.gguf_writer.add_tokenizer_model("gpt2")
    self.gguf_writer.add_tokenizer_pre("default")
    self.gguf_writer.add_token_list([f"token{i}" for i in range(128)])
    self.gguf_writer.add_token_types([gguf.TokenType.NORMAL] * 128)
    self.gguf_writer.add_token_merges(["t o"])
    self.gguf_writer.add_bos_token_id(126)
    self.gguf_writer.add_eos_token_id(127)
QSAQwen35Model.set_vocab = vocab
import convert_hf_to_gguf
convert_hf_to_gguf.main()
