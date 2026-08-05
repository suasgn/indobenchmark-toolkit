import unittest
from pathlib import Path

from indobenchmark import IndoNLGTokenizer


VOCAB_FILE = (
    Path(__file__).parents[1]
    / "examples"
    / "deprecated"
    / "IndoNLG_finals_vocab_model_indo4b_plus_spm_bpe_9995_wolangid_bos_pad_eos_unk.model"
)


class IndoNLGTokenizerDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = IndoNLGTokenizer(vocab_file=VOCAB_FILE)

    def test_decode_single_sequence_returns_string(self):
        decoded = self.tokenizer.decode(
            [0, 4693, 39956, 1119, 3447, 2], skip_special_tokens=True
        )

        self.assertIsInstance(decoded, str)

    def test_batch_decode_returns_one_string_per_sequence(self):
        sequences = [
            [0, 4693, 39956, 1119, 3447, 2],
            [0, 4693, 2, 1, 1, 1],
        ]

        decoded = self.tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )
        expected = [
            self.tokenizer.decode(sequence, skip_special_tokens=True)
            for sequence in sequences
        ]

        self.assertEqual(decoded, expected)


if __name__ == "__main__":
    unittest.main()
