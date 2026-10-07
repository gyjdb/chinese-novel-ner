"""Load GLiNER and feed it Chinese text correctly.

Three things go wrong with GLiNER on Chinese out of the box:

1. Its default word splitter (regex \\w+) treats a whole run of Chinese characters
   as one "word", so the model can only label whole clauses.
2. transformers 5.x rebuilds the mDeBERTa tokenizer without sentencepiece's
   nmt_nfkc normalization, so full-width punctuation (，“”！？：…) becomes [UNK].
   We put the original precompiled normalizer back (checked: token ids then match
   sentencepiece on 10k paragraphs except a handful of byte-fallback characters).
3. With is_split_into_words, the tokenizer puts "▁" (a space marker) before every
   word, i.e. it reads Chinese as if every character were separated by a space.
   space_mode="natural" only marks words that follow real whitespace, which is how
   mDeBERTa saw Chinese during pretraining; "per_word" keeps GLiNER's default.
"""
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from sentencepiece import sentencepiece_model_pb2
from tokenizers import Regex, normalizers, pre_tokenizers

MODEL_ID = "urchade/gliner_multi-v2.1"
MODEL_REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"
BACKBONE_ID = "microsoft/mdeberta-v3-base"
BACKBONE_REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"
SPACE_MARK = "▁"  # "▁", sentencepiece's word-boundary marker

SPACE_MODES = ("per_word", "natural")


class WhitespaceSplitter:
    """GLiNER's default splitter, kept only as a baseline."""

    pattern = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __call__(self, text):
        for m in self.pattern.finditer(text):
            yield m.group(), m.start(), m.end()


class CharSplitter:
    """One word per character; runs of ASCII letters/digits stay together (E27, 014)."""

    pattern = re.compile(r"[A-Za-z0-9]+|\S")

    def __call__(self, text):
        for m in self.pattern.finditer(text):
            yield m.group(), m.start(), m.end()


class JiebaSplitter:
    """jieba word segmentation (default dictionary, HMM on for unseen words)."""

    def __init__(self):
        import jieba

        jieba.setLogLevel(60)
        self._jieba = jieba

    def __call__(self, text):
        for word, start, end in self._jieba.tokenize(text):
            if not word.isspace():
                yield word, start, end


class NaturalSpaceMarks:
    """Prefix "▁" only to words at the start of the text or after whitespace.

    Used with a tokenizer whose Metaspace pre-tokenizer never adds "▁" itself.
    Offsets are untouched, so entity text is still sliced from the original.
    """

    def __init__(self, splitter):
        self.splitter = splitter

    def __call__(self, text):
        for word, start, end in self.splitter(text):
            if start == 0 or text[start - 1].isspace():
                word = SPACE_MARK + word
            yield word, start, end


SPLITTERS = {"whitespace": WhitespaceSplitter, "char": CharSplitter, "jieba": JiebaSplitter}


def _backbone_charsmap():
    path = Path(snapshot_download(BACKBONE_ID, revision=BACKBONE_REVISION, allow_patterns=["spm.model"]))
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((path / "spm.model").read_bytes())
    return proto.normalizer_spec.precompiled_charsmap


class ChineseGLiNER:
    """A GLiNER model whose Chinese input handling can be switched per variant."""

    def __init__(self, device="cuda"):
        import torch
        from gliner import GLiNER

        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                 allow_patterns=["*.json", "*.md", "model.safetensors"])
        self.model = GLiNER.from_pretrained(path).to(device)
        self.model.eval()
        self.device = device
        self.torch = torch
        backend = self.tokenizer.backend_tokenizer
        self._stock_normalizer = backend.normalizer
        self._stock_pre_tokenizer = backend.pre_tokenizer
        self._fixed_normalizer = normalizers.Sequence([
            normalizers.Strip(),
            normalizers.Precompiled(_backbone_charsmap()),
            normalizers.Replace(Regex(" {2,}"), " "),
        ])
        self.space_mode = "per_word"

    @property
    def tokenizer(self):
        return self.model.data_processor.transformer_tokenizer

    def configure(self, splitter="char", space_mode="per_word", fix_normalizer=True):
        if space_mode not in SPACE_MODES:
            raise ValueError(f"space_mode must be one of {SPACE_MODES}")
        backend = self.tokenizer.backend_tokenizer
        backend.normalizer = self._fixed_normalizer if fix_normalizer else self._stock_normalizer
        if space_mode == "natural":
            backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement=SPACE_MARK, prepend_scheme="never",
                                                             split=True)
            words = NaturalSpaceMarks(SPLITTERS[splitter]())
        else:
            backend.pre_tokenizer = self._stock_pre_tokenizer
            words = SPLITTERS[splitter]()
        self.model.data_processor.words_splitter = words
        self.space_mode = space_mode

    def words(self, text):
        return list(self.model.data_processor.words_splitter(text))

    def token_count(self, text, labels):
        """Encoder length for one input: label prompt + text, as GLiNER builds it."""
        prompt = []
        for label in self._label_words(labels):
            prompt += [self.model.config.ent_token, label]
        prompt.append(self.model.config.sep_token)
        words = [w for w, _, _ in self.words(text)]
        return len(self.tokenizer(prompt + words, is_split_into_words=True)["input_ids"])

    def _label_words(self, labels):
        # In natural mode a label is a word after a space, as in GLiNER's training prompts.
        if self.space_mode == "natural":
            return [SPACE_MARK + label for label in labels]
        return list(labels)

    def predict(self, texts, labels, threshold, batch_size=16):
        """Entities per text: dicts with start, end (char offsets), text, label, score."""
        label_words = self._label_words(labels)
        back = dict(zip(label_words, labels))
        with self.torch.inference_mode():
            out = self.model.inference(texts, label_words, flat_ner=True, threshold=threshold,
                                       batch_size=batch_size)
        for entities in out:
            for e in entities:
                e["label"] = back[e["label"]]
        return out
