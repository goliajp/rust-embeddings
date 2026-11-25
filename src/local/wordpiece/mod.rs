//! BERT WordPiece tokenization with the uncased BERT pipeline the bundled models
//! use: added-token extraction, `BertNormalizer` (clean text, pad CJK, strip
//! accents, lowercase), `BertPreTokenizer`, greedy longest-match WordPiece with
//! `##` continuations and `[UNK]` for words over 100 characters, then
//! `[CLS] text [SEP]`. The character tables are generated from the reference
//! implementation, so the ids match it exactly.

mod tables;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::LazyLock;

use tables::{BLOCKS, INDEX, MAP_KEYS, MAP_POOL, MAP_SPANS, SHIFT, SPECIAL};

const VOCAB: &str = include_str!("../vocab.txt");

// added tokens, matched in the raw text before normalization
const ADDED: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
const MAX_WORD_CHARS: usize = 100;

// character kinds (bits 2..5 of a table entry)
const IDENT: u8 = 0;
const LOWER: u8 = 1;
const CJK: u8 = 2;
const MAP: u8 = 3;
const DROP: u8 = 4;
const BREAK: u8 = 5;
const SPECIAL_KIND: u8 = 6;
const HANGUL: u8 = 7;

// pre-tokenizer classes (bits 0..2)
const WS: u8 = 1;
const PUNCT: u8 = 2;

// the ascii fast path reads block 0 directly
const _: () = assert!(INDEX[0] == 0 && SHIFT >= 7);

type Map = HashMap<&'static str, u32, BuildHasherDefault<FxHasher>>;

/// The shared vocabulary of the bundled models with its lookup maps.
pub(crate) struct WordPiece {
    // every token, as written in the vocabulary
    words: Map,
    // continuation tokens without their `##`
    cont: Map,
    max_word: usize,
    max_cont: usize,
    added: [u32; 5],
    unk: u32,
    cls: u32,
    sep: u32,
}

/// The tokenizer, built on first use.
pub(crate) fn get() -> &'static WordPiece {
    static WP: LazyLock<WordPiece> = LazyLock::new(WordPiece::new);
    &WP
}

impl WordPiece {
    fn new() -> Self {
        let mut words = Map::default();
        let mut cont = Map::default();
        for (id, token) in VOCAB.lines().enumerate() {
            words.insert(token, id as u32);
            if let Some(rest) = token.strip_prefix("##") {
                cont.insert(rest, id as u32);
            }
        }
        let max_word = words.keys().map(|t| t.len()).max().unwrap_or(0);
        let max_cont = cont.keys().map(|t| t.len()).max().unwrap_or(0);
        let added = ADDED.map(|t| words[t]);
        WordPiece {
            unk: words["[UNK]"],
            cls: words["[CLS]"],
            sep: words["[SEP]"],
            words,
            cont,
            max_word,
            max_cont,
            added,
        }
    }

    /// Append the ids of `[CLS] text [SEP]` to `out`, keeping at most `max_len`
    /// ids: the text's tokens are cut from the end. `max_len` must be at least 2.
    pub(crate) fn encode(&self, text: &str, max_len: usize, out: &mut Vec<u32>) {
        assert!(max_len >= 2, "max_len must leave room for [CLS] and [SEP]");
        out.push(self.cls);
        let limit = out.len() + max_len - 2;
        let mut run = Run {
            wp: self,
            out,
            limit,
            word: String::new(),
            pending: Vec::new(),
        };
        run.text(text);
        out.truncate(limit);
        out.push(self.sep);
    }

    // greedy longest match; a word that cannot be covered becomes one [UNK]
    fn word(&self, word: &str, out: &mut Vec<u32>) {
        if word.len() > MAX_WORD_CHARS && word.chars().count() > MAX_WORD_CHARS {
            out.push(self.unk);
            return;
        }
        let mark = out.len();
        let mut start = 0;
        while start < word.len() {
            let (map, longest) = if start == 0 {
                (&self.words, self.max_word)
            } else {
                (&self.cont, self.max_cont)
            };
            let mut end = word.len().min(start + longest);
            while !word.is_char_boundary(end) {
                end -= 1;
            }
            loop {
                if end <= start {
                    out.truncate(mark);
                    out.push(self.unk);
                    return;
                }
                if let Some(&id) = map.get(&word[start..end]) {
                    out.push(id);
                    break;
                }
                end -= 1;
                while !word.is_char_boundary(end) {
                    end -= 1;
                }
            }
            start = end;
        }
    }
}

struct Run<'a> {
    wp: &'a WordPiece,
    out: &'a mut Vec<u32>,
    limit: usize,
    // the word being collected by the pre-tokenizer
    word: String,
    // marks with a nonzero combining class waiting for canonical ordering
    pending: Vec<(u8, &'static str)>,
}

impl Run<'_> {
    fn full(&self) -> bool {
        self.out.len() >= self.limit
    }

    // split off the added tokens, then normalize and tokenize each piece between them
    fn text(&mut self, text: &str) {
        let mut rest = text;
        let mut from = 0;
        while let Some(at) = rest[from..].find('[') {
            let at = from + at;
            match ADDED.iter().position(|t| rest[at..].starts_with(t)) {
                Some(k) => {
                    self.piece(&rest[..at]);
                    if self.full() {
                        return;
                    }
                    self.out.push(self.wp.added[k]);
                    rest = &rest[at + ADDED[k].len()..];
                    from = 0;
                }
                None => from = at + 1,
            }
        }
        self.piece(rest);
    }

    fn piece(&mut self, piece: &str) {
        let bytes = piece.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if self.full() {
                return;
            }
            let b = bytes[i];
            if self.pending.is_empty() && is_word_byte(b) {
                // a run of ascii characters that pass through unchanged into a word
                let start = i;
                while i < bytes.len() && is_word_byte(bytes[i]) {
                    i += 1;
                }
                let ends_word = bytes.get(i).is_none_or(|&b| ends_word(b));
                if self.word.is_empty() && ends_word {
                    self.wp.word(&piece[start..i], self.out);
                } else {
                    self.word.push_str(&piece[start..i]);
                }
            } else if b < 0x80 {
                let code = BLOCKS[b as usize] - b'0';
                match code >> 2 {
                    IDENT if self.pending.is_empty() => self.emit_classed(b as char, code & 3),
                    LOWER if self.pending.is_empty() => {
                        self.word.push(b.to_ascii_lowercase() as char)
                    }
                    _ => self.normalize(b as char),
                }
                i += 1;
            } else {
                let c = piece[i..].chars().next().unwrap();
                self.normalize(c);
                i += c.len_utf8();
            }
        }
        self.flush();
        self.end_word();
    }

    fn normalize(&mut self, c: char) {
        let code = lookup(c);
        match code >> 2 {
            DROP => {}
            SPECIAL_KIND => {
                let units = SPECIAL[SPECIAL.binary_search_by_key(&c, |s| s.0).unwrap()].1;
                for &(class, s) in units {
                    if class == 0 {
                        self.flush();
                        self.emit_str(s);
                    } else {
                        self.pending.push((class, s));
                    }
                }
            }
            kind => {
                self.flush();
                match kind {
                    IDENT => self.emit_classed(c, code & 3),
                    LOWER => self.emit(c.to_ascii_lowercase()),
                    CJK => {
                        self.end_word();
                        self.emit(c);
                        self.end_word();
                    }
                    MAP => {
                        let at = MAP_KEYS.binary_search(&(c as u32)).unwrap();
                        let span = MAP_SPANS[at] as usize;
                        self.emit_str(&MAP_POOL[span >> 8..(span >> 8) + (span & 0xFF)]);
                    }
                    HANGUL => {
                        let s = c as u32 - 0xAC00;
                        self.emit(char::from_u32(0x1100 + s / 588).unwrap());
                        self.emit(char::from_u32(0x1161 + (s % 588) / 28).unwrap());
                        if !s.is_multiple_of(28) {
                            self.emit(char::from_u32(0x11A7 + s % 28).unwrap());
                        }
                    }
                    BREAK => {}
                    _ => unreachable!("kinds are three bits"),
                }
            }
        }
    }

    // canonical ordering: a stable sort by combining class
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut pending = std::mem::take(&mut self.pending);
        pending.sort_by_key(|p| p.0);
        for (_, s) in pending.drain(..) {
            self.emit_str(s);
        }
        self.pending = pending;
    }

    fn emit_str(&mut self, s: &str) {
        for c in s.chars() {
            self.emit(c);
        }
    }

    fn emit(&mut self, c: char) {
        self.emit_classed(c, lookup(c) & 3);
    }

    // the pre-tokenizer: whitespace ends a word, punctuation is a word of its own
    fn emit_classed(&mut self, c: char, class: u8) {
        match class {
            WS => self.end_word(),
            PUNCT => {
                self.end_word();
                let mut buf = [0; 4];
                self.wp.word(c.encode_utf8(&mut buf), self.out);
            }
            _ => self.word.push(c),
        }
    }

    fn end_word(&mut self) {
        if !self.word.is_empty() {
            self.wp.word(&self.word, self.out);
            self.word.clear();
        }
    }
}

// an ascii byte that normalizes to itself and belongs to a word
#[inline]
fn is_word_byte(b: u8) -> bool {
    b < 0x80 && BLOCKS[b as usize] - b'0' == IDENT << 2
}

// an ascii byte that normalizes to itself and ends the word before it
#[inline]
fn ends_word(b: u8) -> bool {
    b < 0x80 && {
        let code = BLOCKS[b as usize] - b'0';
        code >> 2 == IDENT && code & 3 != 0
    }
}

fn lookup(c: char) -> u8 {
    let c = c as usize;
    let block = INDEX[c >> SHIFT] as usize;
    BLOCKS[(block << SHIFT) | (c & ((1 << SHIFT) - 1))] - b'0'
}

// FxHash: the vocabulary is fixed, so a fast non-keyed hash is enough
#[derive(Default)]
struct FxHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            self.add(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            self.add(u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64);
            bytes = &bytes[4..];
        }
        for &b in bytes {
            self.add(b as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests;
