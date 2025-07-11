// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use anyhow::Result;

type SpTokenizer = std::sync::Arc<sentencepiece::SentencePieceProcessor>;

static RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#"<break\s+time="([0-9.]+)s"\s*/>"#).unwrap());

fn normalize(text: &str) -> String {
    text.replace('’', "'").replace('–', "").replace(':', " ").replace(['(', ')'], "")
}

fn parse_segments(input: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut last = 0;

    for mat in RE.find_iter(input) {
        if mat.start() > last {
            let text = &input[last..mat.start()];
            if !text.trim().is_empty() {
                segments.push(Segment::Str(text));
            }
        }
        if let Some(caps) = RE.captures(mat.as_str()) {
            if let Ok(secs) = caps[1].parse::<f64>() {
                segments.push(Segment::BreakTime(secs));
            }
        }

        last = mat.end();
    }
    if last < input.len() {
        let text = input[last..].trim();
        if !text.is_empty() {
            segments.push(Segment::Str(text));
        }
    }
    segments
}

#[derive(Debug, PartialEq, Clone, serde::Deserialize, serde::Serialize)]
pub struct WordWithTokens {
    pub word: String,
    pub tokens: Vec<u32>,
}

pub struct Tokenizer {
    tok: SpTokenizer,
    inserted_bos: bool,
    text_bos_token: u32,
}

#[derive(Debug, PartialEq)]
enum Segment<'a> {
    Str(&'a str),
    BreakTime(f64),
}

impl Tokenizer {
    pub fn new(tok: SpTokenizer, text_bos_token: u32) -> Self {
        Self { tok, inserted_bos: false, text_bos_token }
    }

    pub fn preprocess(&mut self, query: &str) -> Result<Vec<WordWithTokens>> {
        let segments = parse_segments(query);
        let mut word_with_tokens = Vec::new();
        for segment in segments.into_iter() {
            match segment {
                Segment::Str(text) => {
                    let text = normalize(text);
                    for word in text.split_whitespace() {
                        if word.is_empty() {
                            continue;
                        }
                        let mut word_tokens: Vec<_> =
                            self.tok.encode(word)?.into_iter().map(|v| v.id).collect();
                        if !self.inserted_bos {
                            self.inserted_bos = true;
                            word_tokens.insert(0, self.text_bos_token);
                        }
                        word_with_tokens
                            .push(WordWithTokens { word: word.to_string(), tokens: word_tokens });
                    }
                }
                Segment::BreakTime(secs) => {
                    if secs > 0.0 {
                        let npad = usize::max((secs.min(10.) * 12.5) as usize, 1);
                        word_with_tokens.push(WordWithTokens {
                            word: format!("<break time=\"{secs:.2}s\">"),
                            tokens: vec![self.tok.pad_id().unwrap_or(3); npad],
                        });
                    }
                }
            }
        }
        Ok(word_with_tokens)
    }
}

#[test]
fn test_segment_parser() {
    let input = r#"Hello <break time="0.5s"/> world <break time="1.0s"/>!"#;
    let segments = parse_segments(input);
    assert_eq!(
        segments,
        vec![
            Segment::Str("Hello "),
            Segment::BreakTime(0.5),
            Segment::Str(" world "),
            Segment::BreakTime(1.0),
            Segment::Str("!")
        ]
    );
    let input = r#"Hello <break time="0.5s"/> world <break time="1.0s"/>  "#;
    let segments = parse_segments(input);
    assert_eq!(
        segments,
        vec![
            Segment::Str("Hello "),
            Segment::BreakTime(0.5),
            Segment::Str(" world "),
            Segment::BreakTime(1.0),
        ]
    );
    let input = r#"<break time="0.5s"/>yay!<break time="1.0s"/>  "#;
    let segments = parse_segments(input);
    assert_eq!(
        segments,
        vec![Segment::BreakTime(0.5), Segment::Str("yay!"), Segment::BreakTime(1.0),]
    );
}
