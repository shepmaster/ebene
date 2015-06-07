#![feature(collections)]

extern crate strata;
extern crate itertools;
extern crate rustc_serialize;
extern crate punkt;

use strata::*;
use itertools::Itertools;
use rustc_serialize::json::{self, Json};
use punkt::trainer::{TrainingData, Trainer};
use punkt::tokenizer::{SentenceTokenizer, WordTokenizer};

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[derive(Debug,Copy,Clone,PartialEq)]
pub struct ZZZ {
    doc_id: u32,
    offset: u32,
}

impl From<u64> for ZZZ {
    fn from(v: u64) -> ZZZ {
        ZZZ { doc_id: (v >> 32) as u32, offset: (v & 0xFFFFFFFF) as u32 }
    }
}

impl From<ZZZ> for u64 {
    fn from(v: ZZZ) -> u64 {
        (v.doc_id as u64) << 32 | (v.offset as u64)
    }
}

#[derive(Debug,Clone,RustcDecodable, RustcEncodable)]
struct InputDocument {
    text: String,
    layers: HashMap<String, Vec<ValidExtent>>,
}

fn read_document(filename: &str) -> InputDocument {
    let mut f = File::open(filename).unwrap();

    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();

    json::decode(&s).unwrap()
}

fn index_document(content: &str) -> HashMap<String, Vec<ValidExtent>> {
    let mut index = HashMap::new();

    {
        let mut chars = content.char_indices();

        loop {
            for _ in chars.by_ref().take_while_ref(|&(_, c)| ! c.is_alphabetic()) {}

            let (first, last) = {
                let mut words = chars.by_ref().take_while_ref(|&(_, c)| c.is_alphabetic());
                (words.next(), words.last())
            };

            let extent = match (first, last) {
                (Some(s), Some(e)) => (s.0 as u64, e.0 as u64 + 1),
                (Some(s), None)    => (s.0 as u64, s.0 as u64 + 1),
                (None, _)          => break,
            };

            let word = content[(extent.0 as usize)..(extent.1 as usize)].to_lowercase();

            index.entry(word).or_insert(vec![]).push((extent.0, extent.1));
        }
    }

    index
}

fn index_sentences(content: &str) -> Vec<ValidExtent> {
    let mut sentences = Vec::new();
    let mut offset = 0;

    let training_data = TrainingData::english();

    for sentence in SentenceTokenizer::new(content, &training_data) {
        let inner_offset = content[offset..].find(sentence).unwrap();
        let start = offset + inner_offset;
        let end = start + sentence.len();

        sentences.push((start as u64, end as u64));

        offset = end;
    }

    sentences
}

fn index_lines(content: &str) -> Vec<ValidExtent> {
    let mut lines = Vec::new();
    let mut offset = 0;

    for line in content.split("\n") {
        let start = offset;
        let end = start + line.len();
        lines.push((start as u64, end as u64));
        offset = end + "\n".len();
    }

    lines
}

fn index_stanzas(content: &str) -> Vec<ValidExtent> {
    let mut stanzas = Vec::new();
    let mut offset = 0;

    for stanza in content.split("\n\n") {
        let start = offset;
        let end = start + stanza.len();
        stanzas.push((start as u64, end as u64));
        offset = end + "\n\n".len();
    }

    stanzas
}

fn json_to_query<'a>(json: &Json,
                     index: &'a HashMap<String, Vec<ValidExtent>>,
                     layers: &'a HashMap<String, Vec<ValidExtent>>)
                     -> Result<Box<Algebra + 'a>, &'static str>
{
    let op: Box<Algebra> = match *json {
        Json::String(ref s) => Box::new(index.get(&s[..]).map(|x| &x[..]).unwrap_or(&[][..])),
        Json::Array(ref a) => {
            let cmd = a.get(0);
            let lhs = a.get(1);
            let rhs = a.get(2);

            let (cmd, lhs, rhs) = match (cmd, lhs, rhs) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => return Err("Malformed op"),
            };

            let cmd = match *cmd {
                Json::String(ref s) => s,
                _ => return Err("Not a valid op"),
            };

            if cmd == "L" {
                if let &Json::String(ref s) = lhs {
                    let z = layers.get(&s[..]).map(|x| &x[..]).unwrap_or(&[][..]);
                    return Ok(Box::new(z));
                } else {
                    return Err("bad layer request");
                }
            }

            let a = try!(json_to_query(lhs, index, layers));
            let b = try!(json_to_query(rhs, index, layers));

            let op: Box<Algebra> = match &cmd[..] {
                "<"  => Box::new(ContainedIn::new(a, b)),
                ">"  => Box::new(Containing::new(a, b)),
                "/<" => Box::new(NotContainedIn::new(a, b)),
                "/>" => Box::new(NotContaining::new(a, b)),
                "&"  => Box::new(BothOf::new(a, b)),
                "|"  => Box::new(OneOf::new(a, b)),
                "->" => Box::new(FollowedBy::new(a, b)),
                _    => return Err("Unknown op"),
            };

            op
        },
        _ => Box::new(Empty),
    };

    Ok(op)
}

fn index() -> (Vec<String>, HashMap<String, Vec<ValidExtent>>, HashMap<String, Vec<ValidExtent>>) {
    let mut data = Vec::new();
    let mut index = HashMap::new();
    let mut layers = HashMap::new();

    for file in env::args().skip(1) {
        let mut doc = read_document(&file);
        let doc_index = index_document(&doc.text);

        doc.layers.insert("sentence".to_owned(), index_sentences(&doc.text));
        doc.layers.insert("line".to_owned(), index_lines(&doc.text));
        doc.layers.insert("stanza".to_owned(), index_stanzas(&doc.text));

        data.push(doc.text);
        for (word, extents) in doc_index {
            index.entry(word).or_insert(vec![]).extend(extents);
        }
        for (name, extents) in doc.layers {
            layers.entry(name).or_insert(vec![]).extend(extents);
        }
    }

    println!("=Index=");
    for (word, extents) in &index {
        println!("{}: {:?}", word, extents);
    }
    println!("=Layers=");
    for (layer, extents) in &layers {
        println!("{}: {:?}", layer, extents);
    }

    (data, index, layers)
}

fn query_stdin(data: Vec<String>, index: HashMap<String, Vec<ValidExtent>>, layers: HashMap<String, Vec<ValidExtent>>) {
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let l = line.unwrap();

        let q = match Json::from_str(&l) {
            Ok(q) => q,
            Err(e) => {
                println!("Error: {}", e);
                continue;
            },
        };

        let op = match json_to_query(&q, &index, &layers) {
            Ok(op) => op,
            Err(e) => { println!("Error: {}", e); continue },
        };

        for extent in op.iter_tau() {
            let ex = (extent.0, extent.1);
            let content = &data[0]; // HACK: 0 isnt right
            println!("{:?}: {}", ex, &content[(ex.0 as usize)..(ex.1 as usize)]);
        }
    }
}

fn main() {
    let (data, index, layers) = index();
    query_stdin(data, index, layers);
}
