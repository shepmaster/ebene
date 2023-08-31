# Ebene example

This program is a small example of how the query functionality can work.

To start, create a JSON file containing the text and pre-indexed
layers of the text:

```json
{
  "text": "Hello, world! Goodbye, world!",
  "layers": {
    "paragraph": [[0,29]],
    "sentence": [[0,13], [14,29]]
  }
}
```

Basic indexing of the text will be applied by finding contiguous runs
of alphabetic characters and downcasing them. No smart language
processing is performed.

Run the program using this JSON file as an argument:

```
cargo run example.json
```

You can now provide queries in a LISP-ish JSON format, one query per line.

Note: It's recommended to use a tool like [rlwrap][] to provide basic command-line editing functionality:

```
rlwrap cargo run example.json
```

[rlwrap]: https://github.com/hanslub42/rlwrap

## Query examples

### Occurrences of the word "world"

```json
"world"
```

Output:

```
(7, 12): world
(23, 28): world
```

### Sentences that contain the word "hello"

```json
[">", ["L", "sentence"], "hello"]
```

Output:

```
(0, 13): Hello, world!
```

### Sentences that do not contain the word "hello"

```json
["/>", ["L", "sentence"], "hello"]
```

Output:

```
(14, 29): Goodbye, world!
```

### The word "world" that occurs in a sentence with the word "goodbye"

```json
["<", "world", [">", ["L", "sentence"], "goodbye"]]
```

Output:

```
(23, 28): world
```

## Query operators

### Unary

- `L`: Layer

### Binary

- `<`: Contained In
- `>`: Containing
- `/<`: Not Contained In
- `/>`: Not Containing
- `&`: Both Of
- `|`: One Of
- `->`: Followed By
