//! Plain-text normalisation applied before hashing and chunking.

use unicode_normalization::UnicodeNormalization;

/// NFC; drop control chars (except that `\n`/`\t` act as whitespace); `\r\n`/`\r` → `\n`;
/// collapse horizontal whitespace runs to one space; trim each line; one or more blank
/// lines → exactly `\n\n`; trim. Empty or garbage-only input returns `""`.
pub fn normalize(input: &str) -> String {
    let unified = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(unified.len());
    let mut breaks = 0usize;
    for (i, line) in unified.split('\n').enumerate() {
        if i > 0 {
            breaks += 1;
        }
        let cleaned = clean_line(line);
        if cleaned.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str(if breaks >= 2 { "\n\n" } else { "\n" });
        }
        out.push_str(&cleaned);
        breaks = 0;
    }
    out.nfc().collect()
}

/// One line without `\n`: drops control chars and U+FEFF, collapses whitespace, trims.
fn clean_line(line: &str) -> String {
    let mut s = String::with_capacity(line.len());
    let mut pending_space = false;
    for c in line.chars() {
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        if c.is_control() || c == '\u{FEFF}' {
            continue;
        }
        if pending_space && !s.is_empty() {
            s.push(' ');
        }
        pending_space = false;
        s.push(c);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_garbage_normalise_to_empty() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   \n\t\n  \r\n "), "");
        assert_eq!(normalize("\u{0000}\u{0007}\u{001b}\u{FEFF}"), "");
        assert_eq!(normalize(" \u{00A0}\u{3000} "), "");
    }

    #[test]
    fn crlf_and_cr_become_lf() {
        assert_eq!(normalize("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn horizontal_whitespace_collapses_and_lines_are_trimmed() {
        assert_eq!(normalize("a  \t b"), "a b");
        assert_eq!(normalize("  a   \n   b  "), "a\nb");
        assert_eq!(normalize("a\u{00A0}\u{00A0}b"), "a b");
    }

    #[test]
    fn paragraph_breaks_are_kept_but_capped_at_two_newlines() {
        assert_eq!(normalize("a\n\nb"), "a\n\nb");
        assert_eq!(normalize("a\n\n\n\n\nb"), "a\n\nb");
        assert_eq!(normalize("a\n   \n\t\nb"), "a\n\nb");
        assert_eq!(normalize("\n\n\na\n\n\n"), "a");
    }

    #[test]
    fn control_chars_are_dropped_but_tab_and_newline_are_whitespace() {
        assert_eq!(normalize("a\u{0007}b\u{0000}c"), "abc");
        assert_eq!(normalize("a \u{0007} b"), "a b");
        assert_eq!(normalize("a\tb\nc"), "a b\nc");
        assert_eq!(normalize("\u{FEFF}title"), "title");
    }

    #[test]
    fn output_is_nfc() {
        assert_eq!(normalize("e\u{0301}"), "\u{00E9}");
        // A control char between base and combining mark must not block composition.
        assert_eq!(normalize("e\u{0000}\u{0301}x"), "\u{00E9}x");
    }

    #[test]
    fn idempotent_on_mixed_input() {
        let samples = [
            "  Hello,\t\tworld!\r\n\r\n\r\nSecond  para\u{0007}.\n line two  ",
            "漢字\u{3000}かな\n\n\n\nカナ",
            "e\u{0301}\u{FEFF} x \n \n y",
        ];
        for s in samples {
            let once = normalize(s);
            assert_eq!(normalize(&once), once, "not idempotent for {s:?}");
        }
    }
}
