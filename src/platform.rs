//! Platform knowledge (spec 2026-10-20-platform-knowledge-design.md): packs of platform-owned
//! stores shared with every tenant. Rows live in `db::platform`; this module holds the key rules.

/// Most `/`-separated segments a pack key (or exclusion prefix) may have.
pub const MAX_KEY_SEGMENTS: usize = 6;

/// A pack key or exclusion prefix: 1 to [`MAX_KEY_SEGMENTS`] `/`-separated segments, each one or
/// more of `[a-z0-9_-]` (`base`, `industry/retail`, `job/store_manager`). Mirrors the CHECK of
/// migration 0022.
pub fn valid_pack_key(key: &str) -> bool {
    let segments: Vec<&str> = key.split('/').collect();
    segments.len() <= MAX_KEY_SEGMENTS
        && segments.iter().all(|s| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
        })
}

/// True when the exclusion `prefix` covers `key`: the key itself or any key below it, segment-wise
/// (`industry/retail` covers `industry/retail/hr`, not `industry/retailx`).
pub fn key_matches_prefix(key: &str, prefix: &str) -> bool {
    key == prefix
        || key
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_one_to_six_lowercase_segments() {
        for ok in [
            "base",
            "industry/retail",
            "industry_dept/retail/hr",
            "job/store-manager",
            "a/b/c/d/e/f",
            "0",
        ] {
            assert!(valid_pack_key(ok), "{ok}");
        }
        for bad in [
            "",
            "/",
            "/base",
            "base/",
            "industry//retail",
            "Industry",
            "a b",
            "a/b/c/d/e/f/g",
            "retail.hr",
            "ü",
        ] {
            assert!(!valid_pack_key(bad), "{bad}");
        }
    }

    #[test]
    fn prefixes_match_whole_segments() {
        assert!(key_matches_prefix("industry/retail", "industry/retail"));
        assert!(key_matches_prefix("industry/retail/hr", "industry/retail"));
        assert!(key_matches_prefix("industry/retail/hr", "industry"));
        assert!(!key_matches_prefix("industry/retailx", "industry/retail"));
        assert!(!key_matches_prefix("industry", "industry/retail"));
        assert!(!key_matches_prefix("base", "industry"));
    }
}
