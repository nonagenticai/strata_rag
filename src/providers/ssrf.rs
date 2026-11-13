//! SSRF guard for tenant-registered `base_url`s. strata calls a registered provider from inside the
//! cluster (the probe, then every embed / rerank), so a tenant must not be able to aim it at
//! internal services: loopback, private and CGNAT ranges, link-local (cloud metadata),
//! unique-local, unspecified, multicast, broadcast, and the `.local` / `.internal` /
//! `.cluster.local` names.
//!
//! Two layers, both skipped only when the dev-only policy flag `allow_internal_provider_hosts` is
//! set (deployment defaults from `providers.toml` are operator-set and never checked):
//! - [`check_registration`] at PUT time: `https` only, no blocked IP literal or name suffix, and
//!   every address the host resolves to must be public.
//! - [`GuardedResolver`] at connect time: tenant clients resolve through it and never see a
//!   blocked address, so a host that re-resolves to an internal IP later (DNS rebinding) fails to
//!   connect. Redirects stay disabled and no proxy is used.
//!
//! Messages name the rule, never a resolved address or a resolver error (no network mapping).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use super::{Capability, Rejection};

pub const HOST_NOT_ALLOWED: &str = "provider_host_not_allowed";

/// Name suffixes that only resolve inside our networks.
const BLOCKED_SUFFIXES: [&str; 3] = [".local", ".internal", ".cluster.local"];

fn v4_blocked(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    ip.is_loopback()                       // 127/8
        || ip.is_private()                 // 10/8, 172.16/12, 192.168/16
        || ip.is_link_local()              // 169.254/16 (metadata)
        || (a == 100 && (64..128).contains(&b)) // 100.64/10 CGNAT
        || a == 0                          // 0/8 "this network", incl. unspecified
        || ip.is_multicast()               // 224/4
        || ip.is_broadcast()               // 255.255.255.255
        || a >= 240                        // 240/4 reserved
        || (a == 192 && b == 0 && c == 0)  // 192.0.0/24 IETF protocol assignments
        || (a == 198 && (b == 18 || b == 19)) // 198.18/15 benchmarking
        || (a == 192 && b == 0 && c == 2)  // 192.0.2/24 TEST-NET-1
        || (a == 198 && b == 51 && c == 100) // 198.51.100/24 TEST-NET-2
        || (a == 203 && b == 0 && c == 113) // 203.0.113/24 TEST-NET-3
        || (a == 192 && b == 88 && c == 99) // 192.88.99/24 6to4 relay anycast
}

/// The IPv4 address in the last 32 bits of `ip`.
fn low_v4(ip: Ipv6Addr) -> Ipv4Addr {
    let s = ip.segments();
    Ipv4Addr::from((u32::from(s[6]) << 16) | u32::from(s[7]))
}

fn v6_blocked(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return v4_blocked(v4);
    }
    let seg = ip.segments();
    let first = seg[0];
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (first & 0xfe00) == 0xfc00 // fc00::/7 unique-local
        || (first & 0xffc0) == 0xfe80 // fe80::/10 link-local
        || (first & 0xffc0) == 0xfec0 // fec0::/10 site-local (deprecated)
        || seg[..4] == [0x100, 0, 0, 0] // 100::/64 discard-only
        || (first == 0x2001 && ip.segments()[1] == 0x0db8) // 2001:db8::/32 documentation
        || (first == 0x2001 && ip.segments()[1] == 0) // 2001::/32 Teredo
        // 6to4 (2002::/16) carries an IPv4 address in its next 32 bits
        || (first == 0x2002
            && v4_blocked(Ipv4Addr::from(
                (u32::from(ip.segments()[1]) << 16) | u32::from(ip.segments()[2]),
            )))
        // IPv4-compatible (::a.b.c.d, deprecated) and NAT64 (64:ff9b::/96) carry an IPv4 address
        || (ip.segments()[..6] == [0; 6])
        || (seg[..6] == [0x64, 0xff9b, 0, 0, 0, 0] && v4_blocked(low_v4(ip)))
        // local-use NAT64 (64:ff9b:1::/48) and IPv4-translated (::ffff:0:a.b.c.d) carry one too
        || (seg[..3] == [0x64, 0xff9b, 1] && v4_blocked(low_v4(ip)))
        || (seg[..6] == [0, 0, 0, 0, 0xffff, 0] && v4_blocked(low_v4(ip)))
}

/// True for an address a tenant provider must never reach.
pub fn ip_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_blocked(v4),
        IpAddr::V6(v6) => v6_blocked(v6),
    }
}

/// `addrs` without the blocked ones (the connect-time filter).
pub fn public_only(addrs: impl IntoIterator<Item = SocketAddr>) -> Vec<SocketAddr> {
    addrs.into_iter().filter(|a| !ip_blocked(a.ip())).collect()
}

fn reject(cap: Capability, why: &str) -> Rejection {
    Rejection {
        code: HOST_NOT_ALLOWED,
        message: format!("{cap}: base_url {why}"),
    }
}

/// Why a `base_url` failed to parse (see [`parse_base_url`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlError {
    /// Backslash, whitespace or a control character in the raw input (the WHATWG parser would
    /// silently strip or reinterpret them, so what we check would differ from what is fetched).
    Unsafe,
    NotAUrl,
    Scheme,
    Credentials,
    QueryOrFragment,
    NoHost,
    /// `url.as_str()` differs from the input (case, IDNA, percent-encoding, default port, path
    /// normalisation): only the canonical form is accepted, so what is stored is what is fetched.
    NotCanonical,
}

impl UrlError {
    pub fn code(self) -> &'static str {
        match self {
            UrlError::NotAUrl | UrlError::Scheme | UrlError::QueryOrFragment => "invalid_provider",
            _ => HOST_NOT_ALLOWED,
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            UrlError::Unsafe => "must not contain backslashes, whitespace or control characters",
            UrlError::NotAUrl => "is not a valid URL",
            UrlError::Scheme => "must start with http:// or https://",
            UrlError::Credentials => "must not carry credentials (send the key as api_key)",
            UrlError::QueryOrFragment => "must not have a query or fragment",
            UrlError::NoHost => "needs a host",
            UrlError::NotCanonical => {
                "must be in canonical form (lowercase ASCII host, no default port, no \
                 percent-encoded host, normalised path)"
            }
        }
    }
}

/// Parses `base_url` with `url::Url`, the WHATWG parser reqwest uses, and accepts it only when
/// nothing about it is ambiguous: no backslash / whitespace / control characters in the raw text,
/// http(s), no userinfo, no query or fragment, a host, and `url.as_str()` equal to the input (up
/// to a trailing `/`). The stored `base_url` is therefore exactly what reqwest will fetch.
pub fn parse_base_url(raw: &str) -> Result<url::Url, UrlError> {
    if raw
        .chars()
        .any(|c| c == '\\' || c.is_whitespace() || c.is_control())
    {
        return Err(UrlError::Unsafe);
    }
    let url = url::Url::parse(raw).map_err(|_| UrlError::NotAUrl)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(UrlError::Scheme);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(UrlError::Credentials);
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(UrlError::QueryOrFragment);
    }
    if url.host().is_none() {
        return Err(UrlError::NoHost);
    }
    if url.as_str().trim_end_matches('/') != raw.trim_end_matches('/') {
        return Err(UrlError::NotCanonical);
    }
    Ok(url)
}

/// The static checks (no DNS) on the host reqwest will connect to: https, a parsed IP in no blocked
/// range, or a domain that is not internal-looking (suffixes, single label, numeric spellings).
/// Returns that host.
pub fn check_static(cap: Capability, base_url: &str) -> Result<url::Host<String>, Rejection> {
    let url = parse_base_url(base_url).map_err(|e| Rejection {
        code: HOST_NOT_ALLOWED,
        message: format!("{cap}: base_url {}", e.message()),
    })?;
    if url.scheme() != "https" {
        return Err(reject(cap, "must use https"));
    }
    let host = url
        .host()
        .map(|h| h.to_owned())
        .ok_or_else(|| reject(cap, "has no host"))?;
    match &host {
        url::Host::Ipv4(ip) if v4_blocked(*ip) => Err(reject(cap, "is a non-public address")),
        url::Host::Ipv6(ip) if v6_blocked(*ip) => Err(reject(cap, "is a non-public address")),
        url::Host::Ipv4(_) | url::Host::Ipv6(_) => Ok(host),
        url::Host::Domain(d) => {
            let d = d.trim_end_matches('.').to_ascii_lowercase();
            // decimal / hex / short IPv4 spellings some resolvers accept ("2130706433")
            if d.chars().all(|c| c.is_ascii_digit() || c == '.')
                || d.split('.').any(|label| {
                    label.strip_prefix("0x").is_some_and(|hex| {
                        !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit())
                    })
                })
            {
                return Err(reject(cap, "host must be a DNS name or a dotted public IP"));
            }
            if d == "localhost"
                || d.ends_with(".localhost")
                || BLOCKED_SUFFIXES.iter().any(|s| d.ends_with(s))
                || !d.contains('.')
            {
                return Err(reject(cap, "is an internal host name"));
            }
            Ok(host)
        }
    }
}

/// The PUT-time check: [`check_static`], then every address the (same) host resolves to must be
/// public. A host that does not resolve is refused too (it could not be probed anyway).
pub async fn check_registration(cap: Capability, base_url: &str) -> Result<(), Rejection> {
    let url::Host::Domain(host) = check_static(cap, base_url)? else {
        return Ok(());
    };
    let addrs: Vec<SocketAddr> = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::net::lookup_host((host.as_str(), 443)),
    )
    .await
    {
        Ok(Ok(a)) => a.collect(),
        _ => return Err(reject(cap, "host does not resolve")),
    };
    if addrs.is_empty() {
        return Err(reject(cap, "host does not resolve"));
    }
    if addrs.iter().any(|a| ip_blocked(a.ip())) {
        return Err(reject(cap, "resolves to a non-public address"));
    }
    Ok(())
}

/// reqwest resolver for tenant clients: the system resolver, minus every blocked address. All
/// blocked (or nothing resolved) is a connect error with a fixed message.
#[derive(Debug, Default, Clone, Copy)]
pub struct GuardedResolver;

#[derive(Debug)]
struct NotAllowed;

impl std::fmt::Display for NotAllowed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("provider host not allowed")
    }
}

impl std::error::Error for NotAllowed {}

impl reqwest::dns::Resolve for GuardedResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|_| Box::new(NotAllowed) as Box<dyn std::error::Error + Send + Sync>)?;
            let ok = public_only(addrs);
            if ok.is_empty() {
                return Err(Box::new(NotAllowed) as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(ok.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_ranges() {
        for ip in [
            "127.0.0.1",
            "127.255.0.1",
            "10.0.0.1",
            "10.43.0.10",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.64.0.2",
            "100.127.255.255",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "::1",
            "::",
            "fe80::1",
            "fc00::1",
            "fd12:3456::1",
            "ff02::1",
            "::ffff:10.0.0.1",
            "::ffff:127.0.0.1",
            "64:ff9b::a9fe:a9fe",
        ] {
            assert!(ip_blocked(ip.parse().unwrap()), "{ip} must be blocked");
        }
        for ip in [
            "1.1.1.1",
            "8.8.8.8",
            "100.63.255.255",
            "100.128.0.1",
            "172.32.0.1",
            "192.169.0.1",
            "2606:4700::1111",
            "::ffff:1.1.1.1",
        ] {
            assert!(!ip_blocked(ip.parse().unwrap()), "{ip} is public");
        }
    }

    #[test]
    fn remaining_reserved_ranges_are_blocked() {
        for ip in [
            "192.88.99.1",
            "64:ff9b:1::a00:1",       // local-use NAT64 of 10.0.0.1
            "64:ff9b:1:abcd::7f00:1", // ... of 127.0.0.1
            "::ffff:0:a00:1",         // IPv4-translated 10.0.0.1
            "::ffff:0:a9fe:a9fe",     // ... 169.254.169.254
            "fec0::1",
            "feff::1",
            "100::1",
            "100::ffff:ffff:ffff:ffff",
        ] {
            assert!(ip_blocked(ip.parse().unwrap()), "{ip} must be blocked");
        }
        for ip in [
            "192.88.100.1",
            "64:ff9b:1::101:101", // local-use NAT64 of 1.1.1.1
            "::ffff:0:101:101",   // IPv4-translated 1.1.1.1
            "100:0:0:1::1",       // outside 100::/64
            "2606:4700::1111",
        ] {
            assert!(!ip_blocked(ip.parse().unwrap()), "{ip} is public");
        }
    }

    #[test]
    fn only_a_real_hex_label_is_refused() {
        let c = Capability::Embed;
        for bad in [
            "https://0x0a000001/v1",
            "https://0xff.example.com/v1",
            "https://a.0x7f.com/v1",
        ] {
            assert_eq!(
                check_static(c, bad).err().map(|r| r.code),
                Some(HOST_NOT_ALLOWED),
                "{bad}"
            );
        }
        for ok in [
            "https://0xfoo.example.com/v1",
            "https://0x.example.com/v1",
            "https://api.0xproject.com/v1",
        ] {
            assert!(
                check_static(c, ok).is_ok(),
                "{ok}: {:?}",
                check_static(c, ok)
            );
        }
    }

    #[test]
    fn documentation_benchmark_and_tunnel_ranges_are_blocked() {
        for ip in [
            "192.0.0.8",
            "198.18.0.1",
            "198.19.255.255",
            "192.0.2.1",
            "198.51.100.7",
            "203.0.113.9",
            "2001:db8::1",
            "2001::1",           // Teredo
            "2002:0a00:0001::1", // 6to4 of 10.0.0.1
            "2002:7f00:0001::1", // 6to4 of 127.0.0.1
        ] {
            assert!(ip_blocked(ip.parse().unwrap()), "{ip} must be blocked");
        }
        // 6to4 of a public address, and neighbours of the new ranges, stay public
        for ip in [
            "2002:0101:0101::1",
            "198.20.0.1",
            "192.0.3.1",
            "203.0.114.1",
        ] {
            assert!(!ip_blocked(ip.parse().unwrap()), "{ip} is public");
        }
    }

    /// The guard's host is the host reqwest connects to: every input is either refused, or its
    /// stored form parses (with reqwest's own `Url`) to exactly the host the guard checked.
    #[test]
    fn the_guard_checks_the_host_reqwest_connects_to() {
        let c = Capability::Embed;
        let must_refuse = [
            "https://10.0.0.1\\.x.attacker.com/v1",
            "https://10.0.0.1\\@public.example.com/v1",
            "https://public.example.com\\@10.0.0.1/v1",
            "https://10.0.0.1\t.public.example.com/v1",
            "https://public.exa\nmple.com/v1",
            "https://10.0.0.1\n.public.example.com/v1",
            " https://public.example.com/v1",
            "https://evil@10.0.0.1/",
            "https://public.com@10.0.0.1",
            "https://user:pw@public.example.com/v1",
            "HTTPS://FOO.INTERNAL",
            "HTTPS://PUBLIC.EXAMPLE.COM/v1",
            "https://foo.internal./v1",
            "https://bücher.example/v1",
            "https://[::ffff:10.0.0.1]/",
            "https://[::ffff:a00:1]/",
            "https://%31%30.0.0.1/v1",
            "https://public.example.com:443/v1",
            "https://public.example.com/v1/../admin",
            "https://public.example.com/v1#frag",
            "https://public.example.com/v1?x=1",
            "https://0x0a000001/v1",
            "https://10.1/v1",
        ];
        for raw in must_refuse {
            let stored = raw.trim_end_matches('/');
            assert!(
                check_static(c, stored).is_err(),
                "{raw:?} must be refused, got {:?}",
                check_static(c, stored)
            );
        }
        for raw in [
            "https://public.example.com/v1",
            "https://xn--bcher-kva.example/v1",
            "https://1.1.1.1/v1",
            "https://[2606:4700::1111]/v2",
            "https://api.example.com:8443/v1",
            "https://public.example.com./v1",
        ] {
            let host = check_static(c, raw).expect(raw);
            for fetched in [raw.to_string(), format!("{raw}/embeddings")] {
                let u = reqwest::Url::parse(&fetched).unwrap();
                assert_eq!(
                    u.host().map(|h| h.to_owned()),
                    Some(host.clone()),
                    "{fetched}"
                );
            }
        }
    }

    #[test]
    fn the_connect_time_filter_drops_internal_addresses() {
        let addrs: Vec<SocketAddr> = [
            "10.0.0.5:443",
            "1.1.1.1:443",
            "[fd00::1]:443",
            "100.100.1.1:443",
            "[2606:4700::1111]:443",
            "169.254.169.254:80",
        ]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
        let kept: Vec<String> = public_only(addrs).iter().map(|a| a.to_string()).collect();
        assert_eq!(kept, ["1.1.1.1:443", "[2606:4700::1111]:443"]);
        assert!(public_only(["127.0.0.1:1".parse().unwrap()]).is_empty());
    }

    #[test]
    fn static_checks() {
        let c = Capability::Embed;
        let code = |u: &str| check_static(c, u).err().map(|r| r.code);
        for bad in [
            "http://api.example.com/v1",
            "https://127.0.0.1:6432/v1",
            "https://[::1]/v1",
            "https://169.254.169.254/latest",
            "https://100.64.0.2:5432",
            "https://10.43.0.1/v1",
            "https://localhost:9131/v1",
            "https://db.example.svc.cluster.local:6432",
            "https://provider.example.internal:9131/v1",
            "https://printer.local/v1",
            "https://pgbouncer/v1",
            "https://2130706433/v1",
            "https://0x7f000001/v1",
            "https://API.Example.INTERNAL./v1",
        ] {
            assert_eq!(code(bad), Some(HOST_NOT_ALLOWED), "{bad}");
        }
        for ok in [
            "https://api.example.com/v1",
            "https://1.1.1.1/v1",
            "https://api.example.com:8443",
            "https://cafe.be/v1",
        ] {
            assert_eq!(code(ok), None, "{ok}");
        }
        let m = check_static(c, "https://10.1.2.3/v1").unwrap_err().message;
        assert!(!m.contains("10.1.2.3"), "{m}");
    }

    #[tokio::test]
    async fn the_guarded_resolver_refuses_a_name_that_resolves_only_internally() {
        use reqwest::dns::Resolve;
        let name: reqwest::dns::Name = "localhost".parse().unwrap();
        let err = match GuardedResolver.resolve(name).await {
            Ok(addrs) => panic!("resolved to {:?}", addrs.collect::<Vec<_>>()),
            Err(e) => e.to_string(),
        };
        assert_eq!(err, "provider host not allowed");
    }

    #[tokio::test]
    async fn a_name_resolving_to_loopback_is_refused() {
        // a public IP literal needs no DNS; a name that resolves only to blocked addresses is
        // refused (the resolution filter itself is `public_only`, tested above)
        assert!(
            check_registration(Capability::Rerank, "https://1.1.1.1/v2")
                .await
                .is_ok()
        );
        let r = check_registration(Capability::Rerank, "https://localhost/v2")
            .await
            .unwrap_err();
        assert_eq!(r.code, HOST_NOT_ALLOWED);
    }
}
