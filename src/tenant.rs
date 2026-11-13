//! The tenant a request acts for (spec §8.6). Every store belongs to exactly one tenant
//! (`stores.tenant`); documents and chunks belong to their store's. Every engine operation but
//! `/health` takes a `Tenant` and never reads or writes another tenant's rows: a store id of
//! another tenant answers 404, exactly like an id that does not exist, so existence never leaks.

use std::fmt;

use uuid::Uuid;

use crate::config::TokenScope;
use crate::error::{Result, StrataError};

/// A tenant id: any UUID except the nil UUID. A newtype, so it cannot be swapped for a store or
/// document id in a call with several `Uuid` parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tenant(Uuid);

impl Tenant {
    /// `InvalidTenant` for the nil UUID (it would be an "everyone" placeholder, never a customer).
    pub fn new(id: Uuid) -> Result<Self> {
        if id.is_nil() {
            return Err(StrataError::InvalidTenant(
                "the nil uuid is not a tenant".to_string(),
            ));
        }
        Ok(Tenant(id))
    }

    /// Parses a bare UUID (surrounding whitespace ignored); `InvalidTenant` otherwise or for nil.
    pub fn parse(s: &str) -> Result<Self> {
        let id = Uuid::parse_str(s.trim())
            .map_err(|_| StrataError::InvalidTenant("a tenant must be a uuid".to_string()))?;
        Tenant::new(id)
    }

    pub fn id(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for Tenant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::str::FromStr for Tenant {
    type Err = StrataError;

    fn from_str(s: &str) -> Result<Self> {
        Tenant::parse(s)
    }
}

/// THE tenant resolution rule, the one place it lives (spec §8.6). `scope` is the authenticated
/// token's scope, `None` when auth is off (loopback only); `asked` is the request's parsed
/// `X-Tenant-Id`. A pinned token acts for its own tenant (the header may repeat it; a different
/// one is `TenantMismatch`, 403). A `*` token, and an unauthenticated request, must name the tenant
/// (`TenantRequired`, 400). Per-tenant lookups (e.g. providers, spec §4.1 step 6a) start from the
/// `Tenant` this returns, never from the token or header.
pub fn resolve_tenant(scope: Option<TokenScope>, asked: Option<Tenant>) -> Result<Tenant> {
    match (scope, asked) {
        (Some(TokenScope::Tenant(pinned)), None) => Ok(pinned),
        (Some(TokenScope::Tenant(pinned)), Some(t)) if t == pinned => Ok(pinned),
        (Some(TokenScope::Tenant(_)), Some(_)) => Err(StrataError::TenantMismatch(
            "X-Tenant-Id names a tenant this token is not for".to_string(),
        )),
        (Some(TokenScope::AnyTenant) | None, Some(t)) => Ok(t),
        (Some(TokenScope::AnyTenant), None) => Err(StrataError::TenantRequired(
            "this token serves several tenants; name one in X-Tenant-Id".to_string(),
        )),
        (None, None) => Err(StrataError::TenantRequired(
            "name the tenant in X-Tenant-Id".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_rules() {
        let a = Tenant::new(Uuid::from_u128(1)).unwrap();
        let b = Tenant::new(Uuid::from_u128(2)).unwrap();
        let pinned = Some(TokenScope::Tenant(a));
        let any = Some(TokenScope::AnyTenant);
        assert_eq!(resolve_tenant(pinned, None).unwrap(), a);
        assert_eq!(resolve_tenant(pinned, Some(a)).unwrap(), a);
        let e = resolve_tenant(pinned, Some(b)).unwrap_err();
        assert_eq!((e.code(), e.status()), ("tenant_mismatch", 403));
        assert_eq!(resolve_tenant(any, Some(b)).unwrap(), b);
        let e = resolve_tenant(any, None).unwrap_err();
        assert_eq!((e.code(), e.status()), ("tenant_required", 400));
        assert_eq!(resolve_tenant(None, Some(b)).unwrap(), b);
        let e = resolve_tenant(None, None).unwrap_err();
        assert_eq!((e.code(), e.status()), ("tenant_required", 400));
    }

    #[test]
    fn parses_a_uuid_and_rejects_nil_and_garbage() {
        let id = Uuid::now_v7();
        assert_eq!(Tenant::parse(&format!(" {id} ")).unwrap().id(), id);
        for bad in [
            "",
            "abc",
            "00000000-0000-0000-0000-000000000000",
            "Bearer x",
        ] {
            let err = Tenant::parse(bad).unwrap_err();
            assert_eq!(err.code(), "invalid_tenant", "{bad}");
        }
        assert!(Tenant::new(Uuid::nil()).is_err());
    }
}
