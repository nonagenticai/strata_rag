# Migrations

Applied by `sqlx::migrate!("./migrations")` at startup (`src/db/mod.rs`), each file in its own
transaction. sqlx records a checksum of every applied file in `_sqlx_migrations`: **never edit a
migration that has shipped** — add a new one. Only `<version>_<description>.sql` files are
migrations; this README is ignored by the migrator.

## Trap: `stores` writes and `ALTER TABLE stores` in one migration

`0010_store_generations.sql` adds `stores_bump_generation`, a `DEFERRABLE INITIALLY DEFERRED`
constraint trigger on `stores` (it fires at COMMIT, to keep the lock order of `db::stores`). While
its event is pending, Postgres refuses `ALTER TABLE stores` in the same transaction:

```
ERROR: cannot ALTER TABLE "stores" because it has pending trigger events  (SQLSTATE 55006)
```

So a migration that INSERTs/UPDATEs/DELETEs `stores` rows (a backfill, a data fix) and then alters
`stores` fails on every database that holds a store, and passes on an empty one (a fresh test DB).
Either:

- fire the pending events first, right after the write and before the `ALTER`:

  ```sql
  UPDATE stores SET ...;
  SET CONSTRAINTS stores_bump_generation IMMEDIATE;
  ALTER TABLE stores ...;
  ```

  (the generation is still bumped; `SET CONSTRAINTS` only lasts until the end of the transaction), or
- split the write and the `ALTER` into two migrations.

Order matters: `ALTER` first, then the write, is fine as-is. Proven in
`tests/db_triggers.rs::a_store_write_then_alter_needs_the_generation_trigger_set_immediate`.

Since `0020_store_content_sums.sql`, statement triggers on `chunks` and `documents` keep each
leaf's running content sums on its `stores` row: a migration that writes `chunks` or `documents`
writes `stores` too, and the same rule applies before an `ALTER TABLE stores`.
