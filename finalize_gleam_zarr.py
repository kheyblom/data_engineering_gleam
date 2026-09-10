"""Finalize a verified GLEAM icechunk zarr store for delivery.

This is the only script here that mutates a finished store, and it is separate
from ``verify_gleam_zarr.py`` on purpose. The verifier is how you prove a
finalization step did no harm, so it stays read only and safe to run by reflex;
a tool that both acted and audited would report one exit status for two
questions.

Three actions, exactly one per invocation:

``--attrs``
    Write the config's ``attrs`` section, plus the coverage and grid attributes
    derived from the data, onto the store's root group. Additive: it rewrites
    one metadata object as a new commit and touches no chunk or manifest.
``--tag NAME``
    Name the current branch tip. icechunk tags are immutable, so a consumer can
    pin a tag and be unaffected by any later commit; a branch tip cannot offer
    that.
``--gc``
    Delete objects no surviving snapshot references -- the chunks and snapshots
    left behind by batches that were killed before they could commit.

Nothing happens without ``--apply``: by default each action reports what it
would do and exits. ``--gc --apply`` is **irreversible**. It is safe in the
sense that icechunk only ever collects unreachable objects, and this script
checks that claim by comparing reachable bytes and history length either side
of the call, but a store with no second copy has nothing to restore from if
that check ever fails.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

import xarray as xr
import zarr

from utils.log_utils import setup_logging
from utils.path_utils import format_attrs, load_config, store_path
from utils.zarr_utils import BRANCH, derive_attrs, open_existing_repository

LOG = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Finalize a verified GLEAM icechunk zarr store.'
    )
    parser.add_argument(
        '--config', required=True, help='Path to the YAML config for the store.'
    )
    # one action per run: chaining them would hide which one failed
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        '--attrs', action='store_true', help='Write global attributes.'
    )
    action.add_argument('--tag', help='Create this tag at the branch tip.')
    action.add_argument(
        '--gc', action='store_true', help='Delete unreachable objects.'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually make the change. Without it, nothing is written.',
    )
    return parser.parse_args()


def resolve_attrs(repository, settings):
    """Work out the attributes the store should carry.

    The store's own attributes are the base, so GLEAM's upstream provenance
    survives; the derived coverage and grid values go on top of those; the
    config goes on top of everything, so any derived value can be overridden by
    hand without changing code.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        settings (dict): The loaded configuration.

    Returns:
        tuple: The merged attributes and the store's current attributes.
    """
    session = repository.readonly_session(branch=BRANCH)
    dataset = xr.open_zarr(session.store, consolidated=False)
    current = dict(dataset.attrs)
    return current | derive_attrs(dataset) | format_attrs(settings), current


def run_attrs(repository, settings, apply_changes):
    """Write the store's global attributes.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        settings (dict): The loaded configuration.
        apply_changes (bool): Whether to commit, rather than only report.

    Returns:
        int: 0 on success.
    """
    merged, current = resolve_attrs(repository, settings)
    added = [key for key in merged if key not in current]
    changed = [
        key for key in merged if key in current and merged[key] != current[key]
    ]
    LOG.info(f'{len(current)} attributes now, {len(merged)} after')
    for key in added:
        LOG.info(f'  add     {key} = {merged[key]!r}')
    for key in changed:
        LOG.info(f'  replace {key} = {current[key]!r} -> {merged[key]!r}')
    if not added and not changed:
        LOG.info('attributes already up to date, nothing to write')
        return 0
    if not apply_changes:
        LOG.info('dry run, nothing written; pass --apply to commit')
        return 0

    session = repository.writable_session(branch=BRANCH)
    group = zarr.open_group(session.store, mode='r+')
    # the merged dict is passed whole rather than relying on update semantics:
    # Group.update_attributes merges but its async twin replaces, and only one
    # of those preserves the upstream attributes
    group.update_attributes(merged)
    snapshot_id = session.commit('add provenance and discovery attributes')
    LOG.info(f'committed {snapshot_id}')
    return 0


def run_tag(repository, tag, apply_changes):
    """Create a tag at the branch tip.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        tag (str): Name of the tag to create.
        apply_changes (bool): Whether to create it, rather than only report.

    Returns:
        int: 0 on success, 1 if the tag already exists.
    """
    existing = repository.list_tags()
    # ancestry yields newest first, so the tip is the head of the list
    tip = next(iter(repository.ancestry(branch=BRANCH)))
    LOG.info(f'tip {tip.id} written {tip.written_at.isoformat()} -- {tip.message}')
    LOG.info(f'existing tags: {sorted(existing) or "none"}')
    if tag in existing:
        LOG.error(f'tag {tag!r} already exists; tags are immutable, pick another')
        return 1
    if not apply_changes:
        LOG.info(f'dry run, nothing created; --apply would tag {tip.id} as {tag!r}')
        return 0

    repository.create_tag(tag, tip.id)
    LOG.info(f'created tag {tag!r} at {tip.id}')
    return 0


def run_gc(repository, apply_changes):
    """Delete objects no surviving snapshot references.

    Brackets the collection with the two properties it must not change:
    reachable chunk bytes and the length of the branch's history. Garbage
    collection is defined to remove only unreachable objects, so either of
    those moving means something went wrong, and on a store without a second
    copy that is worth checking rather than assuming.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        apply_changes (bool): Whether to delete, rather than only report.

    Returns:
        int: 0 on success, 1 if the reachable state changed.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc)
    before_bytes = repository.chunk_storage_stats().native_bytes
    before_history = len(list(repository.ancestry(branch=BRANCH)))
    LOG.info(
        f'before: {before_bytes / 1024**4:.4f} TiB reachable across '
        f'{before_history} snapshots'
    )

    dry = repository.garbage_collect(cutoff, dry_run=True)
    LOG.info(f'dry run would delete {summarize(dry)}')
    if not apply_changes:
        LOG.info('dry run, nothing deleted; pass --apply to collect')
        return 0

    summary = repository.garbage_collect(cutoff, dry_run=False)
    LOG.info(f'deleted {summarize(summary)}')

    after_bytes = repository.chunk_storage_stats().native_bytes
    after_history = len(list(repository.ancestry(branch=BRANCH)))
    LOG.info(
        f'after: {after_bytes / 1024**4:.4f} TiB reachable across '
        f'{after_history} snapshots'
    )
    if after_bytes != before_bytes or after_history != before_history:
        LOG.error(
            f'garbage collection changed the reachable store: '
            f'{before_bytes} -> {after_bytes} bytes, '
            f'{before_history} -> {after_history} snapshots. '
            f'Do not trust this store; restore it before using it.'
        )
        return 1
    LOG.info('reachable bytes and history unchanged, as garbage collection requires')
    return 0


def summarize(summary):
    """Render a garbage collection summary as one line.

    Args:
        summary (icechunk.GCSummary): The summary to render.

    Returns:
        str: The counts, with bytes in GiB.
    """
    return (
        f'{summary.bytes_deleted / 1024**3:.2f} GiB: '
        f'{summary.chunks_deleted} chunks, '
        f'{summary.manifests_deleted} manifests, '
        f'{summary.snapshots_deleted} snapshots, '
        f'{summary.attributes_deleted} attributes, '
        f'{summary.transaction_logs_deleted} transaction logs'
    )


def main(settings, args):
    """Run the selected action and return a process exit status.

    Args:
        settings (dict): The loaded configuration.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: 0 if the action succeeded, 1 otherwise.
    """
    path = store_path(settings)
    LOG.info(f'finalizing {path}')
    if not args.apply:
        LOG.info('--apply not given: reporting only, nothing will be written')
    # Repository.open rather than open_or_create, so a mistyped path fails here
    # instead of producing an empty repository that every later step agrees with
    repository = open_existing_repository(path)

    if args.attrs:
        return run_attrs(repository, settings, args.apply)
    if args.tag:
        return run_tag(repository, args.tag, args.apply)
    return run_gc(repository, args.apply)


if __name__ == '__main__':
    arguments = parse_args()
    configuration = load_config(arguments.config)
    setup_logging(
        os.path.join(configuration['directories']['logs'], 'finalize_gleam_zarr.log')
    )
    sys.exit(main(configuration, arguments))
