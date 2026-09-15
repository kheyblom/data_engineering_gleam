"""Bring a built GLEAM store onto the data engineering style guide, in place.

**Temporary.** This exists to migrate the 28 stores that were built before the
style guide, and should be deleted once they are done. ``gleam_zarr.py`` is
being changed separately so that a build lands in this state directly; this
script only catches up what is already on disk.

Nothing here touches a data value. Every unit conversion in
``nomenclature-key_gleam.md`` is ``none`` -- GLEAM already publishes these
quantities in the canonical units, spelled differently -- and the data variables
are already float32, the original dtype. So the whole migration is three
metadata operations:

1. **rename the array**, ``E`` -> ``evaporation``, through icechunk's
   ``rearrange_session``, which moves a node in the hierarchy without copying a
   chunk.
2. **write the attributes**: canonical ``units``, ``long_name`` and
   ``standard_name`` on the variable, the upstream strings preserved beside them
   as ``original_*``, and the few root attributes that named the old variable or
   the old store path brought up to date.

   Only the few. Most of a store's global attributes come from the config and
   are written by ``finalize_gleam_zarr.py --attrs``; this script deliberately
   does not author those, because an attribute with two sources is an attribute
   that will eventually disagree with itself. So ``cf_compliance``, the record
   of the frequency rename and the note about float64 coordinates belong in the
   config, and the store gets them from the ``--attrs`` step that already
   follows this one in the finalization procedure. What is left here is what
   nothing else regenerates: the variable's own attributes, ``title`` and
   ``summary`` (derived at build time only), ``history``, and ``related_store``.
3. **rename the directory**, ``...daily...E.zarr`` -> ``...day...evaporation.zarr``.

The directory rename goes last on purpose: an interrupted run then always leaves
an openable store at a path this script can find again, and every step is
skipped if it has already been done, so a half-finished run is simply re-run.

**This has an undo**, which is worth saying because ``finalize_gleam_zarr.py
--gc --apply`` does not. The pre-migration snapshot stays reachable through the
tag it already carries, so ``repository.reset_branch(...)`` to that snapshot puts
the store back, and the directory rename is one more ``os.rename``. Do not run
``--gc`` on a migrated store until it has been re-verified and re-tagged.

Every run compares the chunk manifest, the chunk storage statistics and the
array's shape, chunks, dtype and fill value from before the migration against
after. They must be identical; the run fails if anything moved. That is a
stronger statement than a sampled re-verification -- it proves not one chunk
changed -- and it costs only a manifest walk, reading no data.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import subprocess
import sys

import icechunk
import xarray as xr
import zarr

from utils.log_utils import setup_logging
from utils.nomenclature import (
    canonical_frequency,
    canonical_variable,
    key_path,
    variables,
)
from utils.path_utils import (
    format_filename,
    load_config,
    resolve_variables,
    store_path,
)
from utils.zarr_utils import BRANCH, describe_variable, open_existing_repository

LOG = logging.getLogger(__name__)

# the three steps, in the order they must run
STEPS = ('rename-array', 'attrs', 'rename-store')

# The migration writes only the attributes nothing else regenerates. Most of a
# store's global attributes come from the config and are written by
# finalize_gleam_zarr.py --attrs; authoring them here as well would give one
# attribute two sources and let the two drift. So cf_compliance, the
# nomenclature note, the dtype note and the canonical frequency all belong in
# the config -- they are true of a whole layout, not of one store -- and this
# script leaves them alone. What it owns is listed in intended_root_attrs.

def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Migrate built GLEAM stores onto the style guide nomenclature.'
    )
    parser.add_argument(
        '--config', required=True, help='Path to the YAML config for the layout.'
    )
    parser.add_argument(
        '--variable',
        default=None,
        help='Migrate only this variable, named as GLEAM publishes it (e.g. E). '
        'Omitted, every variable of the layout is migrated.',
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Report which steps each store still needs, and exit. Reads only.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually make the changes. Without it, nothing is written.',
    )
    return parser.parse_args()


def revision():
    """Short git revision of this working tree, for the history attribute.

    Returns:
        str: The short SHA, or 'unknown' outside a git checkout.
    """
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def migrated_settings(settings, original):
    """The settings that render a migrated store's name.

    The canonical variable name and the canonical frequency token are both
    substituted, so ``store_path`` and ``format_filename`` produce the migrated
    spelling from the config as it stands today. ``temporal_resolution`` is only
    overridden for *naming*: the raw netCDF tree keeps GLEAM's own ``daily``
    directory and nothing here reads it.

    Args:
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it, e.g. 'E'.

    Returns:
        dict: A copy of the settings with the canonical name and frequency.
    """
    return {
        **settings,
        'variable': canonical_variable(original),
        'temporal_resolution': canonical_frequency(settings['temporal_resolution']),
    }


def sibling_name(settings, original, migrated):
    """Relative path of the same variable's store in the other layout.

    The ``chunking`` and ``related_store`` attributes quote this path, and it
    changes with the migration, so both spellings are needed to rewrite them.

    Args:
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it.
        migrated (bool): Render the migrated spelling rather than the old one.

    Returns:
        str: e.g. 'temporal/gleam.v_4_3_a.day.native_0p1x0p1.evaporation.zarr'.
    """
    base = migrated_settings(settings, original) if migrated else {
        **settings, 'variable': original
    }
    suffix = settings['output_conventions']['suffix']
    other = 'temporal' if suffix == 'spatial' else 'spatial'
    return format_filename(
        {**base, 'output_conventions': {**base['output_conventions'], 'suffix': other}}
    )


def locate(settings, original):
    """Find a store, whether or not its directory has been renamed yet.

    Args:
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it.

    Returns:
        tuple: The path the store is at now, and the path it should end at.

    Raises:
        FileNotFoundError: If there is no store at either path.
    """
    old = store_path({**settings, 'variable': original})
    new = store_path(migrated_settings(settings, original))
    for path in (new, old):
        if os.path.isdir(path):
            return path, new
    raise FileNotFoundError(f'no store for {original!r} at {old} or {new}')


def manifest(session, array_path):
    """Sorted chunk coordinates of one array, read off the manifest.

    ``chunk_coordinates`` is an async generator, so it is drained the same way
    ``verify_gleam_zarr.py`` drains it. No chunk is fetched: this is the same
    metadata-only walk the verifier's ``sweep`` phase does.

    Args:
        session (icechunk.Session): A session on the store.
        array_path (str): Path of the array, e.g. '/E'.

    Returns:
        list: Sorted chunk coordinate tuples.
    """

    async def drain():
        return [tuple(c) async for c in session.chunk_coordinates(array_path)]

    return sorted(asyncio.run(drain()))


def fingerprint(repository, name):
    """Everything about a store that the migration must leave untouched.

    Deliberately not a sample. The chunk manifest is the complete list of which
    chunks exist and where, so comparing it either side of the migration proves
    that not one chunk moved, was added or was dropped -- which no amount of
    reading values back could prove.

    ``chunk_storage_stats`` rather than the deprecated ``total_chunks_storage``:
    the latter counts only native bytes and reports 0 for a store whose chunks
    are all inlined, which would make it useless as a guard on a small store.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        name (str): The data variable's name in the store right now.

    Returns:
        dict: The manifest, the storage statistics and the array's own spec.
    """
    session = repository.readonly_session(branch=BRANCH)
    array = zarr.open_group(session.store, mode='r')[name]
    stats = repository.chunk_storage_stats()
    return {
        'chunks': manifest(session, f'/{name}'),
        'bytes': (stats.native_bytes, stats.inlined_bytes, stats.virtual_bytes),
        'shape': array.shape,
        'chunk_shape': array.chunks,
        'dtype': str(array.dtype),
        'fill_value': repr(array.fill_value),
    }


def compare_fingerprints(before, after, label):
    """Fail loudly if anything that had to stay put has moved.

    Args:
        before (dict): Fingerprint taken before the migration.
        after (dict): Fingerprint taken after it.
        label (str): The store, for the message.

    Returns:
        bool: True if the two are identical.
    """
    moved = [key for key in before if before[key] != after[key]]
    if moved:
        for key in moved:
            # the manifests are far too long to print, so report the shape of
            # the difference rather than the difference
            if key == 'chunks':
                lost = set(before[key]) - set(after[key])
                gained = set(after[key]) - set(before[key])
                LOG.error(
                    f'{label}: chunk manifest changed, {len(before[key])} -> '
                    f'{len(after[key])} chunks, {len(lost)} lost, {len(gained)} gained'
                )
            else:
                LOG.error(f'{label}: {key} changed, {before[key]} -> {after[key]}')
        LOG.error(
            f'{label}: the migration was supposed to be metadata only. Do not '
            f'trust this store; roll it back to its tagged snapshot.'
        )
        return False
    LOG.info(
        f'{label}: unchanged, {len(after["chunks"])} chunks and '
        f'{sum(after["bytes"]) / 1024**3:.2f} GiB either side'
    )
    return True


def intended_variable_attrs(current, original):
    """The attributes the migrated data variable should carry.

    The ``original_*`` values are taken from what the store actually holds
    rather than from the nomenclature key, so they record what GLEAM really
    published instead of what a table claims it did. Once written they are the
    source of truth, so re-running reads them back rather than re-deriving them
    from attributes that have already been replaced.

    Args:
        current (dict): The variable's attributes as they stand.
        original (str): The variable as GLEAM publishes it, e.g. 'E'.

    Returns:
        dict: The full attribute set for the migrated variable.
    """
    entry = variables()[original]
    # on a re-run the canonical values are already in place, so the upstream
    # strings have to come from the original_* attributes written last time
    migrated = 'original_variable_name' in current
    originals = {
        'original_variable_name': current.get('original_variable_name', original),
        'original_standard_name': current.get(
            'original_standard_name' if migrated else 'standard_name', ''
        ),
        'original_long_name': current.get(
            'original_long_name' if migrated else 'long_name', ''
        ),
        'original_units': current.get(
            'original_units' if migrated else 'units', ''
        ),
    }
    return {
        **current,
        'standard_name': entry.canonical,
        'long_name': entry.long_name,
        'units': entry.units,
        **originals,
        'unit_conversion': (
            f'{entry.unit_conversion} -- {originals["original_units"]!r} and '
            f'{entry.units!r} are the same unit under a different spelling; '
            f'no value was changed'
        ),
    }


def intended_root_attrs(dataset, settings, original, sha):
    """The global attributes the migration itself owns.

    Deliberately narrow, because most of a store's globals come from the config
    and ``finalize_gleam_zarr.py --attrs`` is what writes those. What is left is
    the handful nothing else regenerates:

    - ``title`` and ``summary``, derived by ``describe_variable`` at build time
      only -- finalization deliberately does not refresh them, so a rename would
      otherwise leave them naming a variable the store no longer holds. They are
      regenerated here with the same function the build uses, over a dataset
      already carrying the canonical name, units and long name, so this script
      and a fresh build cannot word them differently.
    - ``history``, which no config carries.
    - ``related_store``, which quotes the sibling store's path -- a path this
      migration changes.

    ``chunking`` quotes that same path and *is* config-owned, so it is corrected
    here too and will later be rewritten identically from the config. That costs
    nothing and leaves the store self-consistent the moment the migration ends,
    rather than only after the next ``--attrs`` run.

    Both are rewritten by substituting the new sibling path for the old one
    rather than by re-authoring the prose, which belongs to the config.

    Args:
        dataset (xarray.Dataset): The store, with the canonical variable
            attributes already applied in memory.
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it.
        sha (str): Short git revision, for the history entry.

    Returns:
        dict: The full global attribute set for the migrated store.
    """
    canonical = canonical_variable(original)
    attrs = dict(dataset.attrs)

    # the same generator the build uses, over the canonical variable attributes
    attrs.update(describe_variable(dataset, migrated_settings(settings, original)))

    old_sibling = sibling_name(settings, original, migrated=False)
    new_sibling = sibling_name(settings, original, migrated=True)
    for key in ('chunking', 'related_store'):
        if key in attrs:
            attrs[key] = attrs[key].replace(old_sibling, new_sibling)

    old_name = os.path.basename(store_path({**settings, 'variable': original}))
    new_name = os.path.basename(store_path(migrated_settings(settings, original)))
    entry = (
        f'{datetime.date.today().isoformat()}: renamed the data variable '
        f'{original} -> {canonical} and restandardized its units, long_name and '
        f'standard_name to nomenclature_data.md, keeping the strings GLEAM '
        f'published as original_* attributes; store renamed {old_name} -> '
        f'{new_name}. Metadata only, by migrate_nomenclature.py '
        f'(data_engineering_gleam @ {sha}): no data value was read, written or '
        f'moved, and the chunk manifest is identical either side.'
    )
    history = attrs.get('history', '')
    # idempotent: a re-run must not stack the same sentence up again
    if 'migrate_nomenclature.py' not in history:
        attrs['history'] = f'{history} {entry}'.strip()
    return attrs


def plan_store(repository, settings, original, path, target):
    """Work out what a store still needs, and what it should end up holding.

    Args:
        repository (icechunk.Repository): The repository holding the store.
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it.
        path (str): Where the store is now.
        target (str): Where it should end up.

    Returns:
        tuple: The pending step names, the store's current variable name, and
            the intended variable and root attributes.

    Raises:
        ValueError: If the store holds something other than one recognisable
            data variable.
    """
    canonical = canonical_variable(original)
    session = repository.readonly_session(branch=BRANCH)
    dataset = xr.open_zarr(session.store, consolidated=False)

    names = list(dataset.data_vars)
    if names not in ([original], [canonical]):
        raise ValueError(
            f'{path} holds {names}, expected exactly [{original!r}] or '
            f'[{canonical!r}]'
        )
    name = names[0]

    current_variable_attrs = dict(dataset[name].attrs)
    current_root_attrs = dict(dataset.attrs)
    variable_attrs = intended_variable_attrs(current_variable_attrs, original)

    # describe_variable reads units and long_name off the variable, so the root
    # attributes have to be built from something already carrying the canonical
    # ones. It has to be a *copy*: assigning to dataset[canonical].attrs reaches
    # through to the underlying variable, so doing it in place would rewrite the
    # very attributes the pending check below compares against, and the store
    # would then always look like it needed nothing.
    described = (
        dataset.rename({name: canonical}) if name != canonical else dataset
    ).copy()
    described[canonical].attrs = variable_attrs
    root_attrs = intended_root_attrs(described, settings, original, revision())

    pending = []
    if name != canonical:
        pending.append('rename-array')
    if variable_attrs != current_variable_attrs or root_attrs != current_root_attrs:
        pending.append('attrs')
    if path != target:
        pending.append('rename-store')
    return pending, name, variable_attrs, root_attrs


def report_attr_changes(label, current, intended):
    """Log every attribute the migration would add or replace.

    Args:
        label (str): What the attributes belong to, for the message.
        current (dict): The attributes as they stand.
        intended (dict): The attributes the migration would write.
    """
    for key in intended:
        if key not in current:
            LOG.info(f'  {label} add     {key} = {intended[key]!r}')
        elif current[key] != intended[key]:
            LOG.info(
                f'  {label} replace {key} = {current[key]!r} -> {intended[key]!r}'
            )


def migrate_store(settings, original, apply_changes):
    """Run the three steps against one store.

    Args:
        settings (dict): The loaded configuration.
        original (str): The variable as GLEAM publishes it, e.g. 'E'.
        apply_changes (bool): Write, rather than only report.

    Returns:
        int: 0 if the store is migrated or would be, 1 if a guard failed.
    """
    canonical = canonical_variable(original)
    path, target = locate(settings, original)
    repository = open_existing_repository(path)
    pending, name, variable_attrs, root_attrs = plan_store(
        repository, settings, original, path, target
    )

    LOG.info(f'{original} -> {canonical}')
    LOG.info(f'  at        {path}')
    if not pending:
        LOG.info('  nothing to do, this store is already migrated')
        return 0
    LOG.info(f'  remaining {" then ".join(pending)}')
    if 'rename-store' in pending:
        LOG.info(f'  target    {target}')

    session = repository.readonly_session(branch=BRANCH)
    dataset = xr.open_zarr(session.store, consolidated=False)
    if 'attrs' in pending:
        report_attr_changes(f'{canonical}', dict(dataset[name].attrs), variable_attrs)
        report_attr_changes('root', dict(dataset.attrs), root_attrs)
    if not apply_changes:
        return 0

    before = fingerprint(repository, name)

    if 'rename-array' in pending:
        session = repository.rearrange_session(BRANCH)
        session.move(f'/{name}', f'/{canonical}')
        LOG.info(f'  committed {session.commit(f"rename {name} to {canonical}")}')

    if 'attrs' in pending:
        session = repository.writable_session(BRANCH)
        group = zarr.open_group(session.store, mode='r+')
        # the whole dict each time rather than a partial update: update_attributes
        # merges but its async twin replaces, and only one of those keeps the
        # upstream GLEAM attributes -- the same reasoning finalize_gleam_zarr.py
        # gives for passing the merged dict whole
        group[canonical].update_attributes(variable_attrs)
        group.update_attributes(root_attrs)
        LOG.info(
            f'  committed '
            f'{session.commit(f"align {canonical} with the nomenclature key")}'
        )

    after = fingerprint(repository, canonical)
    if not compare_fingerprints(before, after, canonical):
        return 1

    if path != target:
        # last, so an interrupted run always leaves an openable store at a path
        # this script can find again
        os.rename(path, target)
        LOG.info(f'  renamed   {os.path.basename(path)} -> {os.path.basename(target)}')
        # and prove the moved repository still opens and still holds the same
        # chunks, since nothing else would notice if it did not
        moved = open_existing_repository(target)
        if not compare_fingerprints(before, fingerprint(moved, canonical), canonical):
            return 1
    return 0


def main(settings, args):
    """Migrate every requested store and return a process exit status.

    Args:
        settings (dict): The loaded configuration.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: 0 if every store succeeded, 1 otherwise.
    """
    family = resolve_variables(settings)
    if args.variable:
        if args.variable not in family:
            raise ValueError(f'{args.variable!r} is not in the family {family}')
        family = [args.variable]

    LOG.info(f'nomenclature key {key_path()}')
    LOG.info(f'{len(family)} store(s) of the {settings["output_conventions"]["suffix"]} layout')
    if not args.apply:
        LOG.info('--apply not given: reporting only, nothing will be written')

    failed = 0
    for original in family:
        failed += migrate_store(settings, original, args.apply and not args.status)
    if failed:
        LOG.error(f'{failed} store(s) failed')
    return 1 if failed else 0


if __name__ == '__main__':
    arguments = parse_args()
    configuration = load_config(arguments.config)
    setup_logging(
        os.path.join(
            configuration['directories']['logs'],
            f'migrate_{configuration["log_file"]}',
        )
    )
    sys.exit(main(configuration, arguments))
