"""End to end test of the GLEAM pipeline against the miniature fixture.

Everything the pipeline does was checked by hand when it was written, which
catches nothing later. This runs the same ground repeatably: it builds both
layouts from the staged fixture, verifies every store against the raw files, and
exercises the guards that only matter when they fire.

It drives the three entry points as **subprocesses** rather than importing them.
That costs a little speed and buys the real surface: the command line, the exit
status, and the log a person would actually read. A check that asserts on an
exit code alone would pass just as happily if a phase silently did nothing, so
the cases that matter assert on what the run *said* as well.

Run it from the repo root after staging the fixture:

    uv run python validation/stage_fixture.py
    uv run python validation/test_pipeline.py

About a minute, on a login node, no allocation spent. Failures are collected
rather than raised, so one run reports everything that is wrong; the exit status
is non-zero if any case failed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import icechunk
import numpy as np
import xarray as xr
import zarr

from utils.nomenclature import canonical_variable, variables
from utils.path_utils import load_config, store_path
from utils.zarr_utils import BRANCH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPATIAL = 'config/config_zarr_fixture_spatial.yaml'
TEMPORAL = 'config/config_zarr_fixture_temporal.yaml'

# as staged: Ep and its two components, plus E as a total whose components are
# absent. Sorted, because that is the order resolve_variables returns
VARIABLES = ('E', 'Ep', 'Ep_aero', 'Ep_rad')


class Results:
    """Collects case outcomes so one run reports every failure.

    Deliberately not the verifier's ``Report``: that one logs data checks
    through the module logger, and importing it here would mean either moving it
    or reaching across the repo root, which the layout rules forbid.

    Attributes:
        failures (list): Names of the cases that failed.
        n (int): How many cases ran.
    """

    def __init__(self):
        self.failures = []
        self.n = 0

    def check(self, name, ok, detail=''):
        """Record one case.

        Args:
            name (str): What was checked.
            ok (bool): Whether it held.
            detail (str): What was measured, shown either way.

        Returns:
            bool: The ``ok`` passed in.
        """
        self.n += 1
        print(f'{"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}',
              flush=True)
        if not ok:
            self.failures.append(name)
        return ok


def run(*args, expect=0):
    """Run one entry point and return its completed process.

    Args:
        *args: The command line after the interpreter.
        expect (int): The exit status the caller expects, or None for any.

    Returns:
        tuple: The subprocess.CompletedProcess, and whether the status matched.
    """
    done = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return done, (expect is None or done.returncode == expect)


def output(done):
    """Both streams of a finished process, as one string.

    Args:
        done (subprocess.CompletedProcess): The finished process.

    Returns:
        str: stdout and stderr joined.
    """
    return f'{done.stdout}\n{done.stderr}'


def open_store(config, variable):
    """Open a fixture store read only.

    Args:
        config (str): Path to the config naming the layout.
        variable (str): The variable whose store to open.

    Returns:
        xarray.Dataset: The decoded store.
    """
    settings = load_config(os.path.join(ROOT, config))
    settings['variable'] = variable
    repository = icechunk.Repository.open(
        icechunk.local_filesystem_storage(store_path(settings))
    )
    return xr.open_zarr(
        repository.readonly_session(branch=BRANCH).store, consolidated=False
    )


def fixture_stores(config):
    """Where a layout's fixture stores live.

    Args:
        config (str): Path to the config naming the layout.

    Returns:
        str: The directory holding that layout's stores.
    """
    settings = load_config(os.path.join(ROOT, config))
    settings['variable'] = VARIABLES[0]
    return os.path.dirname(store_path(settings))


def case_builds(results):
    """Build both layouts, each through a different form of the command.

    Args:
        results (Results): Where to record outcomes.
    """
    done, ok = run('gleam_zarr.py', '--config', SPATIAL)
    built = output(done).count('done :-)')
    results.check(
        'spatial: the bare command builds the whole family',
        ok and all(
            os.path.exists(os.path.join(fixture_stores(SPATIAL), n))
            for n in os.listdir(fixture_stores(SPATIAL))
        ) and built == 1,
        f'exit {done.returncode}, {len(os.listdir(fixture_stores(SPATIAL)))} stores',
    )

    statuses = []
    for variable in VARIABLES:
        done, ok = run('gleam_zarr.py', '--config', TEMPORAL, '--variable', variable)
        statuses.append(ok)
    results.check(
        'temporal: --variable builds one store at a time',
        all(statuses)
        and len(os.listdir(fixture_stores(TEMPORAL))) == len(VARIABLES),
        f'{len(os.listdir(fixture_stores(TEMPORAL)))} stores',
    )


def case_verifies(results):
    """Verify every store against the raw files.

    Args:
        results (Results): Where to record outcomes.
    """
    for config, layout in ((SPATIAL, 'spatial'), (TEMPORAL, 'temporal')):
        clean = []
        for variable in VARIABLES:
            done, ok = run(
                'verify_gleam_zarr.py', '--config', config,
                '--variable', variable, '--samples', '4',
            )
            clean.append(ok and 'store verified' in output(done))
        results.check(
            f'{layout}: every store verifies against raw',
            all(clean),
            f'{sum(clean)} of {len(VARIABLES)} clean',
        )


def case_cross_store_checks(results):
    """The checks that read sibling stores must fire, not skip.

    A phase that silently checks nothing still exits 0, so these assert on what
    the run reported rather than on its status.

    Args:
        results (Results): Where to record outcomes.
    """
    done, _ = run(
        'verify_gleam_zarr.py', '--config', TEMPORAL, '--variable', 'Ep',
        '--samples', '2',
    )
    text = output(done)
    results.check(
        'identity is read across the sibling component stores',
        'Ep == Ep_aero + Ep_rad' in text and 'identity read across 3 stores' in text,
        'Ep store reads Ep_aero and Ep_rad',
    )
    results.check(
        'absent tiles are cross-checked against the siblings',
        'absent-tile union taken from 3 sibling stores' in text,
    )

    done, _ = run(
        'verify_gleam_zarr.py', '--config', TEMPORAL, '--variable', 'E',
        '--samples', '2',
    )
    results.check(
        'an identity whose components are absent skips with a note',
        'identity not checked: no store found for' in output(done),
        'E has no component stores here',
    )

    done, ok = run(
        'verify_gleam_zarr.py', '--config', TEMPORAL, '--variable', 'Ep',
        '--phases', 'metadata', '--compare-with', SPATIAL,
    )
    results.check(
        'the two layouts describe the same data',
        ok and 'store verified' in output(done),
    )


def case_attributes(results):
    """Derived and per-variable attributes reach the store.

    Args:
        results (Results): Where to record outcomes.
    """
    dataset = open_store(TEMPORAL, 'Ep_aero')
    title = dataset.attrs.get('title', '')
    results.check(
        'title is derived from the long name, not the variable code',
        'potential evaporation flux from the aerodynamic component' in title
        and 'chunked for time series' in title,
        title[:78],
    )
    results.check(
        'title names the frequency in full, not as its canonical token',
        'daily average' in title and ' day ' not in title,
        title[:44],
    )
    results.check(
        'summary names the variable and its units, canonically',
        '(potential_evaporation_aerodynamic, mm d-1)'
        in dataset.attrs.get('summary', ''),
    )
    results.check(
        'variable_attrs reaches only the variable it names',
        'known_data_gaps' in open_store(TEMPORAL, 'E').attrs
        and 'known_data_gaps' not in dataset.attrs,
        'E has one, Ep_aero does not',
    )


def case_nomenclature(results):
    """A build lands on the style guide without any migration step.

    This is the half of the guide that a config cannot express: the array's name
    and its units come from the nomenclature key, and the strings GLEAM
    published have to survive beside them.

    Args:
        results (Results): Where to record outcomes.
    """
    named, attributed, sourced = [], [], []
    for config in (SPATIAL, TEMPORAL):
        for original in VARIABLES:
            canonical = canonical_variable(original)
            dataset = open_store(config, original)
            named.append(
                list(dataset.data_vars) == [canonical]
                and canonical in os.path.basename(store_path_for(config, original))
            )
            attrs = dataset[canonical].attrs
            attributed.append(
                attrs['units'] == variables()[original].units
                and attrs['long_name'] == variables()[original].long_name
                and attrs['standard_name'] == canonical
                and attrs['original_variable_name'] == original
                and attrs['original_units'] == 'mm.day-1'
                and attrs['original_long_name'].endswith('from GLEAM 4.3a')
            )
            # {original_variable} has to reach a template, or the provenance
            # attributes would name the store rather than the files it came from
            sourced.append(f'holding {original}' in dataset.attrs.get('source', ''))
    results.check(
        'the store and its array carry the canonical name',
        all(named),
        f'{sum(named)} of {len(named)} stores, '
        f'e.g. {os.path.basename(store_path_for(SPATIAL, "Ep_aero"))}',
    )
    results.check(
        'canonical units and long name are written, originals kept beside them',
        all(attributed),
        f'{sum(attributed)} of {len(attributed)} stores',
    )
    results.check(
        'source names the GLEAM variable, not the canonical one',
        all(sourced),
        f'{sum(sourced)} of {len(sourced)} stores',
    )

    frequency = open_store(SPATIAL, 'E').attrs
    results.check(
        'the frequency rename is recorded in the metadata',
        frequency.get('temporal_frequency') == 'day'
        and frequency.get('original_temporal_frequency') == 'daily',
        f'{frequency.get("original_temporal_frequency")} -> '
        f'{frequency.get("temporal_frequency")}',
    )


def case_rebuild(results):
    """A finished store is a no-op on append and a redo on region.

    The asymmetry is real and worth pinning: append resumes on timestep count,
    so a finished store has nothing to do, while region matches blocks by their
    bounds and redoes them all when the block shape has changed.

    Args:
        results (Results): Where to record outcomes.
    """
    done, ok = run('gleam_zarr.py', '--config', SPATIAL, '--variable', 'Ep')
    results.check(
        'rebuilding a finished append store writes nothing',
        ok and 'store is already complete' in output(done),
    )
    done, ok = run('gleam_zarr.py', '--config', TEMPORAL, '--variable', 'Ep')
    results.check(
        'rebuilding a finished region store is also a no-op at the same block shape',
        ok and 'store is already complete' in output(done),
        'same block_shape, so every block matches',
    )


def case_resume(results):
    """A half written store is resumed, not restarted.

    The store is rewound to an earlier commit rather than killed mid-write, so
    the case is deterministic instead of a race against a timeout.

    Args:
        results (Results): Where to record outcomes.
    """
    settings = load_config(os.path.join(ROOT, SPATIAL))
    settings['variable'] = 'Ep_rad'
    path = store_path(settings)
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    history = list(repository.ancestry(branch=BRANCH))
    # four commits in: the initial snapshot plus three batches
    repository.reset_branch(BRANCH, history[-4].id)
    rewound = xr.open_zarr(
        repository.readonly_session(branch=BRANCH).store, consolidated=False
    ).sizes['time']

    done, ok = run('gleam_zarr.py', '--config', SPATIAL, '--variable', 'Ep_rad')
    text = output(done)
    results.check(
        'a rewound store is resumed from its last commit',
        ok and f'store already holds {rewound} timesteps' in text
        and f'resuming from timestep {rewound}' in text,
        f'rewound to {rewound} of 731 timesteps',
    )
    done, ok = run(
        'verify_gleam_zarr.py', '--config', SPATIAL, '--variable', 'Ep_rad',
        '--samples', '4',
    )
    results.check(
        'the resumed store still matches raw',
        ok and 'store verified' in output(done),
    )


def case_time_axis_guard(results, tmp):
    """A variable that is a year short stops the build before it writes.

    Args:
        results (Results): Where to record outcomes.
        tmp (str): A scratch directory for the broken tree.
    """
    settings = load_config(os.path.join(ROOT, SPATIAL))
    source = os.path.join(
        settings['directories']['download'], 'v_4_3_a', 'raw',
        settings['temporal_resolution'],
    )
    broken_root = os.path.join(tmp, 'broken')
    raw = os.path.join(broken_root, 'v_4_3_a', 'raw', settings['temporal_resolution'])
    shutil.rmtree(broken_root, ignore_errors=True)
    for variable in ('E', 'Ep'):
        os.makedirs(os.path.join(raw, variable), exist_ok=True)
        names = sorted(os.listdir(os.path.join(source, variable)))
        # Ep loses a year, so it no longer matches the reference E
        for name in (names[:1] if variable == 'Ep' else names):
            os.symlink(
                os.path.join(source, variable, name),
                os.path.join(raw, variable, name),
            )

    config = os.path.join(tmp, 'broken.yaml')
    with open(os.path.join(ROOT, SPATIAL), encoding='utf-8') as handle:
        text = handle.read()
    text = text.replace(
        f'download: {settings["directories"]["download"]}',
        f'download: {broken_root}/',
    )
    with open(config, 'w', encoding='utf-8') as handle:
        handle.write(text)

    done, ok = run('gleam_zarr.py', '--config', config, '--variable', 'Ep', expect=1)
    wrote_nothing = not os.path.exists(os.path.join(broken_root, 'v_4_3_a', 'zarr'))
    results.check(
        'a short variable raises before anything is written',
        ok and 'the download may be incomplete' in output(done) and wrote_nothing,
        f'exit {done.returncode}, no store created: {wrote_nothing}',
    )


def case_published_guard(results):
    """A tagged store is refused unless forced.

    Args:
        results (Results): Where to record outcomes.
    """
    tag = 'fixture-published'
    settings = load_config(os.path.join(ROOT, SPATIAL))
    settings['variable'] = 'E'
    repository = icechunk.Repository.open(
        icechunk.local_filesystem_storage(store_path(settings))
    )
    if tag not in list(repository.list_tags()):
        tip = next(iter(repository.ancestry(branch=BRANCH)))
        repository.create_tag(tag, tip.id)

    done, ok = run('gleam_zarr.py', '--config', SPATIAL, '--variable', 'E', expect=1)
    results.check(
        'a published store is refused',
        ok and 'has been verified and published' in output(done),
        f'exit {done.returncode}',
    )
    done, ok = run('gleam_zarr.py', '--config', SPATIAL, '--variable', 'E', '--force')
    results.check(
        '--force overrides the refusal',
        ok and 'is being rebuilt anyway' in output(done),
    )
    repository.delete_tag(tag)


def store_path_for(config, original):
    """Where a layout's store for one variable lives.

    Args:
        config (str): Path to the config naming the layout.
        original (str): The variable as GLEAM publishes it, e.g. 'E'.

    Returns:
        str: The store's path, under its canonical name.
    """
    settings = load_config(os.path.join(ROOT, config))
    settings['variable'] = original
    return store_path(settings)


def rewind_to_legacy(config, original):
    """Put a freshly built store back into the shape the 2026-09 stores are in.

    The build is compliant now, so there is nothing left on disk for
    migrate_nomenclature.py to migrate -- and the script still has 28 real
    stores to run against. This manufactures one: the array goes back to GLEAM's
    name, its attributes go back to the strings GLEAM published, and the
    directory goes back to the old spelling.

    title and summary are replaced with a sentinel rather than the old wording.
    The point is not to reproduce the old text, it is to prove the migration
    *regenerates* them -- if they were left correct, the check that they come
    out right would pass without the migration having done anything.

    Args:
        config (str): Path to the config naming the layout.
        original (str): The variable as GLEAM publishes it, e.g. 'E'.

    Returns:
        str: The path the store now sits at, under its legacy name.
    """
    canonical = canonical_variable(original)
    path = store_path_for(config, original)
    repository = icechunk.Repository.open(icechunk.local_filesystem_storage(path))

    session = repository.rearrange_session(BRANCH)
    session.move(f'/{canonical}', f'/{original}')
    session.commit('rewind: back to the GLEAM variable name')

    session = repository.writable_session(BRANCH)
    group = zarr.open_group(session.store, mode='r+')
    attrs = dict(group[original].attrs)
    group[original].update_attributes({
        'standard_name': attrs['original_standard_name'],
        'long_name': attrs['original_long_name'],
        'units': attrs['original_units'],
    })
    # update_attributes merges, so the canonical-era keys have to go explicitly
    for key in ('original_variable_name', 'original_standard_name',
                'original_long_name', 'original_units', 'unit_conversion'):
        del group[original].attrs[key]
    group.update_attributes(
        dict(group.attrs) | {'title': 'LEGACY', 'summary': 'LEGACY'}
    )
    session.commit('rewind: back to the GLEAM variable attributes')

    legacy = os.path.join(
        os.path.dirname(path),
        os.path.basename(path).replace('.day.', '.daily.').replace(
            f'.{canonical}.zarr', f'.{original}.zarr'
        ),
    )
    os.rename(path, legacy)
    return legacy


def case_migration(results):
    """migrate_nomenclature.py reproduces a compliant build, without a rebuild.

    The claim the migration makes is not just "it renames things" -- it is that
    a store brought onto the guide is indistinguishable from one built onto it.
    So the check is an equivalence: build compliant, record what that produced,
    rewind the store to the legacy shape, migrate it, and require the result to
    match what the build made. Anything the two word differently shows up here.

    Runs last, because rewinding renames every fixture store.

    Args:
        results (Results): Where to record outcomes.
    """
    built = {}
    for config in (SPATIAL, TEMPORAL):
        for original in VARIABLES:
            canonical = canonical_variable(original)
            dataset = open_store(config, original)
            built[(config, original)] = {
                'attrs': dict(dataset[canonical].attrs),
                'title': dataset.attrs['title'],
                'summary': dataset.attrs['summary'],
                'values': dataset[canonical].values,
            }
            rewind_to_legacy(config, original)

    results.check(
        'a rewound store is found under its legacy name',
        not os.path.exists(store_path_for(SPATIAL, 'E'))
        and os.path.exists(
            store_path_for(SPATIAL, 'E').replace('.day.', '.daily.').replace(
                '.evaporation.zarr', '.E.zarr'
            )
        ),
        'legacy fixture manufactured',
    )

    done, ok = run('migrate_nomenclature.py', '--config', SPATIAL)
    results.check(
        'a dry run reports every step and writes nothing',
        ok
        and 'rename-array then attrs then rename-store' in output(done)
        and not os.path.exists(store_path_for(SPATIAL, 'E')),
        'store still at its legacy path',
    )

    applied = []
    for config in (SPATIAL, TEMPORAL):
        done, ok = run('migrate_nomenclature.py', '--config', config, '--apply')
        applied.append(ok and 'unchanged' in output(done))
    results.check(
        'both layouts migrate, and the script reports the store unchanged',
        all(applied),
        f'{sum(applied)} of 2 layouts',
    )

    same_attrs, same_prose, same_values = [], [], []
    for (config, original), before in built.items():
        canonical = canonical_variable(original)
        dataset = open_store(config, original)
        same_attrs.append(
            list(dataset.data_vars) == [canonical]
            and dict(dataset[canonical].attrs) == before['attrs']
        )
        same_prose.append(
            dataset.attrs['title'] == before['title']
            and dataset.attrs['summary'] == before['summary']
        )
        same_values.append(
            np.array_equal(
                dataset[canonical].values, before['values'], equal_nan=True
            )
        )
    results.check(
        'a migrated store has the variable attributes a built one has',
        all(same_attrs),
        f'{sum(same_attrs)} of {len(same_attrs)} stores identical',
    )
    results.check(
        'a migrated store has the title and summary a built one has',
        all(same_prose),
        f'{sum(same_prose)} of {len(same_prose)} regenerated from the sentinel',
    )
    results.check(
        'every value survives the migration bit-identically',
        all(same_values),
        f'{sum(same_values)} of {len(same_values)} stores unchanged',
    )

    done, ok = run('migrate_nomenclature.py', '--config', SPATIAL, '--apply')
    results.check(
        're-running a migrated store does nothing',
        ok and output(done).count('already migrated') == len(VARIABLES),
        'idempotent',
    )

    done, ok = run(
        'verify_gleam_zarr.py', '--config', SPATIAL, '--variable', 'evaporation',
        '--phases', 'structure,sweep',
    )
    results.check(
        'the verifier accepts a migrated store, addressed canonically',
        ok and 'store verified' in output(done),
        'exit 0',
    )


def parse_args():
    """Parse the command line.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='test the pipeline on the fixture.')
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Leave the fixture stores behind, to inspect what a case saw.',
    )
    return parser.parse_args()


def main():
    """Run every case and return a process exit status.

    Returns:
        int: 0 if every case passed, 1 otherwise.
    """
    args = parse_args()
    results = Results()
    tmp = os.path.join(
        load_config(os.path.join(ROOT, SPATIAL))['directories']['download'], 'tmp'
    )
    os.makedirs(tmp, exist_ok=True)

    # every case builds on the one before, so the stores start from nothing
    for config in (SPATIAL, TEMPORAL):
        shutil.rmtree(fixture_stores(config), ignore_errors=True)

    case_builds(results)
    case_verifies(results)
    case_cross_store_checks(results)
    case_attributes(results)
    case_nomenclature(results)
    case_rebuild(results)
    case_resume(results)
    case_time_axis_guard(results, tmp)
    case_published_guard(results)
    case_migration(results)

    if not args.keep:
        for config in (SPATIAL, TEMPORAL):
            shutil.rmtree(fixture_stores(config), ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\n{results.n} cases, {len(results.failures)} failures')
    for name in results.failures:
        print(f'  failed: {name}')
    return 1 if results.failures else 0


if __name__ == '__main__':
    sys.exit(main())
