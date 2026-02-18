#!/usr/bin/env python3
"""
onshape_export.py — Export an Onshape assembly to ASAPux format.

Produces:
    <output-dir>/
    ├── 0.obj          # one OBJ per part, in the part's local coordinate frame
    ├── 1.obj
    ├── ...
    └── config.json    # assembled transforms as final_state (and initial_state placeholder)

After export, run preprocessing:
    python assets/process_mesh.py \\
        --source-dir <output-dir> \\
        --target-dir <output-dir>/processed \\
        --subdivide

Usage examples:
    # From an Onshape assembly URL:
    python onshape_export.py \\
        --url "https://cad.onshape.com/documents/abc123/w/def456/e/ghi789" \\
        --output assets/my_assembly/original

    # From explicit IDs:
    python onshape_export.py \\
        --did abc123 --wid def456 --eid ghi789 \\
        --output assets/my_assembly/original

Authentication (choose one):
    1. Environment variables:
           export ONSHAPE_ACCESS_KEY="your-access-key"
           export ONSHAPE_SECRET_KEY="your-secret-key"
    2. Config file at ~/.config/onshape/credentials.json:
           {"access_key": "...", "secret_key": "..."}
    3. Command-line flags --access-key / --secret-key

Generate API keys at: https://dev-portal.onshape.com/keys
"""

import os
import sys
import json
import hmac
import hashlib
import base64
import argparse
import subprocess
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlencode, quote

import requests
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Onshape API client with HMAC-SHA256 authentication
# ---------------------------------------------------------------------------

class _OnshapeAuth(requests.auth.AuthBase):
    """Signs every request with Onshape API-key HMAC-SHA256."""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
        nonce = uuid4().hex

        parsed = urlparse(r.url)
        path = parsed.path
        query = parsed.query or ''
        content_type = r.headers.get('Content-Type', 'application/json')

        # Onshape signing string (all lowercase, newline-delimited)
        msg = '\n'.join([
            r.method.lower(),
            nonce,
            date,
            content_type.lower(),
            path.lower(),
            query.lower(),
            '',          # trailing newline
        ])

        sig = base64.b64encode(
            hmac.new(self.secret_key.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()

        r.headers['Authorization'] = f'On {self.access_key}:HmacSHA256:{sig}'
        r.headers['Date'] = date
        r.headers['On-Nonce'] = nonce
        r.headers['Content-Type'] = content_type
        return r


class OnshapeClient:
    BASE = 'https://cad.onshape.com'
    API  = '/api/v6'

    def __init__(self, access_key: str, secret_key: str):
        self.session = requests.Session()
        self.session.auth = _OnshapeAuth(access_key, secret_key)
        self.session.headers['Accept'] = 'application/json;charset=UTF-8; qs=0.09'

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None, accept: str | None = None):
        url = self.BASE + self.API + path
        headers = {'Accept': accept} if accept else {}
        resp = self.session.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        return self._get(path, params).json()

    # ------------------------------------------------------------------
    # Assembly endpoints
    # ------------------------------------------------------------------

    def get_assembly(self, did: str, wid: str, eid: str) -> dict:
        """Return the full assembly definition (instances + occurrences)."""
        return self._get_json(
            f'/assemblies/d/{did}/w/{wid}/e/{eid}',
            params={
                'includeMateFeatures': 'false',
                'includeNonSolids': 'false',
                'includeMateConnectors': 'false',
            },
        )

    # ------------------------------------------------------------------
    # Part export endpoints
    # ------------------------------------------------------------------

    def export_part_obj(
        self,
        did: str,
        wvm_type: str,
        wvm_id: str,
        eid: str,
        part_id: str,
        units: str = 'centimeter',
    ) -> bytes:
        """Export a single part as OBJ bytes (in part-local coordinates)."""
        path = (
            f'/parts/d/{did}/{wvm_type}/{wvm_id}'
            f'/e/{eid}/partid/{quote(part_id, safe="")}/export'
        )
        return self._get(
            path,
            params={
                'format': 'OBJ',
                'units': units,
                'grouping': 'true',
                'scale': '1.0',
                'angleTolerance': '0.1090830782',
                'chordTolerance': '0.001',
            },
            accept='application/octet-stream',
        ).content


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

UNIT_SCALE = {          # metres → target unit
    'meter':      1.0,
    'centimeter': 100.0,
    'millimeter': 1000.0,
    'inch':       39.3701,
    'foot':       3.28084,
    'yard':       1.09361,
}


def onshape_matrix_to_state(col_major_16: list, unit_scale: float) -> list:
    """
    Convert a 16-element column-major Onshape transform (metres) to
    ASAPux state = [tx, ty, tz, rx, ry, rz].

    Translation is scaled to the target unit.
    Rotation is XYZ Euler in radians (as required by assets/transform.py).
    """
    mat = np.array(col_major_16, dtype=float).reshape(4, 4).T  # col-major → row-major
    translation = mat[:3, 3] * unit_scale
    euler = Rotation.from_matrix(mat[:3, :3]).as_euler('xyz')
    return translation.tolist() + euler.tolist()


def parse_onshape_url(url: str):
    """
    Parse an Onshape URL into (did, wvm_type, wvm_id, eid).

    Supported formats:
        https://cad.onshape.com/documents/{did}/w/{wid}/e/{eid}
        https://cad.onshape.com/documents/{did}/v/{vid}/e/{eid}
    """
    parts = url.rstrip('/').split('/')
    try:
        doc_idx = parts.index('documents')
        did = parts[doc_idx + 1]

        wvm_type, wvm_id = None, None
        for t in ('w', 'v', 'm'):
            if t in parts:
                idx = parts.index(t)
                wvm_type, wvm_id = t, parts[idx + 1]
                break

        eid = None
        if 'e' in parts:
            eid = parts[parts.index('e') + 1]

        if not all([did, wvm_type, wvm_id, eid]):
            raise ValueError

        return did, wvm_type, wvm_id, eid
    except (ValueError, IndexError):
        raise ValueError(
            f'Cannot parse Onshape URL: {url}\n'
            'Expected: https://cad.onshape.com/documents/DID/w/WID/e/EID'
        )


def load_credentials() -> tuple[str, str]:
    """Load API credentials from environment variables or config file."""
    ak = os.environ.get('ONSHAPE_ACCESS_KEY')
    sk = os.environ.get('ONSHAPE_SECRET_KEY')

    if not (ak and sk):
        cfg = Path.home() / '.config' / 'onshape' / 'credentials.json'
        if cfg.exists():
            data = json.loads(cfg.read_text())
            ak = data.get('access_key')
            sk = data.get('secret_key')

    if not (ak and sk):
        raise ValueError(
            'Onshape credentials not found.\n\n'
            'Option 1 — environment variables:\n'
            '    export ONSHAPE_ACCESS_KEY="..."\n'
            '    export ONSHAPE_SECRET_KEY="..."\n\n'
            'Option 2 — config file (~/.config/onshape/credentials.json):\n'
            '    {"access_key": "...", "secret_key": "..."}\n\n'
            'Generate keys at https://dev-portal.onshape.com/keys'
        )
    return ak, sk


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_assembly(
    client: OnshapeClient,
    did: str,
    wid: str,
    eid: str,
    output_dir: str,
    units: str = 'centimeter',
    verbose: bool = True,
) -> bool:
    """
    Export all parts of an Onshape assembly to ASAPux-compatible format.

    Returns True on full success, False if any parts failed.
    """
    unit_scale = UNIT_SCALE[units]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fetch assembly definition
    # ------------------------------------------------------------------
    if verbose:
        print(f'Fetching assembly definition  (did={did}, wid={wid}, eid={eid}) …')

    asm = client.get_assembly(did, wid, eid)
    instances   = asm.get('instances', [])
    occurrences = asm.get('occurrences', [])
    root_asm    = asm.get('rootAssembly', {})

    if verbose:
        print(f'  {len(instances)} instances, {len(occurrences)} occurrences')

    # ------------------------------------------------------------------
    # 2. Build occurrence-by-path lookup (path is a list of instance IDs)
    # ------------------------------------------------------------------
    occ_by_path: dict[tuple, dict] = {}
    for occ in occurrences:
        occ_by_path[tuple(occ['path'])] = occ

    # ------------------------------------------------------------------
    # 3. Filter to leaf Part instances only
    # ------------------------------------------------------------------
    part_instances = [i for i in instances if i.get('type') == 'Part']

    if not part_instances:
        print('ERROR: No Part instances found in this assembly.')
        print('  • Make sure the element ID points to an Assembly (not a Part Studio).')
        return False

    if verbose:
        print(f'\n{len(part_instances)} parts found:')
        for idx, inst in enumerate(part_instances):
            print(f'  [{idx}] {inst.get("name", "unnamed")}')

    # ------------------------------------------------------------------
    # 4. Export each part
    # ------------------------------------------------------------------
    config: dict[str, dict] = {}
    failures: list[tuple] = []

    for idx, instance in enumerate(part_instances):
        inst_id   = instance['id']
        inst_name = instance.get('name', f'part_{idx}')
        part_did  = instance.get('documentId', did)
        part_eid  = instance.get('elementId', eid)
        part_pid  = instance.get('partId', '')

        # Determine workspace/version/microversion for this part's document
        if part_did == did:
            wvm_type, wvm_id = 'w', wid
        elif instance.get('documentVersion'):
            wvm_type, wvm_id = 'v', instance['documentVersion']
        elif instance.get('documentMicroversion'):
            wvm_type, wvm_id = 'm', instance['documentMicroversion']
        else:
            wvm_type, wvm_id = 'w', wid

        # Resolve assembled transform from occurrences
        occ = occ_by_path.get((inst_id,))
        if occ is None:
            # Shallow search: find any occurrence whose path starts with inst_id
            for path_key, occ_val in occ_by_path.items():
                if path_key and path_key[0] == inst_id:
                    occ = occ_val
                    break

        if occ and 'transform' in occ:
            assembled_state = onshape_matrix_to_state(occ['transform'], unit_scale)
        else:
            if verbose:
                print(f'  [{idx}] WARNING: no transform found for "{inst_name}", using identity')
            assembled_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if verbose:
            print(f'\n  [{idx}] {inst_name}')
            print(f'       document : {part_did[:8]}…  element: {part_eid[:8]}…  part: {part_pid[:12]}')
            tx, ty, tz = assembled_state[:3]
            print(f'       position ({units}): [{tx:.4f}, {ty:.4f}, {tz:.4f}]')

        # Export OBJ
        try:
            obj_bytes = client.export_part_obj(
                part_did, wvm_type, wvm_id, part_eid, part_pid, units=units
            )

            obj_path = os.path.join(output_dir, f'{idx}.obj')
            with open(obj_path, 'wb') as f:
                f.write(obj_bytes)

            if verbose:
                print(f'       saved  → {obj_path}  ({len(obj_bytes):,} bytes)')

            config[str(idx)] = {
                'initial_state': assembled_state,   # ← staging area; edit to spread parts out
                'final_state':   assembled_state,   # ← assembled configuration from Onshape
                '_name':         inst_name,         # informational only; stripped below
            }

        except requests.HTTPError as e:
            print(f'  [{idx}] HTTP {e.response.status_code} while exporting "{inst_name}": {e}')
            failures.append((idx, inst_name, str(e)))
        except Exception as e:
            print(f'  [{idx}] Error exporting "{inst_name}": {e}')
            failures.append((idx, inst_name, str(e)))

    # ------------------------------------------------------------------
    # 5. Write config.json  (strip internal _name key)
    # ------------------------------------------------------------------
    config_clean = {
        pid: {'initial_state': v['initial_state'], 'final_state': v['final_state']}
        for pid, v in config.items()
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_clean, f, indent=4)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total   = len(part_instances)
    success = total - len(failures)
    print(f'\n{"="*60}')
    print(f'Exported {success}/{total} parts → {output_dir}')
    print(f'Config   → {config_path}')

    if failures:
        print(f'\nFailed parts:')
        for idx, name, err in failures:
            print(f'  [{idx}] {name}: {err}')

    print(f'''
IMPORTANT — next steps:
  1. final_state   = the assembled configuration (from Onshape). ✓ Done.
  2. initial_state = where each part sits BEFORE assembly (staging area).
     Currently set to the same as final_state.
     Edit config.json to place parts in a flat pre-assembly layout, e.g.:
       "0": {{"initial_state": [x, y, 0, 0, 0, 0], "final_state": [...]}}

  3. Run mesh preprocessing (watertightness check + subdivision):
       python assets/process_mesh.py \\
           --source-dir {output_dir} \\
           --target-dir {output_dir}/processed \\
           --subdivide

  4. Run the planner:
       python plan_sequence/run_seq_plan.py \\
           --assembly-dir {output_dir}/processed
''')

    return len(failures) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Export an Onshape assembly to ASAPux format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--url',  type=str, help='Full Onshape assembly URL')
    src.add_argument('--did',  type=str, help='Onshape document ID')

    parser.add_argument('--wid', type=str, help='Workspace ID (required with --did)')
    parser.add_argument('--eid', type=str, help='Assembly element ID (required with --did)')

    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output directory (e.g. assets/my_assembly/original)',
    )
    parser.add_argument(
        '--units', type=str, default='centimeter',
        choices=list(UNIT_SCALE.keys()),
        help='Length units for exported OBJ geometry and config.json (default: centimeter)',
    )
    parser.add_argument('--access-key', type=str, help='Onshape API access key')
    parser.add_argument('--secret-key', type=str, help='Onshape API secret key')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Resolve document / workspace / element IDs
    if args.url:
        did, wvm_type, wvm_id, eid = parse_onshape_url(args.url)
        if wvm_type != 'w':
            print(f"Warning: URL uses '{wvm_type}' (not a workspace 'w'). "
                  "Export works best from a workspace URL.")
        wid = wvm_id
    else:
        if not (args.wid and args.eid):
            parser.error('--wid and --eid are required when --did is used')
        did, wid, eid = args.did, args.wid, args.eid

    # Credentials
    if args.access_key and args.secret_key:
        access_key, secret_key = args.access_key, args.secret_key
    else:
        try:
            access_key, secret_key = load_credentials()
        except ValueError as e:
            print(f'Error: {e}', file=sys.stderr)
            sys.exit(1)

    client = OnshapeClient(access_key, secret_key)

    try:
        ok = export_assembly(
            client, did, wid, eid,
            output_dir=args.output,
            units=args.units,
            verbose=not args.quiet,
        )
        sys.exit(0 if ok else 1)
    except requests.HTTPError as e:
        print(f'HTTP {e.response.status_code}: {e.response.text[:400]}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nAborted.')
        sys.exit(1)


if __name__ == '__main__':
    main()
