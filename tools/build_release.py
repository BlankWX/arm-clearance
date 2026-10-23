"""Build deterministic install and source ZIPs using an explicit public allowlist."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
STAMP = (1980, 1, 1, 0, 0, 0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_zip(path, entries):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None


def main():
    tree = ast.parse((ROOT / 'arm_clearance' / '__init__.py').read_bytes())
    info = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                        and target.id == 'bl_info' for target in node.targets))
    version = '.'.join(map(str, info['version']))
    license_bytes = (ROOT / 'LICENSE').read_bytes()
    assert b'GNU GENERAL PUBLIC LICENSE' in license_bytes and b'Version 3, 29 June 2007' in license_bytes
    modules = sorted((ROOT / 'arm_clearance').glob('*.py'))
    assert len(modules) == 16, 'Review the release file list after changing module layout'
    install_entries = {}
    for path in modules:
        data = path.read_bytes()
        compile(data, path.name, 'exec')
        install_entries['arm_clearance/' + path.name] = data
    install_entries['arm_clearance/LICENSE'] = license_bytes

    public_paths = [ROOT / name for name in ('.gitignore', '.gitattributes', 'README.md', 'CHANGELOG.md', 'LICENSE')]
    for directory, suffix in (('arm_clearance', '.py'), ('docs', '.md'), ('tests', '.py'),
                              ('tools', '.py'), ('.github/ISSUE_TEMPLATE', '.md')):
        public_paths.extend(sorted((ROOT / directory).glob('*' + suffix)))
    assert all(path.is_file() for path in public_paths)
    assert len(set(public_paths)) == len(public_paths)
    source_entries = {'arm-clearance-v' + version + '/' + path.relative_to(ROOT).as_posix(): path.read_bytes()
                      for path in public_paths}
    DIST.mkdir(exist_ok=True)
    install_zip = DIST / ('arm_clearance-v' + version + '.zip')
    source_zip = DIST / ('arm-clearance-v' + version + '-source.zip')
    write_zip(install_zip, install_entries)
    write_zip(source_zip, source_entries)
    archives = {path.name: sha(path.read_bytes()) for path in (install_zip, source_zip)}
    manifest = {'version': list(info['version']), 'author': info['author'], 'license': 'GPL-3.0-or-later',
                'tested_blender': '5.2.2 LTS', 'module_count': len(modules),
                'source_sha256': {path.name: sha(path.read_bytes()) for path in modules},
                'license_sha256': sha(license_bytes), 'archives_sha256': archives,
                'source_archive_files': sorted(path.relative_to(ROOT).as_posix() for path in public_paths)}
    (DIST / 'release-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (DIST / 'SHA256SUMS.txt').write_text(''.join(digest+'  '+name+'\n' for name, digest in archives.items()), encoding='utf-8')
    print(json.dumps({'version': version, 'modules': len(modules), 'source_files': len(public_paths),
                      'archives_sha256': archives}, ensure_ascii=False))


if __name__ == '__main__':
    main()
