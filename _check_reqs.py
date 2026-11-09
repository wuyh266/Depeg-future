"""Check if requirements.txt covers ALL imports used by pipeline/src/ .py files."""
import sys, os, ast, subprocess, importlib, pkgutil

PIPELINE = r'D:\PycharmProjects\DepegDetector\pipeline'

# 1. Collect all import statements from pipeline/src/*.py
src_dir = os.path.join(PIPELINE, 'src')
used_imports = set()
for f in sorted(os.listdir(src_dir)):
    if not f.endswith('.py'): continue
    fp = os.path.join(src_dir, f)
    with open(fp, encoding='utf-8') as fh:
        try:
            tree = ast.parse(fh.read(), filename=f)
        except SyntaxError:
            print(f"  SKIP (syntax): {f}")
            continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                used_imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                used_imports.add(node.module.split('.')[0])

# 2. Read requirements.txt
with open(os.path.join(PIPELINE, 'requirements.txt'), encoding='utf-8') as f:
    req_pkgs = set()
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-r'):
            continue
        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
        if pkg:
            req_pkgs.add(pkg.lower())

print(f"Found {len(used_imports)} unique top-level import names in src/")
print(f"Found {len(req_pkgs)} packages in requirements.txt")
print()

# 3. Classify each import
stdlib_packages = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

import_names_pkg_map = {}  # import name -> requirement.txt package name (lower)
# Known mappings where import name != pip package name
KNOWN_MAPPINGS = {
    'yaml': 'pyyaml',
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'bs4': 'beautifulsoup4',
    'lzma': None,  # stdlib
}

# Build reverse map: lower(package) -> lower(package)
req_pkgs_norm = {}
for p in req_pkgs:
    key = p.replace('-', '_').replace('.', '_')
    req_pkgs_norm[key] = p

third_party_missing = []

for imp_name in sorted(used_imports):
    if imp_name.startswith('_'):
        continue
    
    # Skip if it's a local src/ module
    if os.path.exists(os.path.join(src_dir, f'{imp_name}.py')):
        continue
    if imp_name in ('src',):
        continue
    
    # Check stdlib
    is_stdlib = False
    try:
        spec = importlib.util.find_spec(imp_name)
        if spec and hasattr(spec, 'submodule_search_locations'):
            # Check if it's in stdlib known list
            pass
    except (ValueError, ImportError):
        pass
    
    # Better stdlib check
    if imp_name in stdlib_packages:
        continue
    
    # Known stdlib
    stdlib_names = {'os', 'sys', 're', 'json', 'math', 'time', 'datetime', 
                    'typing', 'abc', 'collections', 'itertools', 'functools',
                    'pathlib', 'shutil', 'subprocess', 'pickle', 'io', 'textwrap',
                    'copy', 'warnings', 'ast', 'argparse', 'statistics',
                    'enum', 'dataclasses', 'random', 'struct', 'uuid', 'csv',
                    'traceback', 'bisect', 'operator', 'string', 'hashlib',
                    'inspect', 'logging', 'tempfile', 'gc', 'numbers',
                    'weakref', 'threading', 'multiprocessing', 'concurrent',
                    'contextlib', 'types', 'platform', 'glob', 'heapq', 'configparser',
                    'decimal', 'fractions', 'pprint', 'zipfile', 'tarfile',
                    'http', 'urllib', 'xml', 'html', 'base64', 'binascii',
                    'difflib', 'filecmp', 'linecache', 'locale', 'sched',
                    'secrets', 'shelve', 'smtplib', 'socket', 'socketserver',
                    'ssl', 'symtable', 'tabnanny', 'token', 'tokenize',
                    'wave', 'weakref', 'webbrowser', 'getpass', 'gettext'}
    
    # comprehensive stdlib for Python 3.10
    comprehensive_stdlib = {
        'os', 'sys', 're', 'json', 'math', 'time', 'datetime', 'typing', 'abc',
        'collections', 'itertools', 'functools', 'pathlib', 'shutil', 'subprocess',
        'pickle', 'io', 'textwrap', 'copy', 'warnings', 'ast', 'argparse',
        'statistics', 'enum', 'dataclasses', 'random', 'struct', 'uuid', 'csv',
        'traceback', 'bisect', 'operator', 'string', 'hashlib', 'inspect',
        'logging', 'tempfile', 'gc', 'numbers', 'weakref', 'threading',
        'multiprocessing', 'concurrent', 'contextlib', 'types', 'platform',
        'glob', 'heapq', 'configparser', 'decimal', 'fractions', 'pprint',
        'zipfile', 'tarfile', 'http', 'urllib', 'xml', 'html', 'base64',
        'binascii', 'difflib', 'filecmp', 'linecache', 'locale', 'sched',
        'secrets', 'shelve', 'socket', 'ssl', 'token', 'tokenize', 'wave',
        'getpass', 'gettext', 'dis', 'doctest', 'ctypes', 'cmd', 'calendar',
        'bz2', 'gzip', 'lzma', 'zlib', 'zipimport', 'unicodedata',
        'tty', 'turtle', 'this', 'sunaudio', 'imghdr', 'colorsys',
        'cgitb', 'chunk', 'cmath', 'codecs', 'code', 'codeop', 'atexit',
        'antigravity', 'ensurepip', 'venv', 'distutils', 'lib2to3',
        'msvcrt', 'winreg', 'winsound', 'imp', 'importlib', 'pkgutil',
        'pdb', 'profile', 'pstats', 'cProfile', 'timeit', 'trace',
        'unittest', 'test', 'compileall', 'py_compile', 'pyclbr',
        'mailbox', 'mailcap', 'mimetypes', 'base64', 'binhex', 'binascii',
        'quopri', 'uu', 'xdrlib', 'netrc', 'getpass', 'gettext',
        'dbm', 'sqlite3', 'pickletools', 'rlcompleter', 'builtins',
        '__future__', '__main__', 'ntpath', 'posixpath', 'genericpath',
        'errno', 'signal', 'faulthandler', 'socket', 'ssl', 'select',
        'selectors', 'asyncio', 'email', 'smtpd', 'poplib', 'imaplib',
        'nntplib', 'ftplib', 'telnetlib', 'smtplib', 'uuid',
        'socketserver', 'http', 'urllib', 'xmlrpc', 'ipaddress',
        'html', 'xml', 'webbrowser', 'cgi', 'wsgiref',
    }
    
    if imp_name in comprehensive_stdlib:
        continue
    
    # Check if import name matches a requirements.txt package
    imp_name_norm = imp_name.replace('-', '_').lower()
    
    # Check direct match
    found = False
    for req_key, req_orig in req_pkgs_norm.items():
        if imp_name_norm == req_key:
            found = True
            break
        # Handle cases like scikit-learn -> sklearn
        req_short = req_key.replace('scikit_', '').replace('python_', '')
        if imp_name_norm == req_short:
            found = True
            break
    
    if found:
        continue
    
    # Known special cases
    if imp_name in ('tqdm',): # tqdm.auto
        found = True
        continue
    
    if not found:
        third_party_missing.append(imp_name)

print(f"=== Third-party imports NOT in requirements.txt ===")
if third_party_missing:
    for m in sorted(third_party_missing):
        print(f"  MISSING: {m}")
    print(f"\nTotal missing: {len(third_party_missing)}")
else:
    print("  (none - all covered)")

# Also check for unused packages in requirements.txt
print(f"\n=== Packages in requirements.txt NOT directly used in src/ ===")
# Note: some may be transitive deps or needed at runtime
req_not_directly_used = []
for req_pkg in sorted(req_pkgs):
    req_key = req_pkg.replace('-', '_').lower()
    found = False
    for imp_name in used_imports:
        imp_key = imp_name.replace('-', '_').lower()
        if imp_key == req_key:
            found = True
            break
        # reverse mapping
    # try mapping import name to pip name
    import_variants = [
        req_pkg.lower(),
        req_pkg.lower().replace('-', '_'),
        req_pkg.lower().replace('_', '-'),
    ]
    # special reverse mappings
    if req_pkg.lower() == 'pyyaml': import_variants.append('yaml')
    if req_pkg.lower() == 'scikit-learn': import_variants.append('sklearn')
    if req_pkg.lower() == 'pillow': import_variants.append('pil')
    if req_pkg.lower() == 'beautifulsoup4': import_variants.append('bs4')
    if req_pkg.lower() == 'python-pptx': import_variants.append('pptx')
    
    found = False
    for imp_name in used_imports:
        for variant in import_variants:
            if imp_name.lower() == variant.replace('_', '').replace('-', ''):
                found = True
                break
        if found:
            break
    
    if not found:
        req_not_directly_used.append(req_pkg)

if req_not_directly_used:
    print(f"  (transitive / runtime-only packages - {len(req_not_directly_used)} items)")
    for m in req_not_directly_used:
        print(f"    {m}")
else:
    print("  (all directly used)")
