# -*- mode: python ; coding: utf-8 -*-
import os
import site
import sys

block_cipher = None

# Get the correct site-packages directory
site_packages = site.getsitepackages()[0]

# Get the project root directory
project_root = os.path.abspath(os.getcwd())

a = Analysis(
    ['agent/run.py'],  # Point to the correct run.py location
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(site_packages, "altair/vegalite/v5/schema/vega-lite-schema.json"),
         "altair/vegalite/v5/schema/"),
        (os.path.join(site_packages, "streamlit"), "streamlit"),
        ("agent", "agent"),
        ("utils", "utils"),
        (".env", "."),
        ("results", "results"),
    ],
    hiddenimports=[
        'streamlit',
        'altair',
        'pandas',
        'numpy',
        'plotly',
        'matplotlib',
        'seaborn',
        'openpyxl',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AMR_Classifier',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
) 