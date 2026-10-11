# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

block_cipher = None

# license notices of bundled packages that PyInstaller does not collect on its own
datas = collect_data_files('cv2', includes=['LICENSE*.txt'])
for package in ('scipy', 'psutil', 'colorama'):
    datas += copy_metadata(package)

a = Analysis(['eclipse-aligner.py'],
             pathex=['C:\\Users\\Hotdogee\\Dropbox\\Personal\\Projects\\github\\solar-eclipse-timelapse-aligner'],
             binaries=[],
             datas=datas,
             hiddenimports=['scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
# eclipse-aligner only reads and writes JPGs, leave out the FFmpeg (LGPL) video backend of OpenCV (about 30 MB)
a.binaries = [b for b in a.binaries if 'opencv_videoio_ffmpeg' not in b[0]]
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='eclipse-aligner',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='eclipse-aligner')
