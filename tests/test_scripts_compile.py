import glob
import os
import py_compile


def test_python_scripts_compile():
    for path in glob.glob(os.path.join("scripts", "*.py")):
        py_compile.compile(path, doraise=True)

