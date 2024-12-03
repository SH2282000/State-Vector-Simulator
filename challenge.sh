module load python/3.11.6-gcc-7.5.0-jm3mdlq
module load glib/2.70.4/gcc-11.2.0-module-aiehnud
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/h039y10/.local/bin:$PATH"
poetry config virtualenvs.in-project true
poetry install
poetry run python setup.py build_ext --inplace
poetry run python QCP/template/benchmark.py