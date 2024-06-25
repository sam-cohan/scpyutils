scpyutils
=========
These are a collection of useful utilities that I have developed since I first started
using Python as my primary programming language in 2013. Over the years, I have refined
these utilities to be somewhat generic and useful for others but admittedly they are
still somewhat heavily customized for my own particular use cases.

Feel free to use at your own risk!

Note that even though this project has a `setup.py` file for easy `pip install .` usage,
the setup.py does not include all possible dependencies for all modules. For best
results, just copy paste the module into your own repo!

You can also install the repo directly as follows:
```
pip install -e git+https://github.com/sam-cohan/scpyutils.git#egg=scpyutils
```


Setup Instructions for local development
----------------------------------------
### Install Correct Python Version
Install `pyenv-virtualenv`:

On Mac OSX, you can use [homebrew](https://brew.sh/):
```
brew install pyenv-virtualenv
```

You will need to add the following to your shell rc file (e.g. `~/.zshrc`):
```
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi
```

Restart your shell.
You can check what versions are available by running:
```
pyenv install --list
```
Install the latest stable version of python
```
pyenv install 3.11.6
```

### Create a Virtual Environment and Install Dependencies
Create a virtual environment using this project and activate it
```
pyenv virtualenv 3.11.6 scpyutils-env
pyenv activate scpyutils-env
```

Optionally, you can install `autoenv` using brew to auto-enable this virtual environment every time you `cd` into this project folder
```
brew install autoenv
```
autoenv will run the `.env` file in the most recent folder you `cd` into. Since `.env` is also the file used by the commonly used python library `dotenv`, it is wise to change the default for autoenv to be `.autoenv`. You can achieve this by adding the following environment variable to your shell rc file (e.g. ~/.zshrc):
```
export AUTOENV_ENV_FILENAME=.autoenv
```
You can then add a .autoenv file to your project root directory:
```
pyenv activate scpyutils-env
```

Instead of using `pip`, we use [poetry](https://python-poetry.org/docs/) as our package manager. You can install poetry using brew:
```
brew install poetry
```

Before using poetry to install the project requirements, make sure you are in the right python virtual-env by running:
```
(scpyutils-env) ➜  ai-eval git:(master) ✗ pyenv versions
  system
  3.10.13
  3.11.6
  3.11.6/envs/scpyutils-env
* scpyutils-env --> /Users/scohan/.pyenv/versions/3.11.6/envs/scpyutils-env (set by PYENV_VERSION environment variable)
```
If you are not in the right environment, you can switch to it by running:
```
pyenv activate scpyutils-env
```
Once you are confident that you are in the right python virtual environment, go ahead and install project dependencies:
```
poetry install --no-root
```
The above command resolves all the project requirements inside `pyproject.toml` file and creates a detailed `poetry.lock` file which we will check in. The idea is that given some high level version requirements, poetry will run a complete dependency resolution and then write the full list of exact dependency requirements to the `poetry.lock` file to make it faster and more accurate to install exact dependency versions.

For a nice tutorial on the main features of poetry you can check [this article](https://realpython.com/dependency-management-python-poetry/).

#### Some useful Poetry commands:
To add a new dependency:
```
poetry add <lib_name>
```
To add a new dependency to dev:
```
poetry add <lib_name> --group dev
```

#### Important note on poetry add
By default when you do `poetry add <lib_name>`, poetry will add it to your pyproject.toml file with a carat version (e.g. `pytest-asyncio = "^0.23.3"`) meaning only the minor part of the version is allowed to be different. While this default versioning makes it less likely that a `poetry update` command would cause breakages, it leads to very out-of-date library versions over time, so we prefer to manually edit the pyproject.toml file after each add to make sure the library versions are updated to be `>=` (e.g. `pytest-asyncio > ">=0.23.3"`). We will still ensure that exact library versions run in production by checking in updated `poetry.lock` file (using `poetry lock` command).


### Linters and Pre-Commit Hooks
#### Use `black` and `isort` for autoformatting
We make use of `black` and `isort` for auto-formatting. The configuration for both of these is kept inside [pyproject.toml](./pyproject.toml) file under `[too.black]` and `[too.isort]` respectively. Note that we will explicitly limit the `line-length` to 88 to keep the source files more readable with two side-by-side windows. `isort` will use `black` profile to avoid having conflicts.

### Use `flake8` and `mypy` for linting
We will use `flake8` and `mypy` for linting. The configuration files for these will be in [.flake8](./.flake8) and [.mypy.ini](./.mypy.ini) respectively because their compatibility with pyproject.toml is too finnicky.

`flake8` will flag a combination of formatting and potential logical errors in the code (i.e. it is a general linter) and `mypy` will point out any logical errors which may result specifically from `typing` (i.e. it is a typing linter).

Note that for `flake8` we ignore some specific conventions a couple of checks to make it compatible with `black`:
- `E203`: Whitespace before ':' (colon). This is ignored to be compatible with the way `black` formats slices. `black` may put spaces around the colon which doesn't conform to PEP 8, so E203 is often ignored when using `black`.
- `W503`: Line break occurred before a binary operator. This warning is ignored because `black` places binary operators after the line break, which is in conflict with PEP 8's recommendation (before W503 was updated in the PEP). However, the current PEP 8 guidance suggests that either option is acceptable, so ignoring W503 is reasonable.

For mypy, we will not apply a global `ignore_missing_imports = True`. Instead, if we get any imports that cause this error, we will first try to find community stubs and/or add our own.

#### Install pre-commit hooks
Make sure to install pre-commit hooks:
```
pip install pre-commit && pre-commit install
```
If you ever have to skip a specific commit hook for a good reason (e.g. emergency release or intermittent commit to save work at the end of the day), you can either skip all hooks by using the `--no-verify` flag, or you can skip a specific test using the SKIP env-var. e.g. to skip `mypy`, you can do:
```
SKIP=mypy git commit -m "checkpoint"