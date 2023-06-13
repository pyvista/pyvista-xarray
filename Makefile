# Simple makefile to simplify repetitive build env management tasks under posix

CODESPELL_DIRS ?= ./
CODESPELL_SKIP ?= "*.pyc,*.txt,*.gif,*.svg,*.css,*.png,*.jpg,*.ply,*.vtk,*.vti,*.vts,*.vtr,*.js,*.html,*.doctree,*.ttf,*.woff,*.woff2,*.eot,*.mp4,*.inv,*.pickle,*.ipynb,flycheck*,./.git/*,./.hypothesis/*,*.yml,./doc/_build/*,./doc/images/*,./dist/*,*~,.hypothesis*,./doc/examples/*,*.mypy_cache/*,*cover,./tests/tinypages/_build/*,*/_autosummary/*"


stylecheck: codespell lint

codespell:
	@echo "Running codespell"
	@codespell $(CODESPELL_DIRS) -S $(CODESPELL_SKIP)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle pvxarray --match='(?!coverage).*.py'

doctest:
	@echo "Running module doctesting"
	pytest -v --doctest-modules pvxarray

lint:
	@echo "Linting with flake8"
	flake8 pvxarray tests examples

format:
	@echo "Formatting"
	black .
	isort .
