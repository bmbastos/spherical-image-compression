# This makefile was created by Bruno Bastos.

# Para utilizar make no Windows, é necessário utilizar a ferramenta MinGW "mingw32-make.exe"
# Detectar sistema operacional
ifeq ($(OS),Windows_NT)
	OS_NAME := Windows
	PYTHON := python
	PIP := pip
else
	OS_NAME := $(shell uname -s)
	PYTHON := python3
	PIP := pip3
endif

# Diretórios
SRC_DIR = src
TEST_DIR = tests

# Nome do ambiente virtual
VENV = .venv

# Comandos
.PHONY: install

# Criar ambiente virtual
install:
	$(PYTHON) -m venv $(VENV)
	@echo - Ambiente virtual criado com sucesso!
ifeq ($(OS_NAME),Windows)
	@echo - Para ativar o ambiente virtual, execute:
	@echo ".venv\Scripts\activate"
else
	@echo - Para ativar o ambiente virtual, execute: 
	@echo "source .venv/bin/activate"
endif
	@echo - Apos a ativacao do ambiente virtual, execute: 
	@echo "$(PIP) install -r requirements.txt"
	@echo - Quando travar, sera necessario segurar Enter para continuar a instalacao
	@echo - Good Luck! :)


run: 
	$(PYTHON) main.py

test:
	$(PYTHON) -m unittest discover $(TEST_DIR)

lint:
	$(PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR)

format:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + || cmd /c "for /d %i in (__pycache__) do rmdir /s /q %i"
	find . -type d -name ".mypy_cache" -exec rm -r {} + || cmd /c "for /d %i in (.mypy_cache) do rmdir /s /q %i"
	find . -type d -name ".pytest_cache" -exec rm -r {} + || cmd /c "for /d %i in (.pytest_cache) do rmdir /s /q %i"
	rm -rf $(VENV) || rmdir /s /q $(VENV)

build: clean install
