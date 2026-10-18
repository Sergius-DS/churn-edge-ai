# Makefile
# Equivalente a run_all.py
#
# Uso:
#   make
#   make all
#
# Para usar otro intérprete de Python:
#   make PYTHON=python3
#   make PYTHON=.venv/Scripts/python.exe

PYTHON = python
SCRIPTS_DIR = scripts

SCRIPTS = $(SCRIPTS_DIR)/train_model.py $(SCRIPTS_DIR)/export_edge.py $(SCRIPTS_DIR)/validate_parity.py $(SCRIPTS_DIR)/train_edge_model.py $(SCRIPTS_DIR)/export_edge_lite.py

.PHONY: all

all: $(SCRIPTS)
	@echo ======================================================================
	@echo CHURN EDGE AI - Pipeline completo
	@echo ======================================================================
	@echo [1/5] Ejecutando: train_model.py
	$(PYTHON) $(SCRIPTS_DIR)/train_model.py
	@echo [2/5] Ejecutando: export_edge.py
	$(PYTHON) $(SCRIPTS_DIR)/export_edge.py
	@echo [3/5] Ejecutando: validate_parity.py
	$(PYTHON) $(SCRIPTS_DIR)/validate_parity.py
	@echo [4/5] Ejecutando: train_edge_model.py
	$(PYTHON) $(SCRIPTS_DIR)/train_edge_model.py
	@echo [5/5] Ejecutando: export_edge_lite.py
	$(PYTHON) $(SCRIPTS_DIR)/export_edge_lite.py
	@echo ======================================================================
	@echo Pipeline completo ejecutado correctamente
	@echo ======================================================================