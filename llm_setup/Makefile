
.PHONY: all
all: setup download

.PHONY: download-all
download-all: 
	@echo "Downloading all files..."
	@. venv/bin/activate && python download.py all

.PHONY: download-8b
download-8b: 
	@echo "Downloading 8b..."
	@. venv/bin/activate && python download.py 8b

.PHONY: download-70b
download-70b:
	@echo "Downloading 70b..."
	@. venv/bin/activate && python download.py 70b

.PHONY: download-405b
download-405b:
	@echo "Downloading 405b..."
	@. venv/bin/activate && python download.py 405b

.PHONY: quantize-8b
quantize:
	@echo "Quantizing 8b..."
	@. venv/bin/activate && python llama.cpp/convert_hf_to_gguf.py models/8b \
		--outfile models/8b_f16.gguf \
		--outtype f16 
	@llama.cpp/llama-quantize ./models/8b_f16.gguf ./models/8b-Q4_K_M.gguf Q4_K_M

.PHONY: quantize-70b
quantize-70b:
	@echo "Quantizing 70b..."
	@. venv/bin/activate && python llama.cpp/convert_hf_to_gguf.py models/70b \
		--outfile models/70b_f16.gguf \
		--outtype f16 
	@llama.cpp/llama-quantize ./models/70b_f16.gguf ./models/70b-Q4_K_M.gguf Q4_K_M

.PHONY: quantize-405b
quantize-405b:
	@echo "Quantizing 405b..."
	@. venv/bin/activate && python llama.cpp/convert_hf_to_gguf.py models/405b \
		--outfile models/405b_f16.gguf \
		--outtype f16 
	@llama.cpp/llama-quantize ./models/405b_f16.gguf ./models/405b-Q4_K_M.gguf Q4_K_M

.PHONY: setup
setup:
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r requirements.txt && pip install -r llama.cpp/requirements.txt
	@cd llama.cpp && make


.PHONY: clean
clean:
	@rm -rf venv
