build: ## Build the module
	@cargo build
.PHONY: build

docs: ## Generate documents
	cargo doc
.PHONY: doc

help: ## Display this message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
.PHONY: help
