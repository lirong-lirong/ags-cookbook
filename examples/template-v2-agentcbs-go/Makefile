.DEFAULT_GOAL := help

.PHONY: help setup run create-tool delete-volume delete-tool check

help:
	@echo "make setup         Download Go dependencies"
	@echo "make create-tool   Create a tool and request template preparation"
	@echo "make run           Mount a data volume in two sandboxes and verify persistence"
	@echo "make delete-volume Delete AGS_VOLUME_ID, including its data"
	@echo "make delete-tool   Delete AGS_TOOL_ID"
	@echo "make check         Run local tests, vet, and build"

setup:
	go mod download

run:
	go run . demo

create-tool:
	go run . create-tool

delete-volume:
	go run . delete-volume

delete-tool:
	go run . delete-tool

check:
	go test ./...
	go vet ./...
	@mkdir -p .local
	go build -buildvcs=false -o .local/cookbook .
