# Lancius v10S Compatibility Manifest

Version: v10S\
Release Line: 1.0 Stable

## Purpose

This document defines the compatibility contract of Lancius v10S.

## Stable Components

### Runtime

-   Graph execution
-   Tensor management
-   Memory planning
-   Scheduler integration

### API

The stable public interface is provided through the Lancius stable API
headers.

Internal headers may change.

### Model Execution

Supported:

-   Static computation graphs
-   Supported operator set
-   CPU execution path

## Compatibility Guarantees

v10S guarantees:

-   reproducible builds
-   stable public API behavior
-   documented limitations
-   regression-tested core execution

## Non-Guarantees

The following are not guaranteed:

-   binary compatibility with experimental components
-   unsupported operators
-   unfinished backends

## Versioning

v10S represents the first stable release of the 1.0 cycle.
