# Lancius v11A1 Compatibility Manifest

Version: v11A1\
Release Line: 1.0 Stable

## Purpose

This document defines the compatibility contract of Lancius v11A1.

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

v11A1 guarantees:

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

v11A1 represents the first development milestone of the 1.0 cycle.
