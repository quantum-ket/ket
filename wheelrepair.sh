#!/bin/bash
for whl in dist/*.whl; do
    auditwheel repair $whl
done
