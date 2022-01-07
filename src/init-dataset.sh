#!/bin/bash
python3 scrape-conditions.py
python3 scrape-therapies.py
python3 integrate-data.py