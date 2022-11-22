#! /bin/bash

jupyter lab --generate-config
echo -e "c.NotebookApp.ip = '0.0.0.0'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port=8888\nc.NotebookApp.token=''" > ~/.jupyter/jupyter_lab_config.py