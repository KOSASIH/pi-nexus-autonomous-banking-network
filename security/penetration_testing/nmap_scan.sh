#!/bin/bash

nmap -sV -p 1-1000 -oX output.xml $1
