#!/bin/bash

valgrind --tool=memcheck --leak-check=full ./build/neural_network 
