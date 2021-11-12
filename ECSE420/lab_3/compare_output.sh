#! /usr/bin/bash

./build/Helper/compare_node_output Output/nodeOutput.raw Solution/sol_nodeOutput.raw
./build/Helper/compare_next_level_nodes Output/nextLevelNodes.raw Solution/sol_nextLevelNodes.raw