#include <stdio.h>

#include "logic_gates.cuh"

__device__ int logic_gate(int gate_type, int x, int y) {
    int result;

    switch (gate_type) {
        case AND:
            result = x & y;
            break;
        case OR:
            result = x | y;
            break;
        case NAND:
            result = !(x & y);
            break;
        case NOR:
            result = !(x + y);
            break;
        case XOR:
            result = x ^ y;
            break;
        case XNOR:
            result = x == y;
            break;
        default:
            printf("Error: Input gate '%d' invalid", gate_type);
            result = -1;
            break;
    }
    return result;
}