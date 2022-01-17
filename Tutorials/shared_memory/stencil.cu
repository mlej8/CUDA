
/**
 * 1D Stencil
 *  Stencil codes are a class of iterative kernels which update array elements according to some fixed pattern, called a stencil.
 *  Consider applying a 1D stencil to a 1D array of elements.
 *  Each output element is the sum of input elements within a radius.
 * 
 *  If radius is 3, then each output element is the sum of 7 input elements:
 *  Input: [0,1,2,3,4,5,6]
 *          ^ ^ ^   ^ ^ ^
 *  Output: [21]
 * 
 * Implementation
 *  Read (ARRAY_SIZE + 2 * radius) input elements from global memory to shared memory 
 *  Compute ARRAY _SIZE output elements
 *  Write ARRAY _SIZE output elements to global memory
 */

__global__ void stencil() {
    // TODO
}


// TODO do one version with and without shared memory to compare
int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
