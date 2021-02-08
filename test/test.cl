/*------------------------------------------------------------------------------
 **
 ** Kernel:  generate_individual
 **
 ** Purpose: Generate random individuals
 **
 ** Input: 
 **
 ** Output: 
 **
 */




// Kernel
__kernel void generate_individual(__global double *F, __global double *D, __global double *res, const int count)
{
    double I = 5;

    // Create permutation matrix

    // Write result in global memory
    if (i < count)
        res[i] = I;
}