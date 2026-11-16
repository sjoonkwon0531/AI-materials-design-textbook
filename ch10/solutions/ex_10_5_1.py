"""Exercise 10.5.1: size of MPtrj versus the Lab 10.7 student's data."""
nconf, nforce = 1_580_395, 49_295_660
print(f"atoms (force vectors) per configuration: {nforce/nconf:.2f}")
ncomp = 3 * nforce
stud = 34 * 64 * 3
print(f"force components in MPtrj: {ncomp:,}; student (34 cells x 192): {stud:,}; ratio {ncomp/stud:,.0f}")
