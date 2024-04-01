# ZPEBOP

The zero-point energy from bond orders and populations (ZPEBOP) program is a computational chemistry algorithm that computes accurate zero-point vibrational energies with anharmonic effects at equilibrium and vibrational bond energies using well-conditioned Hartree-Fock or B3LYP orbital populations and bond orders from approximate quantum chemistry methods. Moreover, these methods do not require Hessians or higher-order derivatives. 

## Installation

```bash
git clone https://github.com/keithgroup/zpebop-qc
cd zpebop-qc
pip install .
```

## Preparing ZPEBOP Input Files

Currently, we have developed ZPEBOP-1 and ZPEBOP-2 models, and each of these models require output from the MinPop algorithm. Below is an example of how to do this in Gaussian 16 for each model.

1. Optimize your molecular structure using your preferred level of theory (e.g., B3LYP/CBSB7 and B3LYP/cc-pVTZ+1d).

##### ZPEBOP-1:

    # Opt B3LYP/CBSB7

##### ZPEBOP-2:

    # Opt B3LYP/cc-pVTZ+1d

2. Run Hartree-Fock or B3LYP on the optimized structure in Gaussian.

##### ZPEBOP-1:

    # SP B3LYP/CBSB3 Pop=(Full) IOp(6/27=122,6/12=3)
    
##### ZPEBOP-2:

    # SP ROHF/CBSB3 Pop=(Full) IOp(6/27=122,6/12=3)

## Usage

Execute `zpebop1` and `zpebop2` for ZPEBOP-1 and ZPEBOP-2, respectively. Run this in the command line:

##### ZPEBOP-1:

```bash
zpebop1 -f {name_file} --be --sort --json > {name_file}.bop
```

##### ZPEBOP-2

```bash
zpebop2 -f {name_file} -param_folder {param_folder} --be --sort --json > {name_file}.bop
```

where `{name_file}` is the Hartree-Fock or B3LYP MinPop output file and `{param_folder}` is the name or path of the ZPEBOP-2 parameter folder. Note that ZPEBOP-2 parameters are stored in json files under the `opt_parameters` folders.

Some examples of ZPEBOP-1 and ZPEBOP-2 output files are found in the `examples` directory.

Some details of the parsers used in `zpebop1` and `zpebop2` source codes.

##### ZPEBOP-1:

```bash
$ zpebop1 -h
usage: zpebop1 [-h] -f F [--be] [--sort] [--json]

compute ZPE and bond energies (i.e., gross and net)

optional arguments:
  -h, --help  show this help message and exit
  -f F        name of the Gaussian B3LYP output file
  --be        compute ZPEBOP vibrational bond energies (net and gross bond energies)
  --sort      sort the net ZPEBOP bond energies (from lowest to highest in energy)
  --json      save the job output into JSON
```

##### ZPEBOP-2:

```bash
$ zpebop2 -h
zpebop2 [-h] -f F [-param_folder PARAM_FOLDER] [--be] [--sort] [--json]

compute ZPE and bond energies (i.e., gross and net)

optional arguments:
  -h, --help            show this help message and exit
  -f F                  name of the Gaussian Hartree-Fock output file
  -param_folder PARAM_FOLDER
                        name of ZPEBOP-2's parameter path/folder (default: opt_parameters)
  --be                  compute ZPEBOP vibrational bond energies (net and gross bond energies)
  --sort                sort the net ZPEBOP bond energies (from lowest to highest in energy)
  --json                save the job output into JSON
```

## Citations

Please cite:

**ZPEBOP-1**: Jesse Albert Mangiardi. *Zero-Point Energies from Bond Orders*. Undergraduate thesis, Wesleyan University, Middletown, CT, 2015.

**ZPEBOP-2**:  Barbaro Zulueta, Colin D. Rude, Jesse A. Mangiardi, George A. Petersson, and John A. Keith. A Zero-Point Energies from Bond Orders and Populations Relationship. (in preparation), 2024.

## License

Distributed under the MIT License.
See `LICENSE` for more information.
