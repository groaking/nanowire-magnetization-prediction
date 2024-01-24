#!/bin/sh
# Solve the OOMMF simulation problems from each generated MIF file

# The oommf program
alias oommf20='tclsh /home/longbowman/.oommf20/oommf.tcl'

# The directory where all the MIF files are stored
mif_dir="/ssynthesia/ghostcity/git/codename-cyborg/02401.00009_data-generation-test-set/generated-mifs"

# The directory where the output ODT files will be exported to
out_dir="/ssynthesia/ghostcity/git/codename-cyborg/02401.00010_simulation-run-test-set/odt-archives"

# Iterate through every MIF file and simulate every one of them
ls $mif_dir | sort -V | while read -r line; do oommf20 Boxsi -cwd $mif_dir -outdir $out_dir -- $line; done
