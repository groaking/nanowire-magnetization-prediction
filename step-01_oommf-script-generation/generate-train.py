# This file generates the MIF files according to the range of sine wave variations

import os

# The template MIF file
# In this template, the substrings "$$A$$", "$$B$$", and "$$F$$" are to be replaced
# with the fluctuating amplitude, basename, and frequency values, respectively
template_str = """
# MIF 2.1
# Project "Sinusoidal Pulse" of OOMMF

# Define the constants
set pi [expr {4 * atan(1.0)}]
set mu0 [expr {4 * $pi * 1e-7}]

# Set the simulation parameters
RandomSeed 1
set cellsize 10e-9

# Set the boundary of the object
Specify Oxs_MultiAtlas:atlas {
    atlas { Oxs_BoxAtlas {
        name left
        xrange {0 950e-9}
        yrange {0 150e-9}
        zrange {0 10e-9}
    } }
    atlas { Oxs_BoxAtlas {
        name domain_wall
        xrange {950e-9 1050e-9}
        yrange {0 150e-9}
        zrange {0 10e-9}
    } }
    atlas { Oxs_BoxAtlas {
        name right
        xrange {1050e-9 2000e-9}
        yrange {0 150e-9}
        zrange {0 10e-9}
    } }
}

# Create the mesh
Specify Oxs_RectangularMesh:mesh [subst {
    cellsize {$cellsize $cellsize $cellsize}
    atlas :atlas
}]

# Determine the anisotropy parameters
Specify Oxs_CubicAnisotropy {
    K1 0
    axis1 {1 0 0}
    axis2 {0 1 0}
}

# Determine the uniform exchange parameters
Specify Oxs_UniformExchange {
    A 13e-12
}

# The following example produces a sinusoidally varying field of frequency 1 GHz and amplitude 800 A/m,
# directed along the x-axis. 
# SOURCE: https://math.nist.gov/oommf/doc/userguide12a3/userguide/Standard_Oxs_Ext_Child_Clas.html#SU

Specify Oxs_ScriptUZeeman {
   script_args total_time
   script SineField
}

proc SineField { total_time } {
    global pi mu0
    # Amplitude is in mT (militesla)
    set Amp $$A$$
    set Freq [expr {$$F$$e9*(2*$pi)}]
    set Hx [expr {$Amp*sin($Freq*$total_time)}]
    set dHx [expr {$Amp*$Freq*cos($Freq*$total_time)}]
    return [list $Hx 0 0 $dHx 0 0]
}

Specify Oxs_Demag {}
Specify Oxs_RungeKuttaEvolve:evolve {
    alpha 0.01
}
Specify Oxs_TimeDriver [subst {
    basename $$B$$
    evolver :evolve
    stopping_time 3e-9
    mesh :mesh
    Ms 800e3
    m0 { Oxs_AtlasVectorField {
        atlas :atlas
        values {
            left        { 1  0  0}
            domain_wall { 0  1  0}
            right       {-1  0  0}
        }
        norm 1.0
    } }
}]

# Default outputs
Destination archive   mmArchive

Schedule DataTable archive   Step 1

""".strip()

# Move the current working directory to the directory where the MIF files will be generated to
# Replace with the appropriate path according to one's liking
export_dir = '/ssynthesia/ghostcity/git/codename-cyborg/02312.00001_data-generation/generated-mifs'
try:
    os.mkdir(export_dir)
except FileExistsError:
    pass
os.chdir(export_dir)

# Determining the generation parameter
amplitude_range = range(1,51)  # --- mT, excluding 51
frequency_range = range(1,51)  # --- GHz, excluding 51 

# Iterating through every possible range combination
for amp in amplitude_range:
    for freq in frequency_range:
        # Convert int to str
        amp = str(amp)
        freq = str(freq)
        
        # The exported ODT filename
        filename = f'sinusoidal_a{amp}_f{freq}'
        
        # The generated parameters
        generated_str = template_str.replace('$$A$$', amp).replace('$$B$$', filename).replace('$$F$$', freq)
        
        # Logging
        print(f'+ Generating file {filename} with amplitude = {amp} and frequency = {freq}')
        
        # Write out the generated file
        with open((filename + '.mif'), 'w') as fo:
            fo.write(generated_str)
            fo.close()
