#!/bin/bash

# This script downloads hydra from a static link.
# And installs it at the user-specificed location 

if [ "$#" -ne 2 ]; then
    echo "Usage: ./install_hydra.sh src_dir builddir"
    echo "    src_dir: location where hydra source will be downloaded"
    echo "    builddir: installation directory"
    exit 1
fi

srcdir=$1
builddir=$2

if test -f $builddir/bin/mpiexec.hdra; then
    echo "hydra already installed"
    exit 1
fi

cd $srcdir
#Download hydra-3.2 source
wget http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
gunzip hydra-3.2.tar.gz
tar -xvf hydra-3.2.tar

#Install hydra
cd hydra-3.2
touch aclocal.m4; 
touch Makefile.am; 
touch Makefile.in; 
touch ./mpl/aclocal.m4; 
touch ./mpl/Makefile.am; 
touch ./mpl/Makefile.in;

./configure --prefix=$builddir --enable-cuda=no --enable-nvml=no
make
make install
rm -f -- $builddir/include/mpl*
mv $builddir/bin/mpiexec.hydra $builddir/bin/nvshmrun.hydra
# create a soft link with name nvshmrun
ln -s $builddir/bin/nvshmrun.hydra $builddir/bin/nvshmrun
rm -f $builddir/bin/mpiexec $builddir/bin/mpirun

echo "Hydra binaries have been installed in $builddir/bin"
