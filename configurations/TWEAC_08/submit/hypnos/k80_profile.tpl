#!/bin/bash
# Copyright 2013-2014 Axel Huebl, Anton Helm, Rene Widera
# 
# This file is part of PIConGPU. 
# 
# PIConGPU is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 
# 
# PIConGPU is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details. 
# 
# You should have received a copy of the GNU General Public License 
# along with PIConGPU.  
# If not, see <http://www.gnu.org/licenses/>. 
# 


## calculation are done by tbg ##
TBG_queue="k80"
TBG_mailAddress=${MY_MAIL:-"a.debus@hzdr.de"}
TBG_mailSettings=${MY_MAILNOTIFY:-"abe"}
#TBG_jobName="TWEAC01"

# 8 gpus per nodes
TBG_gpusPerNode=8

#number of cores per parallel node / default is 2 cores per gpu on k80 queue
TBG_coresPerNode="$(( TBG_gpusPerNode * 2 ))"

# use ceil to caculate nodes
TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

# PIConGPU batch script for hypnos PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode
# send me a mail on (b)egin, (e)nd, (a)bortion
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath

#PBS -o stdout
#PBS -e stderr

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source ~/picongpu.k80.profile
if [ $? -ne 0 ] ; then
  echo "Error: ~/picongpu.k80.profile not found!"
  exit 1
fi
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

#wait that all nodes see ouput folder
sleep 1

mpiexec --prefix $MPIHOME -tag-output --display-map -x LIBRARY_PATH -x LD_LIBRARY_PATH -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/picongpu/bin/cuda_memtest.sh

if [ $? -eq 0 ] ; then
  mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -tag-output --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks !TBG_dstPath/picongpu/bin/picongpu !TBG_programParams
fi

mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -npernode !TBG_gpusPerNode -n !TBG_tasks killall -9 picongpu