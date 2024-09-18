#!/bin/bash

# This script is used to detect if moshi or moshi_mlx are installed, and run
# their CI only in that case!

package=$1
if python -c "from $package import models"; then
    # package is installed, let's run the command
    eval $2
else
    echo "Package $package not installed, skipping the CI for it."
fi
