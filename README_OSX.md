Installation
============
Follow the steps in `dep_script.sh` with following modifications.

In most cases you don't need to be root (use `sudo`)

Use `conda` where possible. For example

    conda install opencv

Use brew to install SDL:

    brew update
    brew upgrade
    brew install sdl sdl_gfx sdl_image
    
Make sure you don't have an old SDL framework:

    sudo rm  -rf /Library/Frameworks/SDL.framework

If you want to display the simulator on screen (e.g. run `ale_run_watch.py`)
you should install `pygame`:

    hg clone https://bitbucket.org/pygame/pygame
    cd pygame
    pip install .
