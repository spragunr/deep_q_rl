Installation
============
Look for steps at `dep_script.sh` with following modifications.

In most cases you don't need to be root (use `sudo`)

Use `conda` where possible. For example

    conda install opencv

Use brew SDL:

    brew update
    brew upgrade
    brew install sdl sdl_gfx sdl_image
    
Make sure you dont have an old SDL framework:

    sudo rm  -rf /Library/Frameworks/SDL.framework

ALE can be installed like this:

    git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
    cd Arcade-Learning-Environment/
    cmake  -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
    make -j2
    pip install .

If you want to display the simulator on screen (e.g. run `ale_run_watch.py`)
you should install `pygame`:

    hg clone https://bitbucket.org/pygame/pygame
    cd pygame
    pip install .
