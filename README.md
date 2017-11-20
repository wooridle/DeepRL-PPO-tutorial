# DeepRL-PPO-tutorial
This repository contains tutorial material on Doing DeepRL with PPO in GDG DevFest 2017 Seoul.

## 발표자료

[Doing_RL_with_PPO.pdf](./Doing_RL_with_PPO.pdf) 입니다.

## Roboschool 설치 가이드

설치에 앞서 roboschool은 Mac과 Linux 운영체제만 지원합니다.

가장 먼저 roboshcool을 깃헙에서 다운받습니다.

```
git clone https://github.com/openai/roboschool
```



그리고 먼저 ROBOSCHOOL_PATH를 설정해야 합니다. 이 설정은 설치때만 이용하기 때문에 현재 shell에서만 적용되도록 경로를 설정해 줍니다. /path/to/roboschool 대신 자신이 다운받은 roboschool 경로를 설정합니다.

```
ROBOSCHOOL_PATH=/path/to/roboschool
```



이제 roboschool 설치에 필요한 패키지들을 설치합니다. 각각의 운영체제에 맞는 설치를 이용하면 됩니다.

- Linux

```
sudo apt install cmake ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libpython3.5-dev libboost-python-dev libtinyxml-dev
```

- Mac

```
# Will not work on Mavericks: unsupported by homebrew, some libraries won't compile, upgrade first
brew install python3
brew install cmake tinyxml assimp ffmpeg qt
brew install boost-python --without-python --with-python3 --build-from-source
export PATH=/usr/local/bin:/usr/local/opt/qt5/bin:$PATH
export PKG_CONFIG_PATH=/usr/local/opt/qt5/lib/pkgconfig
```

- Mac, Anaconda with Python 3

```
brew install cmake tinyxml assimp ffmpeg
brew install boost-python --without-python --with-python3 --build-from-source
conda install qt
export PKG_CONFIG_PATH=$(dirname $(dirname $(which python)))/lib/pkgconfig
```



그 다음은 roboschool을 돌리는데 필요한 물리엔진인 bullet3를 설치해야 합니다. 먼저 bullet3를 먼저 깃헙에서 받아와서 빌드를 해야합니다.
** git clone한 roboschool 디렉토리에서 bullet3를 설치해주세요.

```
git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
mkdir bullet3/build
cd bullet3/build
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 _DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
make -j4
make install
cd ../..
```



마지막으로 gym과 roboschool을 설치하면 됩니다. (python2 버전을 이용한다면 pip를 이용하면 됩니다.)

```
pip3 install gym
pip3 install -e $ROBOSCHOOL_PATH
```
