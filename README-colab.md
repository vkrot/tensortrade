```python
!pip install git+https://github.com/vkrot/tensortrade.git
!pip install -r https://raw.githubusercontent.com/vkrot/tensortrade/master/requirements_ray.txt
```

install ta-lib
```python
!wget https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files/libta-lib0_0.4.0-oneiric1_amd64.deb -qO libta.deb
!wget https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files/ta-lib0-dev_0.4.0-oneiric1_amd64.deb -qO ta.deb
!dpkg -i libta.deb ta.deb
!pip install ta-lib
```