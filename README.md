checkouts:
- bokeh commit: 6602ea6bcef184303a18493c1311c040ccac0bd9
- kitchensink commit: 79b8316d4b43e4c45c2ca5f2159538382a146186
- ar commit: 1aed89fa19ae0efc8d33698112986680deb540e8

run setup.py develop on bokeh/kitchensink
build bokeh js
add ar directory to a pth file in site-packages
build ar with python setup.py build_ext --inplace


#start redis
redis-server --port 9001

#start kitchensink
python -m kitchensink.scripts.start --node-name=power --node-url=http://localhost:6323/ --num-workers=8 --module ksimports --no-redis --redis-conn=tcp://localhost:9001?db=9 --datadir /data/Raid5/home/hugo/data

#start bokeh - note this assumes that bokeh/ar are peers (you can access ar/app.py with ../ar/app.py)
./bokeh-server -djs --script ../ar/app.py --backend redis --redis-port=9001 --no-start-redis --ip=0.0.0.0 --bokeh-port=9002 --ws-port=903 --zmqaddr ipc:///tmp/hugo

# run bootstrap

python -m bootstrap.py (in ar directory)


### CONTENTS of python env which runs this project:
```
# platform: linux-64
_license=1.1=py27_0
abstract-rendering=0.5.1=np19py27_0
accelerate=1.7.0=np19py27_p0
anaconda=2.1.0=np19py27_0
argcomplete=0.8.1=py27_0
astropy=0.4.2=np19py27_0
atom=0.3.9=py27_0
beautiful-soup=4.3.2=py27_0
binstar=0.7.1=py27_0
bitarray=0.8.1=py27_0
blaze=0.6.3=np19py27_0
blz=0.6.2=np19py27_0
bokeh=0.6.1=np19py27_0
boto=2.32.1=py27_0
cairo=1.12.2=2
casuarius=1.1=py27_0
cdecimal=2.3=py27_0
cffi=0.8.6=py27_0
chaco=4.4.1=np19py27_0
colorama=0.3.1=py27_0
configobj=5.0.6=py27_0
cryptography=0.5.4=py27_0
cudatoolkit=6.0=p0
curl=7.38.0=0
cython=0.21=py27_0
cytoolz=0.7.0=py27_0
datashape=0.3.0=np19py27_1
dateutil=2.1=py27_2
decorator=3.4.0=py27_0
dill=0.2b1=py27_0
docutils=0.12=py27_0
dynd-python=0.6.5=np19py27_0
enable=4.3.0=np19py27_2
enaml=0.9.8=py27_0
flask=0.10.1=py27_1
freetype=2.4.10=0
future=0.13.1=py27_0
futures=2.1.6=py27_0
gevent=1.0.1=py27_0
gevent-websocket=0.9.3=py27_0
greenlet=0.4.4=py27_0
grin=1.2.1=py27_1
h5py=2.3.1=np19py27_0
hdf5=1.8.13=0
ipython=2.2.0=py27_0
ipython-notebook=2.2.0=py27_0
ipython-qtconsole=2.2.0=py27_0
itsdangerous=0.24=py27_0
jdcal=1.0=py27_0
jinja2=2.7.3=py27_1
jpeg=8d=0
kitchensink=0.2=np18py27_0
kiwisolver=0.1.3=py27_0
lcms=1.19=0
libdynd=0.6.5=0
libffi=3.0.13=0
libpng=1.5.13=1
libsodium=0.4.5=0
libtiff=4.0.2=1
libxml2=2.9.0=0
libxslt=1.1.28=0
llvm=3.3=0
llvmpy=0.12.7=py27_0
lxml=3.4.0=py27_0
markupsafe=0.23=py27_0
matplotlib=1.4.0=np19py27_0
mkl=11.1=np18py27_p3
mkl-rt=11.1=p0
mkl-service=1.0.0=py27_p1
mklfft=1.0=np19py27_p0
mock=1.0.1=py27_0
mpi4py=1.3=py27_0
mpich2=1.4.1p1=0
multipledispatch=0.4.7=py27_0
networkx=1.9.1=py27_0
nltk=3.0.0=np19py27_0
nodejs=0.10.18=0
nose=1.3.4=py27_0
numba=0.14.0=np19py27_0
numbapro=0.15.0=np19py27_p0
numbapro_cudalib=0.1=0
numexpr=2.3.1=np18py27_p0
numpy=1.9.1=py27_p0
openpyxl=1.8.5=py27_0
openssl=1.0.1h=1
pandas=0.14.1=np18py27_0
patsy=0.3.0=np19py27_0
pep8=1.5.7=py27_0
pil=1.1.7=py27_1
pip=1.5.6=py27_0
pixman=0.26.2=0
ply=3.4=py27_0
psutil=2.1.1=py27_0
py=1.4.25=py27_0
py2cairo=1.10.0=py27_1
pycosat=0.6.1=py27_0
pycparser=2.10=py27_0
pycrypto=2.6.1=py27_0
pycurl=7.19.5=py27_1
pyface=4.4.0=py27_0
pyflakes=0.8.1=py27_0
pygments=1.6=py27_0
pyopenssl=0.14=py27_0
pyparsing=2.0.1=py27_0
pyqt=4.10.4=py27_0
pytables=3.1.1=np19py27_1
pytest=2.6.3=py27_0
python=2.7.8=1
pytz=2014.7=py27_0
pyyaml=3.11=py27_0
pyzmq=14.3.1=py27_0
qt=4.8.5=0
readline=6.2=2
redis=2.6.9=0
redis-py=2.10.3=py27_0
requests=2.4.3=py27_0
rope=0.9.4=py27_1
rq=0.4.6=py27_0
runipy=0.1.1=py27_0
scikit-image=0.10.1=np19py27_0
scikit-learn=0.15.0b1=np18py27_p0
scipy=0.14.0=np18py27_p0
setuptools=5.8=py27_0
simpleservices=0.2=py27_0
sip=4.15.5=py27_0
six=1.8.0=py27_0
sockjs-tornado=1.0.1=py27_0
sphinx=1.2.3=py27_0
spyder=2.3.1=py27_0
spyder-app=2.3.1=py27_0
sqlalchemy=0.9.7=py27_0
sqlite=3.8.4.1=0
ssl_match_hostname=3.4.0.2=py27_0
statsmodels=0.5.0=np19py27_2
sympy=0.7.5=py27_0
system=5.8=1
theano=0.6.0=np19py27_0
tk=8.5.15=0
toolz=0.7.0=py27_0
tornado=4.0.2=py27_0
traits=4.4.0=py27_0
traitsui=4.4.0=py27_0
ujson=1.33=py27_0
unicodecsv=0.9.4=py27_0
util-linux=2.21=0
werkzeug=0.9.6=py27_1
xlrd=0.9.3=py27_0
xlsxwriter=0.5.7=py27_0
xlwt=0.7.5=py27_0
yaml=0.1.4=0
zeromq=4.0.4=0
zlib=1.2.7=0
```
