# rtsr
recurrently trained super-resolution

you can do with out setting a data and just doing test with sample : (1/20 of DIV2K)

if you want to run with full data :

  1. SET HR
    
    make 100 image crops per a DIV2K train image
    -> positioning them ./dataset/DIV2K_train_HR/images/train_crop_sr
  
  2. SET LR 
  
    compress them
    -> positioning them ./dataset/DIV2K_train_HR/images/compressed_train_crop_png 

  3. SET TEST Image (Valid SET)
    
    set image you want to SR
    -> positioning them ./dataset/DIV2K_train_HR/images/valid_sr

runing : 
  sh ./run_all.sh (rtsr / srcnn / srresnet / vdsr / edsr)
  
  ex) sh ./run_all.sh rtsr

result : 
   ./output



sh batched for 12G GPU-Ram



<details>
 <summary> Running Environment (conda list)</summary>

|	Name	|	Version	|	Build	|	Channel	|
|	:-------------------------	|	:-------	|	:---------------------------	|	:------------------	|
|	_anaconda_depends	|	5.2.0	|	py36_3	|	anaconda	|
|	_libgcc_mutex	|	0.1	|	main	|		|
|	_tflow_190_select	|	0.0.1	|	gpu	|	anaconda	|
|	_tflow_select	|	2.3.0	|	mkl	|	anaconda	|
|	absl-py	|	0.8.0	|	py36_0	|	anaconda	|
|	alabaster	|	0.7.10	|	py36h306e16b_0	|		|
|	anaconda	|	custom	|	py36_1	|	anaconda	|
|	anaconda-client	|	1.6.14	|	py36_0	|		|
|	anaconda-project	|	0.8.2	|	py36h44fb852_0	|		|
|	asn1crypto	|	0.24.0	|	py36_0	|		|
|	astor	|	0.8.0	|	py36_0	|	anaconda	|
|	astroid	|	1.6.3	|	py36_0	|		|
|	astropy	|	3.0.2	|	py36h3010b51_1	|		|
|	attrs	|	18.1.0	|	py36_0	|		|
|	babel	|	2.5.3	|	py36_0	|		|
|	backcall	|	0.1.0	|	py36_0	|		|
|	backports	|	1	|	py36hfa02d7e_1	|		|
|	backports.shutil_get_terminal_size	|	1.0.0	|	py36hfea85ff_2	|		|
|	beautifulsoup4	|	4.6.0	|	py36h49b8c8c_1	|		|
|	bitarray	|	0.8.1	|	py36h14c3975_1	|		|
|	bkcharts	|	0.2	|	py36h735825a_0	|		|
|	blas	|	1	|	mkl	|		|
|	blaze	|	0.11.3	|	py36_0	|	anaconda	|
|	bleach	|	3.1.0	|	py36_0	|	anaconda	|
|	blosc	|	1.14.3	|	hdbcaa40_0	|		|
|	bokeh	|	0.12.16	|	py36_0	|		|
|	boto	|	2.48.0	|	py36h6e4cd66_1	|		|
|	bottleneck	|	1.2.1	|	py36haac1ea0_0	|		|
|	bzip2	|	1.0.6	|	h14c3975_5	|		|
|	ca-certificates	|	2019.9.11	|	hecc5488_0	|	conda-forge	|
|	cairo	|	1.14.12	|	h7636065_2	|		|
|	certifi	|	2019.9.11	|	py36_0	|	conda-forge	|
|	cffi	|	1.11.5	|	py36h9745a5d_0	|		|
|	chardet	|	3.0.4	|	py36h0f667ec_1	|		|
|	click	|	6.7	|	py36h5253387_0	|		|
|	cloudpickle	|	0.5.3	|	py36_0	|		|
|	clyent	|	1.2.2	|	py36h7e57e65_1	|		|
|	colorama	|	0.3.9	|	py36h489cec4_0	|		|
|	conda	|	4.6.14	|	py36_0	|	anaconda	|
|	conda-build	|	3.17.8	|	py36_1	|	conda-forge	|
|	contextlib2	|	0.5.5	|	py36h6c84a62_0	|		|
|	cryptography	|	2.2.2	|	py36h14c3975_0	|		|
|	cudatoolkit	|	10.0.130	|	0	|		|
|	curl	|	7.60.0	|	h84994c4_0	|		|
|	cycler	|	0.10.0	|	py36h93f1223_0	|		|
|	cython	|	0.28.5	|	py36hf484d3e_0	|	anaconda	|
|	cytoolz	|	0.9.0.1	|	py36h14c3975_0	|		|
|	dask	|	0.17.5	|	py36_0	|		|
|	dask-core	|	0.17.5	|	py36_0	|		|
|	datashape	|	0.5.4	|	py36h3ad6b5c_0	|		|
|	dbus	|	1.13.2	|	h714fa37_1	|		|
|	decorator	|	4.3.0	|	py36_0	|		|
|	distributed	|	1.21.8	|	py36_0	|		|
|	docutils	|	0.14	|	py36hb0f60f5_0	|		|
|	entrypoints	|	0.2.3	|	py36h1aec115_2	|		|
|	et_xmlfile	|	1.0.1	|	py36hd6bccc3_0	|		|
|	expat	|	2.2.5	|	he0dffb1_0	|		|
|	fastcache	|	1.0.2	|	py36h14c3975_2	|		|
|	filelock	|	3.0.4	|	py36_0	|		|
|	flask	|	1.0.2	|	py36_1	|		|
|	flask-cors	|	3.0.4	|	py36_0	|		|
|	fontconfig	|	2.12.6	|	h49f89f6_0	|		|
|	freetype	|	2.8	|	hab7d2ae_1	|		|
|	gast	|	0.3.2	|	py_0	|	anaconda	|
|	get_terminal_size	|	1.0.0	|	haa9412d_0	|		|
|	gevent	|	1.3.0	|	py36h14c3975_0	|		|
|	glib	|	2.56.1	|	h000015b_0	|		|
|	glob2	|	0.6	|	py36he249c77_0	|		|
|	gmp	|	6.1.2	|	h6c8ec71_1	|		|
|	gmpy2	|	2.0.8	|	py36hc8893dd_2	|		|
|	graphite2	|	1.3.11	|	h16798f4_2	|		|
|	greenlet	|	0.4.13	|	py36h14c3975_0	|		|
|	grpcio	|	1.12.1	|	py36hdbcaa40_0	|	anaconda	|
|	gst-plugins-base	|	1.14.0	|	hbbd80ab_1	|		|
|	gstreamer	|	1.14.0	|	hb453b48_1	|		|
|	h5py	|	2.8.0	|	py36h7eb728f_3	|	conda-forge	|
|	harfbuzz	|	1.7.6	|	h5f0a787_1	|		|
|	hdf5	|	1.10.2	|	hba1933b_1	|		|
|	heapdict	|	1.0.0	|	py36_2	|		|
|	html5lib	|	1.0.1	|	py36_0	|	anaconda	|
|	icu	|	58.2	|	h9c2bf20_1	|		|
|	idna	|	2.6	|	py36h82fb2a8_1	|		|
|	imageio	|	2.3.0	|	py36_0	|		|
|	imagesize	|	1.0.0	|	py36_0	|		|
|	intel-openmp	|	2018.0.0	|	8	|		|
|	ipykernel	|	4.8.2	|	py36_0	|		|
|	ipython	|	6.4.0	|	py36_0	|		|
|	ipython_genutils	|	0.2.0	|	py36hb52b0d5_0	|		|
|	ipywidgets	|	7.2.1	|	py36_0	|		|
|	isort	|	4.3.4	|	py36_0	|		|
|	itsdangerous	|	0.24	|	py36h93cc618_1	|		|
|	jbig	|	2.1	|	hdba287a_0	|		|
|	jdcal	|	1.4	|	py36_0	|		|
|	jedi	|	0.12.0	|	py36_1	|		|
|	jinja2	|	2.1	|	py36ha16c418_0	|		|
|	jpeg	|	9b	|	h024ee3a_2	|		|
|	jsonschema	|	2.6.0	|	py36h006f8b5_0	|		|
|	jupyter	|	1.0.0	|	py36_4	|		|
|	jupyter_client	|	5.2.3	|	py36_0	|		|
|	jupyter_console	|	5.2.0	|	py36he59e554_1	|		|
|	jupyter_core	|	4.4.0	|	py36h7c827e3_0	|		|
|	jupyterlab	|	0.32.1	|	py36_0	|		|
|	jupyterlab_launcher	|	0.10.5	|	py36_0	|		|
|	kiwisolver	|	1.0.1	|	py36h764f252_0	|		|
|	lazy-object-proxy	|	1.3.1	|	py36h10fcdad_0	|		|
|	libarchive	|	3.3.2	|	hb43526a_6	|	anaconda	|
|	libcurl	|	7.60.0	|	h1ad7b7a_0	|		|
|	libedit	|	3.1.20170329	|	h6b74fdf_2	|		|
|	libffi	|	3.2.1	|	hd88cf55_4	|		|
|	libgcc-ng	|	7.2.0	|	hdf63c60_3	|		|
|	libgfortran-ng	|	7.2.0	|	hdf63c60_3	|		|
|	liblief	|	0.9.0	|	h1532aa0_0	|	anaconda	|
|	libpng	|	1.6.34	|	hb9fc6fc_0	|		|
|	libprotobuf	|	3.6.0	|	hdbcaa40_0	|	anaconda	|
|	libsodium	|	1.0.16	|	h1bed415_0	|		|
|	libssh2	|	1.8.0	|	h9cfc8f7_4	|		|
|	libstdcxx-ng	|	7.2.0	|	hdf63c60_3	|		|
|	libtiff	|	4.0.9	|	he85c1e1_1	|		|
|	libtool	|	2.4.6	|	h544aabb_3	|		|
|	libxcb	|	1.13	|	h1bed415_1	|		|
|	libxml2	|	2.9.8	|	h26e45fe_1	|		|
|	libxslt	|	1.1.32	|	h1312cb7_0	|		|
|	llvmlite	|	0.23.1	|	py36hdbcaa40_0	|		|
|	locket	|	0.2.0	|	py36h787c0ad_1	|		|
|	lxml	|	4.2.1	|	py36h23eabaa_0	|		|
|	lz4-c	|	1.8.1.2	|	h14c3975_0	|	anaconda	|
|	lzo	|	2.1	|	h49e0be7_2	|		|
|	markdown	|	3.1.1	|	py36_0	|	anaconda	|
|	markupsafe	|	1	|	py36hd9260cd_1	|		|
|	matplotlib	|	2.2.2	|	py36h0e671d2_1	|		|
|	mccabe	|	0.6.1	|	py36h5ad9710_1	|		|
|	mistune	|	0.8.3	|	py36h14c3975_1	|		|
|	mkl	|	2018.0.2	|	1	|		|
|	mkl-service	|	1.1.2	|	py36h651fb7a_4	|	anaconda	|
|	mkl_fft	|	1.0.1	|	py36h3010b51_0	|		|
|	mkl_random	|	1.0.1	|	py36h629b387_0	|		|
|	more-itertools	|	4.1.0	|	py36_0	|		|
|	mpc	|	1.0.3	|	hec55b23_5	|		|
|	mpfr	|	3.1.5	|	h11a74b3_2	|		|
|	mpmath	|	1.0.0	|	py36hfeacd6b_2	|		|
|	msgpack-python	|	0.5.6	|	py36h6bb024c_0	|		|
|	multipledispatch	|	0.5.0	|	py36_0	|		|
|	nbconvert	|	5.3.1	|	py36hb41ffb7_0	|		|
|	nbformat	|	4.4.0	|	py36h31c9010_0	|		|
|	ncurses	|	6.1	|	hf484d3e_0	|		|
|	networkx	|	2.1	|	py36_0	|		|
|	ninja	|	1.8.2	|	py36h6bb024c_1	|		|
|	nltk	|	3.3.0	|	py36_0	|		|
|	nose	|	1.3.7	|	py36hcdf7029_2	|		|
|	notebook	|	5.5.0	|	py36_0	|		|
|	numba	|	0.38.0	|	py36h637b7d7_0	|		|
|	numexpr	|	2.6.5	|	py36h7bf3b9c_0	|	anaconda	|
|	numpy	|	1.14.3	|	py36hcd700cb_1	|		|
|	numpy-base	|	1.14.3	|	py36h9be14a7_1	|		|
|	numpydoc	|	0.8.0	|	py36_0	|		|
|	odo	|	0.5.1	|	py36h90ed295_0	|		|
|	olefile	|	0.45.1	|	py36_0	|		|
|	openpyxl	|	2.5.3	|	py36_0	|		|
|	openssl	|	1.0.2p	|	h470a237_2	|	conda-forge	|
|	packaging	|	17.1	|	py36_0	|		|
|	pandas	|	0.23.0	|	py36h637b7d7_0	|		|
|	pandoc	|	1.19.2.1	|	hea2e7c5_1	|		|
|	pandocfilters	|	1.4.2	|	py36ha6701b7_1	|		|
|	pango	|	1.41.0	|	hd475d92_0	|		|
|	parso	|	0.2.0	|	py36_0	|		|
|	partd	|	0.3.8	|	py36h36fd896_0	|		|
|	patchelf	|	0.9	|	hf79760b_2	|		|
|	path.py	|	11.0.1	|	py36_0	|		|
|	pathlib2	|	2.3.2	|	py36_0	|		|
|	patsy	|	0.5.0	|	py36_0	|		|
|	pcre	|	8.42	|	h439df22_0	|		|
|	pep8	|	1.7.1	|	py36_0	|		|
|	pexpect	|	4.5.0	|	py36_0	|		|
|	pickleshare	|	0.7.4	|	py36h63277f8_0	|		|
|	pillow	|	5.1.0	|	py36h3deb7b8_0	|		|
|	pip	|	10.0.1	|	py36_0	|		|
|	pixman	|	0.34.0	|	hceecf20_3	|		|
|	pkginfo	|	1.4.2	|	py36_1	|		|
|	pluggy	|	0.6.0	|	py36hb689045_0	|		|
|	ply	|	3.11	|	py36_0	|		|
|	prompt_toolkit	|	1.0.15	|	py36h17d85b1_0	|		|
|	protobuf	|	3.6.0	|	py36hf484d3e_0	|	anaconda	|
|	psutil	|	5.4.5	|	py36h14c3975_0	|		|
|	ptyprocess	|	0.5.2	|	py36h69acd42_0	|		|
|	py	|	1.5.3	|	py36_0	|		|
|	py-lief	|	0.9.0	|	py36h1532aa0_0	|	anaconda	|
|	pycodestyle	|	2.4.0	|	py36_0	|		|
|	pycosat	|	0.6.3	|	py36h0a5515d_0	|		|
|	pycparser	|	2.18	|	py36hf9f622e_1	|		|
|	pycrypto	|	2.6.1	|	py36h14c3975_8	|		|
|	pycurl	|	7.43.0.1	|	py36hb7f436b_0	|		|
|	pyflakes	|	1.6.0	|	py36h7bd6a15_0	|		|
|	pygments	|	2.2.0	|	py36h0d3125c_0	|		|
|	pylint	|	1.8.4	|	py36_0	|		|
|	pyodbc	|	4.0.23	|	py36hf484d3e_0	|		|
|	pyopenssl	|	18.0.0	|	py36_0	|		|
|	pyparsing	|	2.2.0	|	py36hee85983_1	|		|
|	pyqt	|	5.9.2	|	py36h751905a_0	|		|
|	pysocks	|	1.6.8	|	py36_0	|		|
|	pytables	|	3.4.4	|	py36ha205bf6_0	|	anaconda	|
|	pytest	|	3.5.1	|	py36_0	|		|
|	pytest-arraydiff	|	0.2	|	py36_0	|		|
|	pytest-astropy	|	0.3.0	|	py36_0	|		|
|	pytest-doctestplus	|	0.1.3	|	py36_0	|		|
|	pytest-openfiles	|	0.3.0	|	py36_0	|		|
|	pytest-remotedata	|	0.2.1	|	py36_0	|		|
|	python	|	3.6.5	|	hc3d631a_2	|		|
|	python-dateutil	|	2.7.3	|	py36_0	|		|
|	python-libarchive-c	|	2.8	|	py36_13	|	anaconda	|
|	pytorch	|	1.2.0	|	py3.6_cuda10.0.130_cudnn7.6.2_0	|	pytorch	|
|	pytz	|	2018.4	|	py36_0	|		|
|	pywavelets	|	0.5.2	|	py36he602eb0_0	|		|
|	pyyaml	|	3.12	|	py36hafb9ca4_1	|		|
|	pyzmq	|	17.0.0	|	py36h14c3975_0	|		|
|	qt	|	5.9.5	|	h7e424d6_0	|		|
|	qtawesome	|	0.4.4	|	py36h609ed8c_0	|		|
|	qtconsole	|	4.3.1	|	py36h8f73b5b_0	|		|
|	qtpy	|	1.4.1	|	py36_0	|		|
|	readline	|	7	|	ha6073c6_4	|		|
|	requests	|	2.18.4	|	py36he2e5f8d_1	|		|
|	rope	|	0.10.7	|	py36h147e2ec_0	|		|
|	ruamel_yaml	|	0.15.35	|	py36h14c3975_1	|		|
|	scikit-image	|	0.13.1	|	py36h14c3975_1	|		|
|	scikit-learn	|	0.19.1	|	py36h7aa7ec6_0	|	anaconda	|
|	scipy	|	1.1.0	|	py36hfc37229_0	|		|
|	seaborn	|	0.8.1	|	py36hfad7ec4_0	|		|
|	send2trash	|	1.5.0	|	py36_0	|		|
|	setuptools	|	39.1.0	|	py36_0	|		|
|	simplegeneric	|	0.8.1	|	py36_2	|		|
|	singledispatch	|	3.4.0.3	|	py36h7a266c3_0	|		|
|	sip	|	4.19.8	|	py36hf484d3e_0	|		|
|	six	|	1.11.0	|	py36h372c433_1	|		|
|	snappy	|	1.1.7	|	hbae5bb6_3	|		|
|	snowballstemmer	|	1.2.1	|	py36h6febd40_0	|		|
|	sortedcollections	|	0.6.1	|	py36_0	|		|
|	sortedcontainers	|	1.5.10	|	py36_0	|		|
|	sphinx	|	1.7.4	|	py36_0	|		|
|	sphinxcontrib	|	1	|	py36h6d0f590_1	|		|
|	sphinxcontrib-websupport	|	1.0.1	|	py36hb5cb234_1	|		|
|	spyder	|	3.2.8	|	py36_0	|		|
|	sqlalchemy	|	1.2.7	|	py36h6b74fdf_0	|		|
|	sqlite	|	3.23.1	|	he433501_0	|		|
|	statsmodels	|	0.9.0	|	py36h3010b51_0	|		|
|	sympy	|	1.1.1	|	py36hc6d1c1c_0	|		|
|	tblib	|	1.3.2	|	py36h34cf8b6_0	|		|
|	tensorboard	|	1.9.0	|	py36hf484d3e_0	|	anaconda	|
|	tensorflow	|	1.9.0	|	mkl_py36h6d6ce78_1	|		|
|	tensorflow-base	|	1.9.0	|	mkl_py36h2ca6a6a_0	|	anaconda	|
|	tensorflow-gpu	|	1.9.0	|	hf154084_0	|	anaconda	|
|	termcolor	|	1.1.0	|	py36_1	|	anaconda	|
|	terminado	|	0.8.1	|	py36_1	|		|
|	testpath	|	0.3.1	|	py36h8cadb63_0	|		|
|	tk	|	8.6.7	|	hc745277_3	|		|
|	toolz	|	0.9.0	|	py36_0	|		|
|	torchvision	|	0.4.0	|	py36_cu100	|	pytorch	|
|	tornado	|	5.0.2	|	py36_0	|		|
|	tqdm	|	4.36.1	|	py_0	|	anaconda	|
|	traitlets	|	4.3.2	|	py36h674d592_0	|		|
|	typing	|	3.6.4	|	py36_0	|		|
|	unicodecsv	|	0.14.1	|	py36ha668878_0	|		|
|	unixodbc	|	2.3.6	|	h1bed415_0	|		|
|	urllib3	|	1.22	|	py36hbe7ace6_0	|		|
|	wcwidth	|	0.1.7	|	py36hdf4376a_0	|		|
|	webencodings	|	0.5.1	|	py36h800622e_1	|		|
|	werkzeug	|	0.14.1	|	py36_0	|		|
|	wheel	|	0.31.1	|	py36_0	|		|
|	widgetsnbextension	|	3.2.1	|	py36_0	|		|
|	wrapt	|	1.11.2	|	py36h7b6447c_0	|	anaconda	|
|	xlrd	|	1.1.0	|	py36h1db9f0c_1	|		|
|	xlsxwriter	|	1.0.4	|	py36_0	|		|
|	xlwt	|	1.3.0	|	py36h7b00a1f_0	|		|
|	xz	|	5.2.4	|	h14c3975_4	|		|
|	yaml	|	0.1.7	|	had09818_2	|		|
|	zeromq	|	4.2.5	|	h439df22_0	|		|
|	zict	|	0.1.3	|	py36h3a3bf81_0	|		|
|	zlib	|	1.2.11	|	ha838bed_2	|		|
</details>
