wget http://snap.stanford.edu/data/facebook.tar.gz
tar -xvzf facebook.tar.gz
python src/setup/fb_graph_setup.py
rm facebook.tar.gz
python src/setup/synthetic_graph_setup.py