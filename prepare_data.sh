export PATH="~/anaconda4/bin:$PATH"
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz
gunzip reviews_Movies_and_TV_5.json.gz
gunzip meta_Movies_and_TV.json.gz
python script/process_data.py meta_Movies_and_TV.json reviews_Movies_and_TV_5.json
python script/local_aggretor.py
python script/generate_voc.py
