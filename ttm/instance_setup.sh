#!/usr/bin/env bash
# ami-f0bde196
sudo apt-get install unzip
sudo apt-get -y install virtualenv

export LC_ALL=C
pip install --upgrade pip

mkdir workspace
cd workspace
git clone https://github.com/nadbordrozd/text-top-model.git
cd text-top-model
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

bash get_data.sh
python prepare_polarity_dataset.py
python prepare_subjectivity_dataset.py
