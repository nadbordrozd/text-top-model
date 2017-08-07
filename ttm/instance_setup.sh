#!/usr/bin/env bash

export LC_ALL=C
pip install --upgrade pip

pip install joblib
pip install unidecode
pip install hyperopt

pip uninstall keras
sudo pip uninstall keras
pip install --upgrade keras

mkdir workspace
cd workspace
git clone https://github.com/nadbordrozd/text-top-model.git
cd text-top-model
bash get_data.sh
python prepare_polarity_dataset.py
python prepare_subjectivity_dataset.py

sudo apt-get install unzip


sudo apt-get install zsh
sudo chsh -s /bin/zsh ubuntu
echo "export PATH=/usr/local/cuda/bin:$PATH" > ~/.zshrc
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
