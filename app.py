from unittest import result
import flask
from flask import render_template
import pickle
import pandas as pd
from sklearn.utils import resample


i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('./data/ml-100k/u.item', sep='|',
                    names=i_cols, encoding='latin-1')

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|',
                    names=u_cols, encoding='latin-1')

top = 10
max_user_id = users['user_id'].max()
app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        user_id = int(flask.request.form['user_id'])
        if user_id <=0 or user_id > max_user_id:
            return render_template('main.html', result=[])

        with open('./data/model.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)
            arr = loaded_model[user_id]
            idx = (-arr).argsort()[:top]
            recomendation = items.loc[items['movie id'].isin(
                idx)]['movie title']
            return render_template('main.html', result=recomendation.to_list())


if __name__ == '__main__':
    app.run()
